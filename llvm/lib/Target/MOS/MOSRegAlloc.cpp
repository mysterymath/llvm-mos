//===-- MOSRegAlloc.cpp - MOS Register Allocation -------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Constraint-programming register allocator for the MOS 6502, based on the
// Unison framework (Castañeda Lozano et al., TOPLAS 2019). Uses Google
// OR-Tools CP-SAT solver.
//
// Each virtual register gets a CP variable whose domain is the set of
// MCPhysReg values from its register class, plus start/end variables for
// its live range. Each instruction gets an issue variable (scheduling
// position). The solver simultaneously assigns registers AND schedules
// instructions, using temporal overlap constraints for interference.
// Copy extension inserts COPYs at def/use boundaries where register
// classes narrow, letting the solver decide whether to coalesce or copy.
//
// See MOSRegAllocRoadmap.md for the development roadmap.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSInstrCost.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_checker.h"
#include "ortools/sat/cp_model_solver.h"
#include "ortools/sat/sat_parameters.pb.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;
using namespace operations_research;
using namespace operations_research::sat;

namespace {

class RegAllocProblem;

// ============================================================================
// MOSRegAlloc pass
// ============================================================================

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<LiveVariablesWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setNoVRegs().setNoPHIs();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

// ============================================================================
// RegAllocProblem — CP-SAT model for a single basic block
// ============================================================================

/// Builds a constraint problem for integrated register allocation and
/// instruction scheduling over a single basic block. Uses OR-Tools CP-SAT
/// solver with NoOverlap2D for interference.
class RegAllocProblem {
  MachineFunction &MF;
  const MOSSubtarget &STI;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  MachineRegisterInfo &MRI;
  BitVector Reserved;

  // CP-SAT model and solution.
  CpModelBuilder Model;
  CpSolverResponse Response;

  // CP variables: IntVar is a lightweight value type (index into Model).
  // Default-constructed IntVar has index < 0 (invalid sentinel).
  IndexedMap<sat::IntVar, VirtReg2IndexFunctor> RegVar;
  IndexedMap<sat::IntVar, VirtReg2IndexFunctor> StartVar;

  // Per-lane EndVars: each sub-register lane of a vreg can die
  // independently. A lane is a single bit in the LaneBitmask. Keyed by
  // (vreg, lane bit position). Multiple subreg indices can map to the
  // same lane (e.g., sublo and sublsb both map to bit 1).
  DenseMap<std::pair<Register, unsigned>, sat::IntVar> LaneEndVar;

  IndexedMap<MCPhysReg, VirtReg2IndexFunctor> Solution;
  bool Solved = false;

  static bool isValidVar(sat::IntVar V) { return V.index() >= 0; }

  /// Get the EndVar for a specific lane bit of a vreg.
  sat::IntVar getEndVar(Register VReg, unsigned LaneBit) const {
    auto It = LaneEndVar.find({VReg, LaneBit});
    if (It != LaneEndVar.end())
      return It->second;
    return sat::IntVar(); // invalid
  }

  /// Get all (lane bit, EndVar) pairs for a vreg.
  void getLaneBitEndVars(
      Register VReg,
      SmallVectorImpl<std::pair<unsigned, sat::IntVar>> &Out) const {
    LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(VReg);
    for (unsigned Bit = 0; Bit < LaneBitmask::BitWidth; ++Bit) {
      if ((FullMask & LaneBitmask::getLane(Bit)).none())
        continue;
      auto It = LaneEndVar.find({VReg, Bit});
      if (It != LaneEndVar.end())
        Out.push_back({Bit, It->second});
    }
  }

  // Scheduling variables.
  // Each instruction has an issue slot [0, N). Within each slot, there
  // are 3 sub-points: use (3*issue), clobber (3*issue+1), def (3*issue+2).
  DenseMap<MachineInstr *, sat::IntVar> IssueVar;
  DenseMap<MachineInstr *, sat::IntVar> UsePoint;  // 3 * issue
  DenseMap<MachineInstr *, sat::IntVar> DefPoint;  // 3 * issue + 2
  DenseMap<MachineInstr *, unsigned> InstrIndex;
  DenseMap<MachineInstr *, unsigned> IssueSolution;
  unsigned NumIssueSlots = 0;

  // NoOverlap2D constraint for rectangle packing (initialized in buildConstraints).
  std::optional<NoOverlap2DConstraint> NoOverlap;

  // Cost mode for copy optimization.
  MOSInstrCost::Mode CostMode;

  // --- Coalescing ---
  //
  // For each COPY with two vreg operands, a BoolVar records whether
  // src and dst are at the same location (coalesced). When coalesced,
  // the dst segment's rectangle is suppressed, and the src segment's
  // EndVar extends to cover the dst's uses.
  //
  // CoalesceBool[DstVReg] = BoolVar for the incoming COPY edge.
  // Only populated for COPY destinations (non-structural).
  IndexedMap<BoolVar, VirtReg2IndexFunctor> CoalesceBool;

  // --- Structural op classification ---

  /// Returns true if MI is a constraint marker (not a real instruction).
  static bool isConstraintMarker(const MachineInstr &MI) {
    switch (MI.getOpcode()) {
    case TargetOpcode::REG_SEQUENCE:
    case TargetOpcode::EXTRACT_SUBREG:
    case TargetOpcode::INSERT_SUBREG:
      return true;
    default:
      return false;
    }
  }

  // --- Helpers ---

  SmallVector<MCPhysReg> getClassPhysRegs(const TargetRegisterClass *RC);
  sat::IntVar makeRegVar(const TargetRegisterClass *RC);
  void resolveUseToSegments(Register VReg, LaneBitmask LaneMask,
                            MachineInstr *RealUseMI,
                            SmallVectorImpl<BoolVar> &CoalescePath);

  // --- Model construction ---

  void insertCopies(MachineBasicBlock &MBB);
  void createIssueVariables(MachineBasicBlock &MBB);
  void createVariables();
  void buildConstraints(MachineBasicBlock &MBB);
  void configureObjective(MachineBasicBlock &MBB);

public:
  RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB);

  bool solve();
  bool solved() const { return Solved; }
  MCPhysReg getAssignment(Register VReg) const { return Solution[VReg]; }
  void applySchedule(MachineBasicBlock &MBB);
  void lowerCopies(MachineBasicBlock &MBB);
};

// ============================================================================
// MOSRegAlloc implementation
// ============================================================================

static void applySolution(MachineBasicBlock &MBB, RegAllocProblem &Problem);

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getRegInfo().getNumVirtRegs() == 0)
    return false;

  LLVM_DEBUG(dbgs() << "MOS RegAlloc: " << MF.getName() << " ("
                    << MF.getRegInfo().getNumVirtRegs() << " vregs)\n");

  for (MachineBasicBlock &MBB : MF) {
    LLVM_DEBUG(dbgs() << "  Block " << MBB.getName() << ": " << MBB.size()
                      << " instrs\n");

    // Each block gets a fresh CP-SAT model — no global state to reset.
    RegAllocProblem Problem(MF, MBB);
    if (!Problem.solve())
      report_fatal_error("MOS CP register allocator failed");

    applySolution(MBB, Problem);
  }

  MF.getRegInfo().clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

/// Apply the solved register assignments: reorder, lower COPYs, then
/// replace all virtual register operands with their assigned physical regs.
static void applySolution(MachineBasicBlock &MBB, RegAllocProblem &Problem) {
  Problem.applySchedule(MBB);
  Problem.lowerCopies(MBB);
  for (MachineInstr &MI : MBB)
    for (MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.getReg().isVirtual())
        if (MCPhysReg PhysReg = Problem.getAssignment(MO.getReg()))
          MO.setReg(PhysReg);
}

// ============================================================================
// RegAllocProblem implementation
// ============================================================================

RegAllocProblem::RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB)
    : MF(MF), STI(MF.getSubtarget<MOSSubtarget>()),
      TRI(*MF.getSubtarget().getRegisterInfo()),
      TII(*MF.getSubtarget().getInstrInfo()), MRI(MF.getRegInfo()),
      Reserved(TRI.getReservedRegs(MF)),
      CostMode(MOSInstrCost::getModeFor(MF)) {
  insertCopies(MBB);
  createIssueVariables(MBB);
  createVariables();
  buildConstraints(MBB);
  configureObjective(MBB);
}

bool RegAllocProblem::solve() {
  CpModelProto Proto = Model.Build();
  std::string Error = ValidateCpModel(Proto);
  if (!Error.empty())
    LLVM_DEBUG(dbgs() << "  Model validation: " << Error << "\n");
  LLVM_DEBUG({
    dbgs() << "  Model: " << Proto.variables_size() << " vars, "
           << Proto.constraints_size() << " constraints\n";
    // Dump NoOverlap2D info.
    for (int i = 0; i < Proto.constraints_size(); ++i) {
      const auto &ct = Proto.constraints(i);
      if (ct.has_no_overlap_2d()) {
        const auto &no2d = ct.no_overlap_2d();
        dbgs() << "  NoOverlap2D: " << no2d.x_intervals_size()
               << " x-intervals, " << no2d.y_intervals_size()
               << " y-intervals\n";
      }
    }
  });

  SatParameters Params;
  Params.set_num_workers(1);
  Params.set_log_search_progress(false);
  Response = SolveWithParameters(Proto, Params);

  if (Response.status() != CpSolverStatus::OPTIMAL &&
      Response.status() != CpSolverStatus::FEASIBLE) {
    LLVM_DEBUG(dbgs() << "  CP-SAT status: " << Response.status()
                      << " (" << CpSolverResponseStats(Response) << ")\n");
    return false;
  }

  Solved = true;
  LLVM_DEBUG(dbgs() << "  CP-SAT: "
                    << (Response.status() == CpSolverStatus::OPTIMAL
                            ? "OPTIMAL"
                            : "FEASIBLE")
                    << ", objective=" << Response.objective_value() << "\n");
  unsigned NumVRegs = MRI.getNumVirtRegs();
  Solution.grow(Register::index2VirtReg(NumVRegs ? NumVRegs - 1 : 0));
  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (!isValidVar(RegVar[VReg]))
      continue;
    Solution[VReg] =
        static_cast<MCPhysReg>(SolutionIntegerValue(Response, RegVar[VReg]));
    LLVM_DEBUG({
      dbgs() << "  " << printReg(VReg, &TRI) << " -> "
             << TRI.getName(Solution[VReg]);
      if (isValidVar(StartVar[VReg]))
        dbgs() << " [" << SolutionIntegerValue(Response, StartVar[VReg]);
      SmallVector<std::pair<unsigned, sat::IntVar>> LEVs;
      getLaneBitEndVars(VReg, LEVs);
      for (auto &[Bit, EV] : LEVs)
        dbgs() << ", end_" << Bit << "="
               << SolutionIntegerValue(Response, EV);
      if (isValidVar(StartVar[VReg]))
        dbgs() << ")";
      dbgs() << "\n";
    });
  }
  for (auto &[MI, Var] : IssueVar) {
    IssueSolution[MI] =
        static_cast<unsigned>(SolutionIntegerValue(Response, Var));
    LLVM_DEBUG(dbgs() << "  issue[" << InstrIndex[MI] << "] -> "
                      << IssueSolution[MI] << ": " << *MI);
  }
  return true;
}

// ============================================================================
/// Copy extension: insert COPY instructions at defs and uses
// ============================================================================

/// For each vreg, widen its class to getLargestLegalSuperClass. Then insert
/// COPYs to bridge any gap between the wide class and instruction constraints:
/// - After a def whose instruction constrains the output to a narrower class
/// - Before a use whose instruction constrains the input to a narrower class
void RegAllocProblem::insertCopies(MachineBasicBlock &MBB) {
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;

    const TargetRegisterClass *CurRC = MRI.getRegClass(Reg);
    const TargetRegisterClass *WideRC =
        TRI.getLargestLegalSuperClass(CurRC, MF);
    bool NeedsWidening = WideRC != CurRC;

    // Emit def copy: create a new narrow vreg for the def, COPY into
    // the original (wide) vreg. The original vreg remains the travelling
    // vreg that all uses see, symmetric with use copies.
    if (NeedsWidening) {
      MachineOperand &DefMO = *MRI.def_begin(Reg);
      MachineInstr &DefMI = *DefMO.getParent();
      if (!DefMI.isCopy()) {
        unsigned DefOpIdx = DefMO.getOperandNo();
        if (const auto *DefRC =
                DefMI.getRegClassConstraint(DefOpIdx, &TII, &TRI)) {
          if (!WideRC->hasSuperClassEq(DefRC)) {
            Register DefVReg = MRI.createVirtualRegister(DefRC);
            DefMO.setReg(DefVReg);
            BuildMI(MBB, std::next(DefMI.getIterator()), DefMI.getDebugLoc(),
                    TII.get(TargetOpcode::COPY), Reg)
                .addReg(DefVReg);

            LLVM_DEBUG(dbgs() << "  def-copy: " << printReg(DefVReg, &TRI)
                              << " (" << TRI.getRegClassName(DefRC) << ") -> "
                              << printReg(Reg, &TRI) << " after " << DefMI);
          }
        }
      }
    }

    // Emit use copies: bridge register class gaps, and freshen inputs
    // to structural ops (REG_SEQUENCE, EXTRACT_SUBREG, INSERT_SUBREG).
    // A use at a different sub-register width or in a structural op is
    // treated the same as a use at a narrower register class.
    struct UseCopy {
      MachineInstr *MI;
      unsigned OpIdx;
      const TargetRegisterClass *RC;
    };
    SmallVector<UseCopy> UseCopies;
    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      if (MO.isUndef())
        continue;
      MachineInstr &MI = *MO.getParent();
      unsigned OpIdx = MO.getOperandNo();

      // Unconditionally freshen structural op inputs. Widen to the
      // largest legal class — the structural constraints (element,
      // adjacency) guarantee the solver picks a valid register.
      if (isConstraintMarker(MI)) {
        const TargetRegisterClass *FreshRC =
            TRI.getLargestLegalSuperClass(MRI.getRegClass(Reg), MF);
        UseCopies.push_back({&MI, OpIdx, FreshRC});
        continue;
      }

      if (!NeedsWidening)
        continue;

      // Bridge register class gaps.
      const auto *RequiredRC = MI.getRegClassConstraint(OpIdx, &TII, &TRI);
      if (!RequiredRC)
        continue;
      if (WideRC->hasSuperClassEq(RequiredRC))
        continue;
      UseCopies.push_back({&MI, OpIdx, RequiredRC});
    }
    for (const auto &UC : UseCopies) {
      Register NewVReg = MRI.createVirtualRegister(UC.RC);
      BuildMI(MBB, UC.MI->getIterator(), UC.MI->getDebugLoc(),
              TII.get(TargetOpcode::COPY), NewVReg)
          .addReg(Reg);
      UC.MI->getOperand(UC.OpIdx).setReg(NewVReg);

      LLVM_DEBUG(dbgs() << "  use-copy: " << printReg(Reg, &TRI) << " -> "
                        << printReg(NewVReg, &TRI) << " ("
                        << TRI.getRegClassName(UC.RC) << ") before "
                        << *UC.MI);
    }

    // Widen the vreg's class.
    if (NeedsWidening)
      MRI.setRegClass(Reg, WideRC);
  }
}

// ============================================================================
// Variable and constraint construction
// ============================================================================

/// Create a CP-SAT IntVar whose domain is the allocatable physical
/// registers in RC.
sat::IntVar RegAllocProblem::makeRegVar(const TargetRegisterClass *RC) {
  SmallVector<MCPhysReg> PhysRegs = getClassPhysRegs(RC);
  std::vector<int64_t> Vals;
  for (MCPhysReg R : PhysRegs)
    Vals.push_back(static_cast<int64_t>(R));
  return Model.NewIntVar(Domain::FromValues(Vals));
}

/// Create issue (scheduling position) variables for each instruction.
/// Posts dependency constraints (data, memory, terminator, physreg).
void RegAllocProblem::createIssueVariables(MachineBasicBlock &MBB) {
  unsigned N = 0;
  for (MachineInstr &MI : MBB)
    if (!MI.isDebugInstr() && !isConstraintMarker(MI))
      ++N;
  NumIssueSlots = N;
  if (N == 0)
    return;

  int MaxSubPoint = 3 * (N - 1) + 2;

  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;
    InstrIndex[&MI] = IssueVar.size();
    sat::IntVar V = Model.NewIntVar(Domain(0, N - 1));
    IssueVar[&MI] = V;

    // Sub-points: use = 3*issue, def = 3*issue+2.
    sat::IntVar UP = Model.NewIntVar(Domain(0, MaxSubPoint));
    sat::IntVar DP = Model.NewIntVar(Domain(0, MaxSubPoint));
    Model.AddEquality(UP, 3 * LinearExpr(V));
    Model.AddEquality(DP, 3 * LinearExpr(V) + 2);
    UsePoint[&MI] = UP;
    DefPoint[&MI] = DP;
  }

  // All-different: each instruction at a unique position.
  std::vector<sat::IntVar> AllIssue;
  for (MachineInstr &MI : MBB)
    if (!MI.isDebugInstr() && !isConstraintMarker(MI))
      AllIssue.push_back(IssueVar[&MI]);
  Model.AddAllDifferent(AllIssue);

  // --- Dependency constraints ---

  // Helper: collect all real defining instructions for a vreg,
  // tracing through structural edges.
  auto collectRealDefs = [&](Register Reg,
                             SmallVectorImpl<MachineInstr *> &Defs) {
    SmallVector<Register> WorkList = {Reg};
    while (!WorkList.empty()) {
      Register R = WorkList.pop_back_val();
      MachineInstr *D = MRI.getVRegDef(R);
      if (!D)
        continue;
      if (!isConstraintMarker(*D)) {
        Defs.push_back(D);
      } else {
        for (MachineOperand &MO : D->uses())
          if (MO.isReg() && MO.getReg().isVirtual())
            WorkList.push_back(MO.getReg());
      }
    }
  };

  // Data dependencies: issue(user) > issue(definer) for each vreg.
  // For structural vregs, trace through to find real defining instructions.
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;

    SmallVector<MachineInstr *> RealDefs;
    collectRealDefs(Reg, RealDefs);
    if (RealDefs.empty())
      continue;

    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      MachineInstr *UseMI = MO.getParent();
      if (isConstraintMarker(*UseMI))
        continue;
      for (MachineInstr *RealDef : RealDefs) {
        if (UseMI == RealDef)
          continue;
        Model.AddGreaterThan(IssueVar[UseMI], IssueVar[RealDef]);
      }
    }
  }

  // Memory ordering: chain all memory-accessing instructions.
  MachineInstr *PrevMem = nullptr;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;
    if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall()) {
      if (PrevMem)
        Model.AddGreaterThan(IssueVar[&MI], IssueVar[PrevMem]);
      PrevMem = &MI;
    }
  }

  // Terminators: pin to end of block in original relative order.
  unsigned TermStart = N;
  for (MachineInstr &MI : reverse(MBB)) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;
    if (!MI.isTerminator())
      break;
    --TermStart;
  }
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;
    if (MI.isTerminator())
      Model.AddEquality(IssueVar[&MI],
                         static_cast<int64_t>(InstrIndex[&MI]));
    else
      Model.AddLessThan(IssueVar[&MI], static_cast<int64_t>(TermStart));
  }

  // Physical register ordering: def before use.
  DenseMap<MCPhysReg, MachineInstr *> PhysRegDef;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isUse() || !MO.getReg().isPhysical())
        continue;
      MCPhysReg Reg = MO.getReg().asMCReg();
      if (auto It = PhysRegDef.find(Reg); It != PhysRegDef.end())
        Model.AddGreaterThan(IssueVar[&MI], IssueVar[It->second]);
    }
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
        continue;
      PhysRegDef[MO.getReg().asMCReg()] = &MI;
    }
  }

  LLVM_DEBUG(dbgs() << "  " << N << " issue variables created\n");
}

/// Trace a use of VReg back through the copy-edge DAG, extending the
/// EndVars of the real segments that provide the value. LaneMask
/// specifies which lanes of VReg are needed. For structural ops, the
/// lane mask is transformed as it passes through sub-register edges.
void RegAllocProblem::resolveUseToSegments(
    Register VReg, LaneBitmask LaneMask, MachineInstr *RealUseMI,
    SmallVectorImpl<BoolVar> &CoalescePath) {
  MachineInstr *DefMI = MRI.getVRegDef(VReg);
  if (!DefMI)
    return;

  if (!isConstraintMarker(*DefMI)) {
    // Real vreg — extend matching lane EndVars to cover the real use.
    for (unsigned Bit = 0; Bit < LaneBitmask::BitWidth; ++Bit) {
      if ((LaneMask & LaneBitmask::getLane(Bit)).none())
        continue;
      sat::IntVar EV = getEndVar(VReg, Bit);
      if (!isValidVar(EV))
        continue;
      if (CoalescePath.empty()) {
        Model.AddGreaterThan(EV, UsePoint[RealUseMI]);
        LLVM_DEBUG(dbgs() << "  resolve: " << printReg(VReg, &TRI)
                          << " lane " << Bit << " extends to " << *RealUseMI);
      } else {
        Model.AddGreaterThan(EV, UsePoint[RealUseMI])
            .OnlyEnforceIf(CoalescePath);
        LLVM_DEBUG(dbgs() << "  resolve: " << printReg(VReg, &TRI)
                          << " lane " << Bit << " extends to " << *RealUseMI
                          << "    (conditional on " << CoalescePath.size()
                          << " coalesce vars)\n");
      }
    }

    // If this vreg is the dest of a COPY with a coalesce BoolVar,
    // also propagate to the COPY's source (conditional on coalescing).
    if (DefMI->isCopy() && CoalesceBool[VReg].index() >= 0) {
      Register SrcReg = DefMI->getOperand(1).getReg();
      if (SrcReg.isVirtual()) {
        CoalescePath.push_back(CoalesceBool[VReg]);
        resolveUseToSegments(SrcReg, LaneMask, RealUseMI, CoalescePath);
        CoalescePath.pop_back();
      }
    }
    return;
  }

  // Structural vreg — trace through the marker's inputs.
  switch (DefMI->getOpcode()) {
  case TargetOpcode::REG_SEQUENCE:
    // Each input contributes a lane. Only recurse on inputs whose
    // lane overlaps the requested mask.
    for (unsigned OpIdx = 1, E = DefMI->getNumOperands(); OpIdx < E;
         OpIdx += 2) {
      Register SrcReg = DefMI->getOperand(OpIdx).getReg();
      unsigned SubIdx = DefMI->getOperand(OpIdx + 1).getImm();
      LaneBitmask SubLane = TRI.getSubRegIndexLaneMask(SubIdx);
      if ((LaneMask & SubLane).none())
        continue;
      if (SrcReg.isVirtual()) {
        // The source vreg is a full-register value providing this lane.
        LaneBitmask SrcMask = MRI.getMaxLaneMaskForVReg(SrcReg);
        resolveUseToSegments(SrcReg, SrcMask, RealUseMI, CoalescePath);
      }
    }
    break;
  case TargetOpcode::EXTRACT_SUBREG: {
    // Output is an alias for one lane of the input. Transform the
    // lane mask to the input's lane space.
    unsigned SubIdx = DefMI->getOperand(2).getImm();
    LaneBitmask InputLane = TRI.getSubRegIndexLaneMask(SubIdx);
    resolveUseToSegments(DefMI->getOperand(1).getReg(), InputLane,
                         RealUseMI, CoalescePath);
    break;
  }
  case TargetOpcode::INSERT_SUBREG: {
    // Modified lane from sub-source, inherited lanes from super-source.
    unsigned SubIdx = DefMI->getOperand(3).getImm();
    LaneBitmask ModifiedLane = TRI.getSubRegIndexLaneMask(SubIdx);
    LaneBitmask InheritedLanes = LaneMask & ~ModifiedLane;
    LaneBitmask ModifiedMask = LaneMask & ModifiedLane;
    if (InheritedLanes.any())
      resolveUseToSegments(DefMI->getOperand(1).getReg(), InheritedLanes,
                           RealUseMI, CoalescePath);
    if (ModifiedMask.any()) {
      Register SubReg = DefMI->getOperand(2).getReg();
      LaneBitmask SubMask = MRI.getMaxLaneMaskForVReg(SubReg);
      resolveUseToSegments(SubReg, SubMask, RealUseMI, CoalescePath);
    }
    break;
  }
  }
}

/// Create one CP-SAT IntVar per real (non-structural) vreg.
/// Structural vregs don't get CP variables — their liveness is resolved
/// by tracing through the structural DAG to real segments.
void RegAllocProblem::createVariables() {
  unsigned NumVRegs = MRI.getNumVirtRegs();
  RegVar.grow(Register::index2VirtReg(NumVRegs - 1));
  StartVar.grow(Register::index2VirtReg(NumVRegs - 1));
  Solution.grow(Register::index2VirtReg(NumVRegs - 1));
  CoalesceBool.grow(Register::index2VirtReg(NumVRegs - 1));

  // Sub-point space: 3 sub-points per issue slot.
  int MaxSubPoint = std::max(1u, NumIssueSlots) * 3 - 1;

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg))
      continue;

    MachineInstr *DefMI = MRI.getVRegDef(VReg);
    if (!DefMI)
      continue;

    const TargetRegisterClass *RC = MRI.getRegClass(VReg);
    RegVar[VReg] = makeRegVar(RC);

    // Structural vregs get a RegVar (for cost model and solution
    // extraction) but no StartVar/EndVar/rectangles. Their RegVar
    // is constrained by the structural op in a later pass.
    if (isConstraintMarker(*DefMI))
      continue;

    // Live range: start at def point, one EndVar per sub-register lane.
    StartVar[VReg] = Model.NewIntVar(Domain(0, MaxSubPoint));

    MachineOperand &DefMO = *MRI.def_begin(VReg);
    if (DefMO.isEarlyClobber())
      Model.AddEquality(StartVar[VReg], UsePoint[DefMI]);
    else
      Model.AddEquality(StartVar[VReg], DefPoint[DefMI]);

    // Create one EndVar per lane bit. Each set bit in the vreg's lane
    // mask gets an independent EndVar.
    LaneBitmask FullMask = MRI.getMaxLaneMaskForVReg(VReg);
    SmallVector<unsigned> LaneBits;
    for (unsigned Bit = 0; Bit < LaneBitmask::BitWidth; ++Bit) {
      if ((FullMask & LaneBitmask::getLane(Bit)).none())
        continue;
      LaneBits.push_back(Bit);
      sat::IntVar EV = Model.NewIntVar(Domain(0, MaxSubPoint));
      LaneEndVar[{VReg, Bit}] = EV;
      // EndVar > StartVar ensures at least size 1. A def always
      // occupies its reg-units at the def point, even if never used.
      Model.AddGreaterThan(EV, StartVar[VReg]);
    }

    // Extend lane EndVars past each real use point. Uses by constraint
    // markers are resolved by tracing through the structural DAG.
    for (MachineOperand &MO : MRI.use_nodbg_operands(VReg)) {
      MachineInstr *UseMI = MO.getParent();
      if (isConstraintMarker(*UseMI))
        continue;
      // A full-register use extends all lanes.
      for (unsigned Bit : LaneBits) {
        sat::IntVar EV = getEndVar(VReg, Bit);
        if (isValidVar(EV))
          Model.AddGreaterThan(EV, UsePoint[UseMI]);
      }
    }

    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " ("
                      << TRI.getRegClassName(RC) << "): "
                      << getClassPhysRegs(RC).size() << " phys regs\n");
  }

  // Second pass: constrain structural vregs' RegVars based on their
  // defining instruction.
  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg))
      continue;
    MachineInstr *DefMI = MRI.getVRegDef(VReg);
    if (!DefMI || !isConstraintMarker(*DefMI))
      continue;

    switch (DefMI->getOpcode()) {
    case TargetOpcode::EXTRACT_SUBREG: {
      // RegVar[dst] == getSubReg(RegVar[src], subIdx)
      // Expressed via element constraint: map parent physreg → sub-physreg.
      Register SrcReg = DefMI->getOperand(1).getReg();
      unsigned SubIdx = DefMI->getOperand(2).getImm();
      if (!SrcReg.isVirtual() || !isValidVar(RegVar[SrcReg]))
        break;
      const TargetRegisterClass *SrcRC = MRI.getRegClass(SrcReg);
      SmallVector<MCPhysReg> SrcRegs = getClassPhysRegs(SrcRC);
      int Lo = *llvm::min_element(SrcRegs);
      int Hi = *llvm::max_element(SrcRegs);
      std::vector<int64_t> SubTable(Hi - Lo + 1, 0);
      for (MCPhysReg R : SrcRegs)
        if (MCPhysReg Sub = TRI.getSubReg(R, SubIdx))
          SubTable[R - Lo] = static_cast<int64_t>(Sub);
      Model.AddElement(LinearExpr(RegVar[SrcReg]) - Lo, SubTable,
                       RegVar[VReg]);
      break;
    }
    case TargetOpcode::INSERT_SUBREG: {
      // Output IS the super-source (tied constraint).
      Register SuperReg = DefMI->getOperand(1).getReg();
      if (SuperReg.isVirtual() && isValidVar(RegVar[SuperReg]))
        Model.AddEquality(RegVar[VReg], RegVar[SuperReg]);

      // The inserted value must land in the correct sub-register slot:
      // RegVar[inserted] == getSubReg(RegVar[output], subidx)
      Register InsertedReg = DefMI->getOperand(2).getReg();
      unsigned SubIdx = DefMI->getOperand(3).getImm();
      if (InsertedReg.isVirtual() && isValidVar(RegVar[InsertedReg]) &&
          isValidVar(RegVar[VReg])) {
        const TargetRegisterClass *OutRC = MRI.getRegClass(VReg);
        SmallVector<MCPhysReg> OutRegs = getClassPhysRegs(OutRC);
        int Lo = *llvm::min_element(OutRegs);
        int Hi = *llvm::max_element(OutRegs);
        std::vector<int64_t> SubTable(Hi - Lo + 1, 0);
        for (MCPhysReg R : OutRegs)
          if (MCPhysReg Sub = TRI.getSubReg(R, SubIdx))
            SubTable[R - Lo] = static_cast<int64_t>(Sub);
        Model.AddElement(LinearExpr(RegVar[VReg]) - Lo, SubTable,
                         RegVar[InsertedReg]);
      }
      break;
    }
    case TargetOpcode::REG_SEQUENCE:
      // REG_SEQUENCE: the output's RegVar is an RS register. Its
      // constraint comes from adjacency (posted in buildConstraints).
      // The RegVar domain already covers valid RS registers.
      // We also need: RegVar[dst].sublo == RegVar[lo_input] and
      // RegVar[dst].subhi == RegVar[hi_input]. This is handled by
      // element constraints similar to EXTRACT_SUBREG but in reverse.
      for (unsigned OpIdx = 1, E = DefMI->getNumOperands(); OpIdx < E;
           OpIdx += 2) {
        Register InputReg = DefMI->getOperand(OpIdx).getReg();
        unsigned SubIdx = DefMI->getOperand(OpIdx + 1).getImm();
        if (!InputReg.isVirtual() || !isValidVar(RegVar[InputReg]))
          continue;
        // getSubReg(RegVar[dst], SubIdx) == RegVar[input]
        const TargetRegisterClass *DstRC = MRI.getRegClass(VReg);
        SmallVector<MCPhysReg> DstRegs = getClassPhysRegs(DstRC);
        int Lo = *llvm::min_element(DstRegs);
        int Hi = *llvm::max_element(DstRegs);
        std::vector<int64_t> SubTable(Hi - Lo + 1, 0);
        for (MCPhysReg R : DstRegs)
          if (MCPhysReg Sub = TRI.getSubReg(R, SubIdx))
            SubTable[R - Lo] = static_cast<int64_t>(Sub);
        // Element: SubTable[RegVar[dst] - Lo] == RegVar[input]
        Model.AddElement(LinearExpr(RegVar[VReg]) - Lo, SubTable,
                         RegVar[InputReg]);
      }
      break;
    }
  }

  // Third pass: for each real instruction that uses a structural vreg,
  // resolve the use back through the DAG to extend real segments.
  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg))
      continue;
    MachineInstr *DefMI = MRI.getVRegDef(VReg);
    if (!DefMI || !isConstraintMarker(*DefMI))
      continue;

    // For each real use of this structural vreg, trace back to the
    // real segments that provide the value.
    for (MachineOperand &MO : MRI.use_nodbg_operands(VReg)) {
      MachineInstr *UseMI = MO.getParent();
      if (isConstraintMarker(*UseMI))
        continue;
      SmallVector<BoolVar> CoalescePath;
      // Use the full lane mask of the vreg — a real use of a structural
      // vreg keeps all its lanes alive.
      LaneBitmask UseMask = MRI.getMaxLaneMaskForVReg(VReg);
      resolveUseToSegments(VReg, UseMask, UseMI, CoalescePath);
    }
  }
}

/// Build interference constraints using CP-SAT NoOverlap2D.
/// Creates rectangles in (reg_unit x time) space for both vregs and
/// physreg segments.
void RegAllocProblem::buildConstraints(MachineBasicBlock &MBB) {
  unsigned NumVRegs = MRI.getNumVirtRegs();
  int MaxSubPoint = std::max(1u, NumIssueSlots) * 3 - 1;
  unsigned NumRegUnits = TRI.getNumRegUnits();

  // Tied operands: def and use must be the same register.
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.isTied())
        continue;
      Register DefReg = MO.getReg();
      if (!DefReg.isVirtual())
        continue;
      Register UseReg =
          MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo())).getReg();
      if (UseReg.isVirtual())
        Model.AddEquality(RegVar[DefReg], RegVar[UseReg]);
    }
  }

  // --- COPY coalescing BoolVars ---
  //
  // For each COPY between two vregs (both with RegVars), create a
  // BoolVar: Coal <=> (RegVar[Src] == RegVar[Dst]).
  // When coalesced, Dst's NoOverlap rectangle is suppressed.
  for (MachineInstr &MI : MBB) {
    if (!MI.isCopy())
      continue;
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    if (!DstReg.isVirtual() || !SrcReg.isVirtual())
      continue;
    if (!isValidVar(RegVar[DstReg]) || !isValidVar(RegVar[SrcReg]))
      continue;

    BoolVar Coal = Model.NewBoolVar();
    Model.AddEquality(RegVar[SrcReg], RegVar[DstReg]).OnlyEnforceIf(Coal);
    Model.AddNotEqual(RegVar[SrcReg], RegVar[DstReg])
        .OnlyEnforceIf(Coal.Not());
    CoalesceBool[DstReg] = Coal;

    LLVM_DEBUG(dbgs() << "  coalesce: " << printReg(SrcReg, &TRI) << " -> "
                      << printReg(DstReg, &TRI) << "\n");
  }

  // --- Build NoOverlap2D rectangles ---
  //
  // For each real vreg (with StartVar/EndVar), for each allocatable
  // register R in its class, for each contiguous reg-unit range in R:
  // create an optional rectangle present iff RegVar == R AND the
  // segment is active (not coalesced into its parent).

  NoOverlap = Model.AddNoOverlap2D();
  unsigned RectCount = 0;

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (!isValidVar(RegVar[VReg]) || !isValidVar(StartVar[VReg]))
      continue;
    const TargetRegisterClass *RC = MRI.getRegClass(VReg);

    // If this vreg is the destination of a COPY with a coalesce BoolVar,
    // its rectangle is suppressed when coalesced.
    bool HasCoalesce = CoalesceBool[VReg].index() >= 0;

    // Build a map from lane bit → (SizeVar, EndVar) for this vreg.
    SmallVector<std::pair<unsigned, sat::IntVar>> LaneEndVarList;
    getLaneBitEndVars(VReg, LaneEndVarList);
    DenseMap<unsigned, std::pair<sat::IntVar, sat::IntVar>> LaneInfo;
    for (auto &[Bit, EV] : LaneEndVarList) {
      sat::IntVar SV = Model.NewIntVar(Domain(0, MaxSubPoint));
      Model.AddEquality(SV, LinearExpr(EV) - LinearExpr(StartVar[VReg]));
      LaneInfo[Bit] = {SV, EV};
    }

    for (MCPhysReg R : *RC) {
      if (Reserved[R])
        continue;

      // present <=> RegVar == R (AND !Coal if coalescing applies)
      BoolVar RegMatch = Model.NewBoolVar();
      Model.AddEquality(RegVar[VReg], static_cast<int64_t>(R))
          .OnlyEnforceIf(RegMatch);
      Model.AddNotEqual(RegVar[VReg], static_cast<int64_t>(R))
          .OnlyEnforceIf(RegMatch.Not());

      BoolVar PresentVar;
      if (HasCoalesce) {
        PresentVar = Model.NewBoolVar();
        Model.AddImplication(PresentVar, RegMatch);
        Model.AddImplication(PresentVar, CoalesceBool[VReg].Not());
        Model.AddBoolOr({PresentVar, RegMatch.Not(), CoalesceBool[VReg]});
      } else {
        PresentVar = RegMatch;
      }

      // Walk R's reg-units with their lane masks. Group contiguous
      // reg-units that share the same EndVar into single rectangles.
      int RangeStart = -1, Prev = -1;
      sat::IntVar CurSizeVar, CurEV;
      unsigned CurLaneBit = ~0u;

      auto FlushRange = [&]() {
        if (RangeStart < 0)
          return;
        int Width = Prev - RangeStart + 1;
        IntervalVar XIv = Model.NewOptionalFixedSizeIntervalVar(
            RangeStart, Width, PresentVar);
        IntervalVar YIv = Model.NewOptionalIntervalVar(
            StartVar[VReg], CurSizeVar, CurEV, PresentVar);
        NoOverlap->AddRectangle(XIv, YIv);
        ++RectCount;
        RangeStart = -1;
        Prev = -1;
      };

      for (MCRegUnitMaskIterator RUMI(R, &TRI); RUMI.isValid(); ++RUMI) {
        auto [U, Mask] = *RUMI;
        // Find which lane bit this reg-unit belongs to.
        unsigned LaneBit = ~0u;
        for (auto &[Bit, Info] : LaneInfo) {
          if ((Mask & LaneBitmask::getLane(Bit)).any()) {
            LaneBit = Bit;
            break;
          }
        }
        if (LaneBit == ~0u) {
          // Reg-unit not covered by any of our lanes — flush and skip.
          FlushRange();
          continue;
        }

        auto &[SV, EV] = LaneInfo[LaneBit];
        int UInt = static_cast<int>(U);

        // If lane changed or reg-units aren't contiguous, flush.
        if (LaneBit != CurLaneBit ||
            (Prev >= 0 && UInt != Prev + 1)) {
          FlushRange();
        }

        if (RangeStart < 0) {
          RangeStart = UInt;
          CurSizeVar = SV;
          CurEV = EV;
          CurLaneBit = LaneBit;
        }
        Prev = UInt;
      }
      FlushRange();
    }
  }

  // Physical register segments: scan for physreg defs/uses, build
  // segments, add as always-present rectangles to NoOverlap2D.
  {
  struct PhysSegment {
    MCPhysReg Reg;
    sat::IntVar Start;
    sat::IntVar End;
  };
  SmallVector<PhysSegment> PhysSegments;
  DenseMap<MCPhysReg, unsigned> OpenSegment;

  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr() || isConstraintMarker(MI))
      continue;

    // Physreg uses: extend end to use point (alive through use phase).
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isUse() || !MO.getReg().isPhysical())
        continue;
      MCPhysReg Reg = MO.getReg().asMCReg();
      if (Reserved[Reg])
        continue;
      auto It = OpenSegment.find(Reg);
      if (It != OpenSegment.end())
        // end > UsePoint, i.e., end >= UsePoint + 1
        Model.AddGreaterOrEqual(PhysSegments[It->second].End,
                                LinearExpr(UsePoint[&MI]) + 1);
    }

    // Regmask clobbers: point at clobber sub-point (3*issue+1).
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isRegMask())
        continue;

      BitVector ClobberedUnits(NumRegUnits);
      for (unsigned R = 1; R < TRI.getNumRegs(); ++R) {
        if (Reserved[R] || !MO.clobbersPhysReg(R))
          continue;
        OpenSegment.erase(static_cast<MCPhysReg>(R));
        for (MCRegUnit U : TRI.regunits(static_cast<MCPhysReg>(R)))
          ClobberedUnits.set(static_cast<unsigned>(U));
      }

      // Clobber point = UsePoint + 1 = 3*issue + 1.
      sat::IntVar S = Model.NewIntVar(Domain(0, MaxSubPoint));
      sat::IntVar E = Model.NewIntVar(Domain(0, MaxSubPoint));
      Model.AddEquality(S, LinearExpr(UsePoint[&MI]) + 1);
      Model.AddEquality(E, LinearExpr(UsePoint[&MI]) + 1);
      // S == E, so size is 0 — point clobber.
      sat::IntVar ClobSize = Model.NewConstant(0);
      int RangeStart = -1, Prev = -1;
      for (int U = ClobberedUnits.find_first(); U != -1;
           U = ClobberedUnits.find_next(U)) {
        if (Prev >= 0 && U != Prev + 1) {
          int Width = Prev - RangeStart + 1;
          IntervalVar XIv =
              Model.NewFixedSizeIntervalVar(RangeStart, Width);
          IntervalVar YIv = Model.NewIntervalVar(S, ClobSize, E);
          NoOverlap->AddRectangle(XIv, YIv);
          ++RectCount;
          RangeStart = U;
        } else if (Prev < 0) {
          RangeStart = U;
        }
        Prev = U;
      }
      if (RangeStart >= 0) {
        int Width = Prev - RangeStart + 1;
        IntervalVar XIv =
            Model.NewFixedSizeIntervalVar(RangeStart, Width);
        IntervalVar YIv = Model.NewIntervalVar(S, ClobSize, E);
        NoOverlap->AddRectangle(XIv, YIv);
        ++RectCount;
      }
    }

    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
        continue;
      MCPhysReg Reg = MO.getReg().asMCReg();
      if (Reserved[Reg])
        continue;
      OpenSegment.erase(Reg);
      if (MO.isDead())
        continue;
      // Physreg def: start at def point (or use point if earlyclobber).
      sat::IntVar Start =
          MO.isEarlyClobber() ? UsePoint[&MI] : DefPoint[&MI];
      sat::IntVar S = Model.NewIntVar(Domain(0, MaxSubPoint));
      sat::IntVar E = Model.NewIntVar(Domain(0, MaxSubPoint));
      Model.AddEquality(S, Start);
      Model.AddGreaterOrEqual(E, Start);
      unsigned Idx = PhysSegments.size();
      PhysSegments.push_back({Reg, S, E});
      OpenSegment[Reg] = Idx;
    }
  }

  // Add physreg segments as always-present rectangles,
  // one per contiguous reg-unit range.
  for (const auto &Seg : PhysSegments) {
    sat::IntVar SegSize = Model.NewIntVar(Domain(0, MaxSubPoint));
    Model.AddEquality(SegSize,
                      LinearExpr(Seg.End) - LinearExpr(Seg.Start));
    int RangeStart = -1, Prev = -1;
    for (MCRegUnit U : TRI.regunits(Seg.Reg)) {
      int UInt = static_cast<int>(U);
      if (Prev >= 0 && UInt != Prev + 1) {
        int Width = Prev - RangeStart + 1;
        IntervalVar XIv =
            Model.NewFixedSizeIntervalVar(RangeStart, Width);
        IntervalVar YIv = Model.NewIntervalVar(Seg.Start, SegSize, Seg.End);
        NoOverlap->AddRectangle(XIv, YIv);
        ++RectCount;
        RangeStart = UInt;
      } else if (Prev < 0) {
        RangeStart = UInt;
      }
      Prev = UInt;
    }
    if (RangeStart >= 0) {
      int Width = Prev - RangeStart + 1;
      IntervalVar XIv =
          Model.NewFixedSizeIntervalVar(RangeStart, Width);
      IntervalVar YIv = Model.NewIntervalVar(Seg.Start, SegSize, Seg.End);
      NoOverlap->AddRectangle(XIv, YIv);
      ++RectCount;
    }
  } // end physreg segments
  }

  LLVM_DEBUG(dbgs() << "  NoOverlap2D: " << RectCount << " rectangles, "
                    << NumRegUnits << " reg units\n");

  // --- Adjacency constraints for REG_SEQUENCE ---
  //
  // For each REG_SEQUENCE composing an Imag16 from two Imag8 inputs:
  // RegVar(hi_input) == RegVar(lo_input) + 1, and lo must be even.
  for (MachineInstr &MI : MBB) {
    if (MI.getOpcode() != TargetOpcode::REG_SEQUENCE)
      continue;

    Register LoReg, HiReg;
    for (unsigned OpIdx = 1, E = MI.getNumOperands(); OpIdx < E;
         OpIdx += 2) {
      Register SrcReg = MI.getOperand(OpIdx).getReg();
      unsigned SubIdx = MI.getOperand(OpIdx + 1).getImm();
      if (SubIdx == MOS::sublo)
        LoReg = SrcReg;
      else if (SubIdx == MOS::subhi)
        HiReg = SrcReg;
    }

    if (LoReg.isValid() && HiReg.isValid() && LoReg.isVirtual() &&
        HiReg.isVirtual() && isValidVar(RegVar[LoReg]) &&
        isValidVar(RegVar[HiReg])) {
      // hi = lo + 1
      Model.AddEquality(RegVar[HiReg], LinearExpr(RegVar[LoReg]) + 1);

      // lo must be the sublo of a valid RS pair (even-aligned RC).
      Register OutReg = MI.getOperand(0).getReg();
      const TargetRegisterClass *OutRC = MRI.getRegClass(OutReg);
      std::vector<int64_t> ValidLo;
      for (MCPhysReg R : *OutRC) {
        if (Reserved[R])
          continue;
        if (MCPhysReg Lo = TRI.getSubReg(R, MOS::sublo))
          ValidLo.push_back(static_cast<int64_t>(Lo));
      }
      Model.AddLinearConstraint(RegVar[LoReg],
                                Domain::FromValues(ValidLo));

      LLVM_DEBUG(dbgs() << "  adjacency: " << printReg(LoReg, &TRI)
                        << " + 1 == " << printReg(HiReg, &TRI) << "\n");
    }
  }
}

/// Minimize total copy cost. For each COPY with at least one vreg
/// operand, the cost depends on the vreg's assignment (0 when coalesced).
void RegAllocProblem::configureObjective(MachineBasicBlock &MBB) {
  const auto &MOSTRI = static_cast<const MOSRegisterInfo &>(TRI);
  std::vector<sat::IntVar> CostVars;

  for (MachineInstr &MI : MBB) {
    if (!MI.isCopy())
      continue;
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();

    // Identify the variable operand (vreg with a CP variable) and the
    // set of possible physical registers for each side.
    sat::IntVar IndexVar;
    const TargetRegisterClass *IndexRC = nullptr;
    bool IndexIsSrc = true;
    auto DstPhysRegs = SmallVector<MCPhysReg>();
    auto SrcPhysRegs = SmallVector<MCPhysReg>();

    if (SrcReg.isVirtual() && isValidVar(RegVar[SrcReg]) &&
        DstReg.isPhysical()) {
      // vreg -> physreg: index by source vreg.
      IndexVar = RegVar[SrcReg];
      IndexRC = MRI.getRegClass(SrcReg);
      SrcPhysRegs = getClassPhysRegs(IndexRC);
      DstPhysRegs.push_back(static_cast<MCPhysReg>(DstReg.asMCReg()));
    } else if (DstReg.isVirtual() && isValidVar(RegVar[DstReg]) &&
               SrcReg.isPhysical()) {
      // physreg -> vreg: index by dest vreg.
      IndexVar = RegVar[DstReg];
      IndexRC = MRI.getRegClass(DstReg);
      IndexIsSrc = false;
      DstPhysRegs = getClassPhysRegs(IndexRC);
      SrcPhysRegs.push_back(static_cast<MCPhysReg>(SrcReg.asMCReg()));
    } else if (DstReg.isVirtual() && isValidVar(RegVar[DstReg]) &&
               SrcReg.isVirtual() && isValidVar(RegVar[SrcReg])) {
      // vreg -> vreg: cost depends on both variables. Use a table
      // constraint over (SrcVar, DstVar, CostVar) with one tuple per
      // (src_phys, dst_phys) pair.
      const TargetRegisterClass *SrcRC = MRI.getRegClass(SrcReg);
      const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);
      auto SrcRegs = getClassPhysRegs(SrcRC);
      auto DstRegs = getClassPhysRegs(DstRC);

      int MaxCost = 0;
      std::vector<std::vector<int64_t>> Tuples;
      for (MCPhysReg S : SrcRegs) {
        for (MCPhysReg D : DstRegs) {
          int Cost = 0;
          if (S != D) {
            const TargetRegisterClass *Clobber = nullptr;
            Cost = MOSTRI.copyCost(D, S, STI, &Clobber).value(CostMode);
          }
          Tuples.push_back({static_cast<int64_t>(S),
                            static_cast<int64_t>(D),
                            static_cast<int64_t>(Cost)});
          MaxCost = std::max(MaxCost, Cost);
        }
      }

      if (MaxCost == 0)
        continue;

      LLVM_DEBUG(dbgs() << "  copy cost (vreg-vreg, max=" << MaxCost
                        << "): " << MI);
      sat::IntVar CostVar = Model.NewIntVar(Domain(0, MaxCost));
      std::vector<sat::IntVar> TableVars = {RegVar[SrcReg], RegVar[DstReg],
                                            CostVar};
      TableConstraint TC = Model.AddAllowedAssignments(TableVars);
      for (const auto &Tuple : Tuples)
        TC.AddTuple(Tuple);
      CostVars.push_back(CostVar);
      continue;
    } else {
      continue;
    }

    // Build cost table indexed by IndexVar's physreg (vreg<->phys cases).
    int Lo = *llvm::min_element(getClassPhysRegs(IndexRC));
    int Hi = *llvm::max_element(getClassPhysRegs(IndexRC));

    std::vector<int64_t> CostTable(Hi - Lo + 1, 0);

    for (MCPhysReg IdxReg : *IndexRC) {
      if (Reserved[IdxReg])
        continue;
      int BestCost = INT_MAX;
      auto &OtherRegs = IndexIsSrc ? DstPhysRegs : SrcPhysRegs;
      for (MCPhysReg OtherReg : OtherRegs) {
        if (Reserved[OtherReg])
          continue;
        MCPhysReg Src = IndexIsSrc ? IdxReg : OtherReg;
        MCPhysReg Dst = IndexIsSrc ? OtherReg : IdxReg;
        if (Src == Dst) {
          BestCost = 0;
          break;
        }
        const TargetRegisterClass *Clobber = nullptr;
        int Cost = MOSTRI.copyCost(Dst, Src, STI, &Clobber).value(CostMode);
        BestCost = std::min(BestCost, Cost);
      }
      CostTable[IdxReg - Lo] = BestCost == INT_MAX ? 0 : BestCost;
    }

    int MaxCost = 0;
    for (size_t I = 0; I < CostTable.size(); ++I)
      MaxCost = std::max(MaxCost, static_cast<int>(CostTable[I]));
    if (MaxCost == 0)
      continue; // All paths are free, nothing to optimize.

    LLVM_DEBUG(dbgs() << "  copy cost (max=" << MaxCost << "): " << MI);
    sat::IntVar CostVar = Model.NewIntVar(Domain(0, MaxCost));
    Model.AddElement(LinearExpr(IndexVar) - Lo, CostTable, CostVar);
    CostVars.push_back(CostVar);
  }

  if (CostVars.empty())
    return;
  Model.Minimize(LinearExpr::Sum(CostVars));
}

// ============================================================================
/// Solution application
// ============================================================================

/// Reorder instructions in MBB according to the solved issue values.
/// Debug instructions are left in place relative to their neighbors.
void RegAllocProblem::applySchedule(MachineBasicBlock &MBB) {
  // Erase constraint markers — they have no physical realization.
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB))
    if (isConstraintMarker(MI))
      MI.eraseFromParent();

  // Collect real (non-debug) instructions with solved positions.
  SmallVector<std::pair<unsigned, MachineInstr *>> Ordered;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    Ordered.push_back({IssueSolution[&MI], &MI});
  }

  // Stable sort preserves original order for same-position instructions
  // (shouldn't happen with all_different, but defensive).
  llvm::stable_sort(Ordered,
                    [](const auto &A, const auto &B) { return A.first < B.first; });

  // Re-insert in solved order. Splice each instruction to the end,
  // building the new order incrementally.
  // First, find the insertion point (before any debug instrs at the end).
  for (auto &[Pos, MI] : Ordered)
    MBB.splice(MBB.end(), &MBB, MI->getIterator());

  LLVM_DEBUG({
    dbgs() << "  scheduled order:\n";
    for (auto &[Pos, MI] : Ordered)
      dbgs() << "    [" << Pos << "] " << *MI;
  });
}

/// Lower all COPY instructions: elide if src == dst (coalesced), expand
/// via copyPhysReg if different. When copyPhysReg creates intermediate
/// vregs, assign them physical registers by matching their class to the
/// clobber from copyCost.
void RegAllocProblem::lowerCopies(MachineBasicBlock &MBB) {
  const auto &MOSTRI = static_cast<const MOSRegisterInfo &>(TRI);

  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    if (!MI.isCopy())
      continue;
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();
    // Need at least one virtual operand to lower.
    if (!DstReg.isVirtual() && !SrcReg.isVirtual())
      continue;

    MCPhysReg DstPhys = DstReg.isVirtual()
                            ? Solution[DstReg]
                            : static_cast<MCPhysReg>(DstReg.asMCReg());
    MCPhysReg SrcPhys = SrcReg.isVirtual()
                            ? Solution[SrcReg]
                            : static_cast<MCPhysReg>(SrcReg.asMCReg());

    if (DstPhys == SrcPhys) {
      // Coalesced — elide the COPY.
      MI.eraseFromParent();
      continue;
    }

    // Expand via copyPhysReg. Track new vregs for clobber assignment.
    unsigned VRegsBefore = MRI.getNumVirtRegs();
    TII.copyPhysReg(MBB, MI.getIterator(), MI.getDebugLoc(), DstPhys,
                    SrcPhys, /*KillSrc=*/false);
    MI.eraseFromParent();

    unsigned NewVRegs = MRI.getNumVirtRegs() - VRegsBefore;
    if (NewVRegs > 0) {
      // Determine the clobber register from copyCost.
      const TargetRegisterClass *ClobberRC = nullptr;
      MOSTRI.copyCost(DstPhys, SrcPhys, STI, &ClobberRC);

      Solution.grow(Register::index2VirtReg(MRI.getNumVirtRegs() - 1));
      for (unsigned I = 0; I < NewVRegs; ++I) {
        Register VReg = Register::index2VirtReg(VRegsBefore + I);
        assert(ClobberRC && "copyPhysReg created vreg but no clobber expected");
        // Pick the first allocatable register in the clobber class.
        // copyCost guarantees this class has a valid register.
        MCPhysReg ClobberPhys = 0;
        for (MCPhysReg R : *ClobberRC)
          if (!Reserved[R]) {
            ClobberPhys = R;
            break;
          }
        assert(ClobberPhys && "No allocatable register in clobber class");
        Solution[VReg] = ClobberPhys;
        LLVM_DEBUG(dbgs() << "  clobber vreg " << printReg(VReg, &TRI)
                          << " -> " << TRI.getName(ClobberPhys) << "\n");
      }
    }
  }
}

// ============================================================================
// Helpers
// ============================================================================

SmallVector<MCPhysReg>
RegAllocProblem::getClassPhysRegs(const TargetRegisterClass *RC) {
  SmallVector<MCPhysReg> Regs;
  for (MCPhysReg Reg : *RC)
    if (!Reserved[Reg])
      Regs.push_back(Reg);
  return Regs;
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
