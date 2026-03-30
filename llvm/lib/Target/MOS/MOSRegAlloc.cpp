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
  IndexedMap<sat::IntVar, VirtReg2IndexFunctor> EndVar;
  IndexedMap<MCPhysReg, VirtReg2IndexFunctor> Solution;
  bool Solved = false;

  static bool isValidVar(sat::IntVar V) { return V.index() >= 0; }

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

  // --- Helpers ---

  SmallVector<MCPhysReg> getClassPhysRegs(const TargetRegisterClass *RC);
  sat::IntVar makeRegVar(const TargetRegisterClass *RC);

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
    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " -> "
                      << TRI.getName(Solution[VReg]) << "\n");
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
    if (WideRC == CurRC)
      continue; // Already maximally wide.

    // Emit def copy: create a new narrow vreg for the def, COPY into
    // the original (wide) vreg. The original vreg remains the travelling
    // vreg that all uses see, symmetric with use copies.
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

    // Emit use copies.
    struct UseCopy {
      MachineInstr *MI;
      unsigned OpIdx;
      const TargetRegisterClass *RequiredRC;
    };
    SmallVector<UseCopy> UseCopies;
    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      if (MO.isUndef())
        continue;
      MachineInstr &MI = *MO.getParent();
      unsigned OpIdx = MO.getOperandNo();
      const auto *RequiredRC = MI.getRegClassConstraint(OpIdx, &TII, &TRI);
      if (!RequiredRC)
        continue;
      if (WideRC->hasSuperClassEq(RequiredRC))
        continue;
      UseCopies.push_back({&MI, OpIdx, RequiredRC});
    }
    for (const auto &UC : UseCopies) {
      Register NewVReg = MRI.createVirtualRegister(UC.RequiredRC);
      BuildMI(MBB, UC.MI->getIterator(), UC.MI->getDebugLoc(),
              TII.get(TargetOpcode::COPY), NewVReg)
          .addReg(Reg);
      UC.MI->getOperand(UC.OpIdx).setReg(NewVReg);

      LLVM_DEBUG(dbgs() << "  use-copy: " << printReg(Reg, &TRI) << " -> "
                        << printReg(NewVReg, &TRI) << " ("
                        << TRI.getRegClassName(UC.RequiredRC) << ") before "
                        << *UC.MI);
    }

    // Widen the vreg's class.
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
    if (!MI.isDebugInstr())
      ++N;
  NumIssueSlots = N;
  if (N == 0)
    return;

  int MaxSubPoint = 3 * (N - 1) + 2;

  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
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
    if (!MI.isDebugInstr())
      AllIssue.push_back(IssueVar[&MI]);
  Model.AddAllDifferent(AllIssue);

  // --- Dependency constraints ---

  // Data dependencies: issue(user) > issue(definer) for each vreg.
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(Reg))
      continue;
    MachineInstr *DefMI = MRI.getVRegDef(Reg);
    if (!DefMI)
      continue;
    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      MachineInstr *UseMI = MO.getParent();
      if (UseMI == DefMI)
        continue;
      Model.AddGreaterThan(IssueVar[UseMI], IssueVar[DefMI]);
    }
  }

  // Memory ordering: chain all memory-accessing instructions.
  MachineInstr *PrevMem = nullptr;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
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
    if (MI.isDebugInstr())
      continue;
    if (!MI.isTerminator())
      break;
    --TermStart;
  }
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
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
    if (MI.isDebugInstr())
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

/// Create one CP-SAT IntVar per allocatable vreg. The domain is the
/// vreg's MRI class, which insertCopies has already widened to
/// getLargestLegalSuperClass.
void RegAllocProblem::createVariables() {
  unsigned NumVRegs = MRI.getNumVirtRegs();
  RegVar.grow(Register::index2VirtReg(NumVRegs - 1));
  StartVar.grow(Register::index2VirtReg(NumVRegs - 1));
  EndVar.grow(Register::index2VirtReg(NumVRegs - 1));
  Solution.grow(Register::index2VirtReg(NumVRegs - 1));

  // Sub-point space: 3 sub-points per issue slot.
  int MaxSubPoint = std::max(1u, NumIssueSlots) * 3 - 1;

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg))
      continue;

    const TargetRegisterClass *RC = MRI.getRegClass(VReg);
    RegVar[VReg] = makeRegVar(RC);

    // Live range variables in sub-point space, half-open [start, end).
    // Each instruction has 3 sub-points: use, clobber, def.
    // - Normal def: start = DefPoint (3*issue+2)
    // - Earlyclobber def: start = UsePoint (3*issue)
    // - Use: end >= UsePoint+1 (alive through the use sub-point)
    StartVar[VReg] = Model.NewIntVar(Domain(0, MaxSubPoint));
    EndVar[VReg] = Model.NewIntVar(Domain(0, MaxSubPoint));

    // Channeling: start at def point, end past use point.
    MachineOperand &DefMO = *MRI.def_begin(VReg);
    MachineInstr *DefMI = DefMO.getParent();
    if (DefMO.isEarlyClobber())
      Model.AddEquality(StartVar[VReg], UsePoint[DefMI]);
    else
      Model.AddEquality(StartVar[VReg], DefPoint[DefMI]);

    for (MachineOperand &MO : MRI.use_nodbg_operands(VReg)) {
      MachineInstr *UseMI = MO.getParent();
      Model.AddGreaterThan(EndVar[VReg], UsePoint[UseMI]);
    }

    // end >= start
    Model.AddGreaterOrEqual(EndVar[VReg], StartVar[VReg]);

    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " ("
                      << TRI.getRegClassName(RC) << "): "
                      << getClassPhysRegs(RC).size() << " phys regs\n");
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
    if (MI.isDebugInstr())
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

  // --- Build NoOverlap2D rectangles ---
  //
  // For each vreg, for each allocatable register R in its class, for
  // each contiguous reg-unit range in R: create an optional rectangle
  // present iff RegVar == R.

  NoOverlap = Model.AddNoOverlap2D();
  unsigned RectCount = 0;

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (!isValidVar(RegVar[VReg]))
      continue;
    const TargetRegisterClass *RC = MRI.getRegClass(VReg);

    // Size variable for this vreg's live range (reused across all rectangles).
    sat::IntVar SizeVar = Model.NewIntVar(Domain(0, MaxSubPoint));
    Model.AddEquality(SizeVar,
                      LinearExpr(EndVar[VReg]) - LinearExpr(StartVar[VReg]));

    for (MCPhysReg R : *RC) {
      if (Reserved[R])
        continue;

      // present <=> RegVar[VReg] == R
      BoolVar PresentVar = Model.NewBoolVar();
      Model.AddEquality(RegVar[VReg], static_cast<int64_t>(R))
          .OnlyEnforceIf(PresentVar);
      Model.AddNotEqual(RegVar[VReg], static_cast<int64_t>(R))
          .OnlyEnforceIf(PresentVar.Not());

      // Split R's reg units into contiguous ranges.
      int RangeStart = -1, Prev = -1;
      for (MCRegUnit U : TRI.regunits(R)) {
        int UInt = static_cast<int>(U);
        if (Prev >= 0 && UInt != Prev + 1) {
          int Width = Prev - RangeStart + 1;
          IntervalVar XIv = Model.NewOptionalFixedSizeIntervalVar(
              RangeStart, Width, PresentVar);
          IntervalVar YIv = Model.NewOptionalIntervalVar(
              StartVar[VReg], SizeVar, EndVar[VReg], PresentVar);
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
        IntervalVar XIv = Model.NewOptionalFixedSizeIntervalVar(
            RangeStart, Width, PresentVar);
        IntervalVar YIv = Model.NewOptionalIntervalVar(
            StartVar[VReg], SizeVar, EndVar[VReg], PresentVar);
        NoOverlap->AddRectangle(XIv, YIv);
        ++RectCount;
      }
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
    if (MI.isDebugInstr())
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
  // Collect non-debug instructions with their solved positions.
  SmallVector<std::pair<unsigned, MachineInstr *>> Ordered;
  for (MachineInstr &MI : MBB)
    if (!MI.isDebugInstr())
      Ordered.push_back({IssueSolution[&MI], &MI});

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
