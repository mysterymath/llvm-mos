//===-- MOSRegAlloc.cpp - MOS Register Allocation -------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Constraint-programming register allocator for the MOS 6502, based on the
// Unison framework (Castañeda Lozano et al., TOPLAS 2019). Uses the Chuffed
// lazy clause generation solver.
//
// Each virtual register's CP variable domain is the set of MCPhysReg enum
// values from its register class. Interference between simultaneously-live
// vregs is enforced via != constraints (with alias-aware forbidden pairs for
// sub-register overlap). Copy extension inserts COPY instructions before
// uses that require a narrower class, letting the solver decide whether to
// coalesce or copy.
//
// See MOSRegAllocRoadmap.md for the development roadmap.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

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

#include "chuffed/core/engine.h"
#include "chuffed/core/options.h"
#include "chuffed/core/sat.h"
#include "chuffed/globals/globals.h"
#include "chuffed/ldsb/ldsb.h"
#include "chuffed/primitives/primitives.h"
#include "chuffed/support/vec.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

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
// RegAllocProblem — Chuffed CP model for a single basic block
// ============================================================================

/// Builds a constraint satisfaction problem over virtual register assignments
/// for a single basic block. The solver picks a physical register (MCPhysReg)
/// for each virtual register such that no two simultaneously-live vregs share
/// a physical register (accounting for aliasing and physical reg clobbers).
class RegAllocProblem : public Problem {
  MachineFunction &MF;
  const MOSSubtarget &STI;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  MachineRegisterInfo &MRI;
  BitVector Reserved;

  // CP variables and solution.
  IndexedMap<IntVar *, VirtReg2IndexFunctor> RegVar;
  IndexedMap<MCPhysReg, VirtReg2IndexFunctor> Solution;
  bool Solved = false;

  // Scheduling variables.
  DenseMap<MachineInstr *, IntVar *> IssueVar;
  DenseMap<MachineInstr *, unsigned> InstrIndex;

  // Liveness state for the backwards walk.
  SmallSet<Register, 16> Live;
  BitVector PhysLive;

  // Cost mode for copy optimization.
  MOSInstrCost::Mode CostMode;

  // --- Helpers ---

  SmallVector<MCPhysReg> getClassPhysRegs(const TargetRegisterClass *RC);
  bool classesCanOverlap(const TargetRegisterClass *A,
                         const TargetRegisterClass *B);
  SmallVector<std::pair<MCPhysReg, MCPhysReg>>
  getAliasingPairs(const TargetRegisterClass *RC1,
                   const TargetRegisterClass *RC2);
  bool physRegConflictsWithLive(MCPhysReg ClassReg);
  IntVar *makeRegVar(const TargetRegisterClass *RC);

  // --- Model construction ---

  void insertCopies(MachineBasicBlock &MBB);
  void createIssueVariables(MachineBasicBlock &MBB);
  void postInterference(Register VReg);
  void createVariables();
  void buildConstraints(MachineBasicBlock &MBB);
  void configureBranching();
  void configureObjective(MachineBasicBlock &MBB);

public:
  RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB);

  void recordSolution();
  bool solved() const { return Solved; }
  MCPhysReg getAssignment(Register VReg) const { return Solution[VReg]; }
  void lowerCopies(MachineBasicBlock &MBB);

  void print(std::ostream &) override {}
};

// ============================================================================
// MOSRegAlloc implementation
// ============================================================================

static void resetChuffedState();
static void applySolution(MachineBasicBlock &MBB, RegAllocProblem &Problem);

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getRegInfo().getNumVirtRegs() == 0)
    return false;

  LLVM_DEBUG(dbgs() << "MOS RegAlloc: " << MF.getName() << " ("
                    << MF.getRegInfo().getNumVirtRegs() << " vregs)\n");

  for (MachineBasicBlock &MBB : MF) {
    LLVM_DEBUG(dbgs() << "  Block " << MBB.getName() << ": " << MBB.size()
                      << " instrs\n");

    resetChuffedState();

    auto *Problem = new RegAllocProblem(MF, MBB);
    engine.setSolutionCallback([](::Problem *P) {
      static_cast<RegAllocProblem *>(P)->recordSolution();
    });
    engine.solve(Problem);

    if (!Problem->solved())
      report_fatal_error("MOS CP register allocator failed");

    applySolution(MBB, *Problem);
  }

  MF.getRegInfo().clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

/// Chuffed accumulates variables and propagators in global state across
/// solve() calls. Reset everything so each block gets a fresh solver.
static void resetChuffedState() {
  engine.~Engine();
  new (&engine) Engine();
  sat.~SAT();
  new (&sat) SAT();
  ldsb.~LDSB();
  new (&ldsb) LDSB();

  so.nof_solutions = 1;
  so.print_sol = false;
  so.verbosity = 0;
}

/// Apply the solved register assignments: lower COPYs, then replace all
/// virtual register operands with their assigned physical registers.
static void applySolution(MachineBasicBlock &MBB, RegAllocProblem &Problem) {
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
  configureBranching();
  configureObjective(MBB);
}

void RegAllocProblem::recordSolution() {
  Solved = true;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (!RegVar[VReg])
      continue;
    Solution[VReg] = static_cast<MCPhysReg>(RegVar[VReg]->getVal());
    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " -> "
                      << TRI.getName(Solution[VReg]) << "\n");
  }
}

// ============================================================================
// Copy extension: insert COPY instructions at defs and uses
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

/// Create a Chuffed IntVar whose domain is the allocatable physical
/// registers in RC.
IntVar *RegAllocProblem::makeRegVar(const TargetRegisterClass *RC) {
  SmallVector<MCPhysReg> PhysRegs = getClassPhysRegs(RC);
  int Lo = *llvm::min_element(PhysRegs);
  int Hi = *llvm::max_element(PhysRegs);
  IntVar *V = newIntVar(Lo, Hi);
  for (int Val = Lo; Val <= Hi; ++Val)
    if (!RC->contains(static_cast<MCPhysReg>(Val)) || Reserved[Val])
      int_rel(V, IRT_NE, Val);
  return V;
}

/// Create issue (scheduling position) variables for each instruction.
/// Posts dependency constraints and pins to original order for now.
void RegAllocProblem::createIssueVariables(MachineBasicBlock &MBB) {
  // Count non-debug instructions first, then create variables.
  unsigned N = 0;
  for (MachineInstr &MI : MBB)
    if (!MI.isDebugInstr())
      ++N;
  if (N == 0)
    return;

  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    InstrIndex[&MI] = IssueVar.size();
    IssueVar[&MI] = newIntVar(0, N - 1);
  }

  // All-different: each instruction at a unique position.
  vec<IntVar *> AllIssue;
  for (MachineInstr &MI : MBB)
    if (!MI.isDebugInstr())
      AllIssue.push(IssueVar[&MI]);
  all_different(AllIssue);

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
      // issue(use) > issue(def), i.e., issue(use) >= issue(def) + 1
      int_rel(IssueVar[UseMI], IRT_GE, IssueVar[DefMI], 1);
    }
  }

  // Memory ordering: chain all memory-accessing instructions in original
  // program order (conservative).
  MachineInstr *PrevMem = nullptr;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    if (MI.mayLoadOrStore() || MI.hasUnmodeledSideEffects() || MI.isCall()) {
      if (PrevMem)
        int_rel(IssueVar[&MI], IRT_GE, IssueVar[PrevMem], 1);
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
      int_rel(IssueVar[&MI], IRT_EQ, static_cast<int>(InstrIndex[&MI]));
    else
      int_rel(IssueVar[&MI], IRT_LT, static_cast<int>(TermStart));
  }

  // Physical register ordering: def before use in original order.
  // Scan forward, tracking last physreg def; post ordering to first use.
  DenseMap<MCPhysReg, MachineInstr *> PhysRegDef;
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    // Process uses first — they read the value from the prior def.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isUse() || !MO.getReg().isPhysical())
        continue;
      MCPhysReg Reg = MO.getReg().asMCReg();
      if (auto It = PhysRegDef.find(Reg); It != PhysRegDef.end())
        int_rel(IssueVar[&MI], IRT_GE, IssueVar[It->second], 1);
    }
    // Then process defs — update the tracking.
    for (const MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || !MO.getReg().isPhysical())
        continue;
      PhysRegDef[MO.getReg().asMCReg()] = &MI;
    }
  }

  // M5a: Pin all issue variables to original order.
  for (auto &[MI, Idx] : InstrIndex)
    int_rel(IssueVar[MI], IRT_EQ, static_cast<int>(Idx));

  LLVM_DEBUG(dbgs() << "  " << N << " issue variables created\n");
}

/// Create one Chuffed IntVar per allocatable vreg. The domain is the
/// vreg's MRI class, which insertCopies has already widened to
/// getLargestLegalSuperClass.
void RegAllocProblem::createVariables() {
  unsigned NumVRegs = MRI.getNumVirtRegs();
  RegVar.grow(Register::index2VirtReg(NumVRegs - 1));
  Solution.grow(Register::index2VirtReg(NumVRegs - 1));

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg))
      continue;

    const TargetRegisterClass *RC = MRI.getRegClass(VReg);
    RegVar[VReg] = makeRegVar(RC);
    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " ("
                      << TRI.getRegClassName(RC) << "): "
                      << getClassPhysRegs(RC).size() << " phys regs\n");
  }
}

/// Walk the block backwards to compute liveness and post interference
/// constraints. Also handles tied operands, earlyclobber, physical
/// register defs/uses, and regmasks.
void RegAllocProblem::buildConstraints(MachineBasicBlock &MBB) {
  PhysLive.resize(TRI.getNumRegs());

  for (MachineInstr &MI : reverse(MBB)) {
    if (MI.isDebugInstr())
      continue;

    // Phase 1: Defs — post interference with live set, then kill.
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg() || !MO.isDef() || MO.isEarlyClobber())
        continue;
      Register Reg = MO.getReg();
      if (Reg.isVirtual()) {
        postInterference(Reg);
        if (MO.isTied()) {
          Register UseReg =
              MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo())).getReg();
          assert(UseReg.isVirtual() &&
                 "Expected tied use to be virtual before regalloc");
          int_rel(RegVar[Reg], IRT_EQ, RegVar[UseReg], 0);
        }
        Live.erase(Reg);
      } else if (Reg.isPhysical()) {
        PhysLive.reset(Reg);
      }
    }

    // Phase 2: Regmasks — clobbered registers interfere with live vregs.
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isRegMask())
        continue;
      for (Register VReg : Live) {
        for (MCPhysReg PhysReg : *MRI.getRegClass(VReg))
          if (MO.clobbersPhysReg(PhysReg)) {
            int_rel(RegVar[VReg], IRT_NE, static_cast<int>(PhysReg));
            LLVM_DEBUG(dbgs() << "  regmask: " << printReg(VReg, &TRI)
                              << " != " << TRI.getName(PhysReg) << "\n");
          }
      }
      for (unsigned R = 1; R < TRI.getNumRegs(); ++R)
        if (MO.clobbersPhysReg(R))
          PhysLive.reset(R);
    }

    // Phase 3: Uses and earlyclobber defs.
    for (MachineOperand &MO : MI.operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isUse() && !MO.isUndef()) {
        Register Reg = MO.getReg();
        if (Reg.isVirtual()) {
          if (!Live.contains(Reg)) {
            postInterference(Reg);
            Live.insert(Reg);
          }
        } else if (Reg.isPhysical()) {
          PhysLive.set(Reg);
        }
      } else if (MO.isDef() && MO.isEarlyClobber()) {
        Register Reg = MO.getReg();
        if (Reg.isVirtual()) {
          postInterference(Reg);
          Live.erase(Reg);
        } else if (Reg.isPhysical()) {
          PhysLive.reset(Reg);
        }
      }
    }
  }
}

void RegAllocProblem::configureBranching() {
  // Branch on register variables first (most constrained), then issue.
  vec<IntVar *> RegVars;
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (RegVar[VReg])
      RegVars.push(RegVar[VReg]);
  }
  branch(RegVars, VAR_SIZE_MIN, VAL_MIN);

  vec<IntVar *> IssueVars;
  for (auto &[MI, Var] : IssueVar)
    IssueVars.push(Var);
  if (IssueVars.size() > 0)
    branch(IssueVars, VAR_SIZE_MIN, VAL_MIN);
}

/// Minimize total copy cost. For each COPY with at least one vreg
/// operand, the cost depends on the vreg's assignment (0 when coalesced).
void RegAllocProblem::configureObjective(MachineBasicBlock &MBB) {
  const auto &MOSTRI = static_cast<const MOSRegisterInfo &>(TRI);
  vec<IntVar *> CostVars;

  for (MachineInstr &MI : MBB) {
    if (!MI.isCopy())
      continue;
    Register DstReg = MI.getOperand(0).getReg();
    Register SrcReg = MI.getOperand(1).getReg();

    // Identify the variable operand (vreg with a CP variable) and the
    // set of possible physical registers for each side.
    IntVar *IndexVar = nullptr;
    const TargetRegisterClass *IndexRC = nullptr;
    bool IndexIsSrc = true;
    auto DstPhysRegs = SmallVector<MCPhysReg>();
    auto SrcPhysRegs = SmallVector<MCPhysReg>();

    if (SrcReg.isVirtual() && RegVar[SrcReg] && DstReg.isPhysical()) {
      // vreg → physreg: index by source vreg.
      IndexVar = RegVar[SrcReg];
      IndexRC = MRI.getRegClass(SrcReg);
      SrcPhysRegs = getClassPhysRegs(IndexRC);
      DstPhysRegs.push_back(static_cast<MCPhysReg>(DstReg.asMCReg()));
    } else if (DstReg.isVirtual() && RegVar[DstReg] &&
               SrcReg.isPhysical()) {
      // physreg → vreg: index by dest vreg.
      IndexVar = RegVar[DstReg];
      IndexRC = MRI.getRegClass(DstReg);
      IndexIsSrc = false;
      DstPhysRegs = getClassPhysRegs(IndexRC);
      SrcPhysRegs.push_back(static_cast<MCPhysReg>(SrcReg.asMCReg()));
    } else if (DstReg.isVirtual() && RegVar[DstReg] &&
               SrcReg.isVirtual() && RegVar[SrcReg]) {
      // vreg → vreg: cost depends on both variables. Use a table
      // constraint over (SrcVar, DstVar, CostVar) with one tuple per
      // (src_phys, dst_phys) pair.
      const TargetRegisterClass *SrcRC = MRI.getRegClass(SrcReg);
      const TargetRegisterClass *DstRC = MRI.getRegClass(DstReg);
      auto SrcRegs = getClassPhysRegs(SrcRC);
      auto DstRegs = getClassPhysRegs(DstRC);

      int MaxCost = 0;
      vec<vec<int>> Tuples;
      for (MCPhysReg S : SrcRegs) {
        for (MCPhysReg D : DstRegs) {
          int Cost = 0;
          if (S != D) {
            const TargetRegisterClass *Clobber = nullptr;
            Cost = MOSTRI.copyCost(D, S, STI, &Clobber).value(CostMode);
          }
          Tuples.push();
          Tuples.last().push(static_cast<int>(S));
          Tuples.last().push(static_cast<int>(D));
          Tuples.last().push(Cost);
          MaxCost = std::max(MaxCost, Cost);
        }
      }

      if (MaxCost == 0)
        continue;

      LLVM_DEBUG(dbgs() << "  copy cost (vreg-vreg, max=" << MaxCost
                        << "): " << MI);
      IntVar *CostVar = newIntVar(0, MaxCost);
      vec<IntVar *> TableVars;
      TableVars.push(RegVar[SrcReg]);
      TableVars.push(RegVar[DstReg]);
      TableVars.push(CostVar);
      table(TableVars, Tuples);
      CostVars.push(CostVar);
      continue;
    } else {
      continue;
    }

    // Build cost table indexed by IndexVar's physreg (vreg↔phys cases).
    int Lo = *llvm::min_element(getClassPhysRegs(IndexRC));
    int Hi = *llvm::max_element(getClassPhysRegs(IndexRC));

    vec<int> CostTable;
    CostTable.growTo(Hi - Lo + 1, 0);

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
    for (unsigned I = 0; I < CostTable.size(); ++I)
      MaxCost = std::max(MaxCost, CostTable[I]);
    if (MaxCost == 0)
      continue; // All paths are free, nothing to optimize.

    LLVM_DEBUG(dbgs() << "  copy cost (max=" << MaxCost << "): " << MI);
    IntVar *CostVar = newIntVar(0, MaxCost);
    array_int_element(IndexVar, CostTable, CostVar, Lo);
    CostVars.push(CostVar);
  }

  if (CostVars.size() == 0)
    return;
  IntVar *TotalCost = newIntVar(0, 10000);
  int_linear(CostVars, IRT_EQ, TotalCost);
  optimize(TotalCost, OPT_MIN);
}

// ============================================================================
// Solution application: lower COPYs
// ============================================================================

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

bool RegAllocProblem::classesCanOverlap(const TargetRegisterClass *A,
                                        const TargetRegisterClass *B) {
  for (MCPhysReg RegA : *A)
    for (MCPhysReg RegB : *B)
      if (TRI.regsOverlap(RegA, RegB))
        return true;
  return false;
}

/// Return all (phys1, phys2) pairs from (RC1, RC2) where the registers
/// overlap but have different enum values. These are the aliasing pairs
/// that a simple != constraint would miss (e.g., A and ALSB).
SmallVector<std::pair<MCPhysReg, MCPhysReg>>
RegAllocProblem::getAliasingPairs(const TargetRegisterClass *RC1,
                                   const TargetRegisterClass *RC2) {
  SmallVector<std::pair<MCPhysReg, MCPhysReg>> Pairs;
  for (MCPhysReg R1 : *RC1)
    for (MCPhysReg R2 : *RC2)
      if (R1 != R2 && TRI.regsOverlap(R1, R2))
        Pairs.push_back({R1, R2});
  return Pairs;
}

/// Check whether ClassReg (or any of its aliases) is physically live.
bool RegAllocProblem::physRegConflictsWithLive(MCPhysReg ClassReg) {
  for (MCRegAliasIterator AI(ClassReg, &TRI, /*IncludeSelf=*/true);
       AI.isValid(); ++AI)
    if (PhysLive[*AI])
      return true;
  return false;
}

/// Post != constraints between VReg and all live vregs/physregs it
/// conflicts with. Uses the widened def class for both the vreg and
/// live vregs to correctly model the wider domains.
void RegAllocProblem::postInterference(Register VReg) {
  const TargetRegisterClass *RC = MRI.getRegClass(VReg);
  for (Register LiveReg : Live) {
    if (LiveReg == VReg)
      continue;
    const TargetRegisterClass *LiveRC = MRI.getRegClass(LiveReg);
    if (!classesCanOverlap(RC, LiveRC))
      continue;
    int_rel(RegVar[VReg], IRT_NE, RegVar[LiveReg], 0);
    LLVM_DEBUG(dbgs() << "  interference: " << printReg(VReg, &TRI) << " != "
                      << printReg(LiveReg, &TRI) << "\n");
    // Forbid aliasing pairs that != misses (e.g., A vs ALSB).
    for (auto [P1, P2] : getAliasingPairs(RC, LiveRC)) {
      BoolView B = newBoolVar();
      int_rel_reif(RegVar[VReg], IRT_EQ, static_cast<int>(P1), B);
      int_rel_half_reif(RegVar[LiveReg], IRT_NE, static_cast<int>(P2), B);
    }
  }
  // Alias-aware physreg interference: exclude class members whose
  // aliases are physically live.
  for (MCPhysReg ClassReg : *RC) {
    if (Reserved[ClassReg])
      continue;
    if (!physRegConflictsWithLive(ClassReg))
      continue;
    int_rel(RegVar[VReg], IRT_NE, static_cast<int>(ClassReg));
    LLVM_DEBUG(dbgs() << "  interference: " << printReg(VReg, &TRI) << " != "
                      << TRI.getName(ClassReg) << "\n");
  }
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
