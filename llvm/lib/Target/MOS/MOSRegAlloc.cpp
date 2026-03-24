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
// vregs is enforced via != constraints, with aliasing checked via
// TRI.regsOverlap().
//
// See MOSRegAllocRoadmap.md for the development roadmap.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MOS.h"
#include "MOSInstrCost.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"

#include "llvm/ADT/DenseMap.h"
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
#include "chuffed/ldsb/ldsb.h"
#include "chuffed/primitives/primitives.h"
#include "chuffed/support/vec.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

class RegAllocProblem;

/// A use-copy opportunity: before a use that requires a narrower class
/// than the vreg's definition class, the solver can optionally insert a copy.
/// Modeled per-possibility (à la Unison): each possible (src, dst) pair is
/// an alternative with its own cost and clobber.
struct CopyOp {
  Register SrcVReg;                   ///< The original vreg being copied.
  IntVar *DstRegVar;                  ///< CP variable for the copy dest reg.
  BoolView Active;                    ///< Whether this copy is active.
  const TargetRegisterClass *DstRC;   ///< Destination (required) class.
  MachineInstr *MI;                   ///< Instruction containing the use.
  unsigned UseOpIdx;                  ///< Operand index of the use.

  /// Per-alternative cost: CostVar = CostTable[RegVar[SrcVReg]].
  /// The table maps each source physreg to the copy cost for that path
  /// (0 when the vreg is already in the required class).
  IntVar *CostVar = nullptr;

  /// Per-alternative clobbers. Each entry is a source physreg whose copy
  /// path requires an intermediate register, plus the clobber's CP variable
  /// and class. The clobber is only active when RegVar[SrcVReg] equals
  /// that source physreg AND the copy itself is active.
  struct Clobber {
    MCPhysReg SrcPhysReg;
    IntVar *Var;
    const TargetRegisterClass *RC;
    MCPhysReg SolvedReg = 0; ///< Populated by recordSolution.
  };
  SmallVector<Clobber, 1> Clobbers;

  // Solution values, populated by recordSolution so they survive
  // after engine.solve() returns (solver state may be stale).
  bool SolvedActive = false;
  MCPhysReg SolvedDstReg = 0;
};

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
  const MOSSubtarget &STI;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;
  MachineRegisterInfo &MRI;
  BitVector Reserved;

  // CP variables and solution.
  IndexedMap<IntVar *, VirtReg2IndexFunctor> RegVar;
  IndexedMap<MCPhysReg, VirtReg2IndexFunctor> Solution;
  bool Solved = false;

  // Liveness state for the backwards walk.
  SmallSet<Register, 16> Live;
  BitVector PhysLive;

  // Copy extension (M4): use-copies at use operands.
  MOSInstrCost::Mode CostMode;
  SmallVector<CopyOp> CopyOps;
  DenseMap<const MachineOperand *, unsigned> CopyOpForOperand;

  // --- Helpers ---

  SmallVector<MCPhysReg> getClassPhysRegs(const TargetRegisterClass *RC);
  bool classesCanOverlap(const TargetRegisterClass *A,
                         const TargetRegisterClass *B);
  SmallVector<std::pair<MCPhysReg, MCPhysReg>>
  getAliasingPairs(const TargetRegisterClass *RC1,
                   const TargetRegisterClass *RC2);
  bool physRegConflictsWithLive(MCPhysReg ClassReg);
  const TargetRegisterClass *getDefClass(Register VReg);
  IntVar *makeRegVar(const TargetRegisterClass *RC);

  // --- Model construction ---

  void postInterference(Register VReg);
  void createVariables();
  void identifyCopyOpportunities();
  void buildConstraints(MachineBasicBlock &MBB);
  void postCopyConstraints();
  void configureBranching();
  void configureObjective();

public:
  RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB);

  void recordSolution();
  bool solved() const { return Solved; }
  MCPhysReg getAssignment(Register VReg) const { return Solution[VReg]; }
  MCPhysReg getOperandReg(const MachineOperand &MO) const;
  void emitCopies();

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

/// Apply the solved register assignments: emit copy instructions and replace
/// all virtual register operands with their assigned physical registers.
static void applySolution(MachineBasicBlock &MBB, RegAllocProblem &Problem) {
  Problem.emitCopies();
  for (MachineInstr &MI : MBB)
    for (MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.getReg().isVirtual())
        if (MCPhysReg PhysReg = Problem.getOperandReg(MO))
          MO.setReg(PhysReg);
}

// ============================================================================
// RegAllocProblem implementation
// ============================================================================

RegAllocProblem::RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB)
    : STI(MF.getSubtarget<MOSSubtarget>()),
      TRI(*MF.getSubtarget().getRegisterInfo()),
      TII(*MF.getSubtarget().getInstrInfo()), MRI(MF.getRegInfo()),
      Reserved(TRI.getReservedRegs(MF)),
      CostMode(MOSInstrCost::getModeFor(MF)) {
  createVariables();
  identifyCopyOpportunities();
  buildConstraints(MBB);
  postCopyConstraints();
  configureBranching();
  configureObjective();
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
  for (CopyOp &CO : CopyOps) {
    CO.SolvedActive = CO.Active.isTrue();
    CO.SolvedDstReg = static_cast<MCPhysReg>(CO.DstRegVar->getVal());
    for (auto &Clob : CO.Clobbers)
      Clob.SolvedReg = static_cast<MCPhysReg>(Clob.Var->getVal());
  }
  LLVM_DEBUG(for (const CopyOp &CO : CopyOps) {
    dbgs() << "  copy " << printReg(CO.SrcVReg, &TRI) << " -> "
           << TRI.getRegClassName(CO.DstRC) << ": "
           << (CO.SolvedActive ? "active" : "coalesced") << "\n";
  });
}

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

/// Get the widest register class for a vreg based on its defining
/// instruction's output constraint. Falls back to MRI.getRegClass() for
/// COPY or other instructions without a fixed output class.
const TargetRegisterClass *RegAllocProblem::getDefClass(Register VReg) {
  const TargetRegisterClass *NarrowRC = MRI.getRegClass(VReg);

  MachineInstr *DefMI = MRI.getVRegDef(VReg);
  if (!DefMI)
    return NarrowRC;
  for (unsigned I = 0, E = DefMI->getNumOperands(); I < E; ++I) {
    const MachineOperand &MO = DefMI->getOperand(I);
    if (!MO.isReg() || !MO.isDef() || MO.getReg() != VReg)
      continue;
    if (const auto *RC = DefMI->getRegClassConstraint(I, &TII, &TRI))
      return RC;
    break;
  }
  return NarrowRC;
}

/// Create one Chuffed IntVar per allocatable vreg. The domain is the
/// defining instruction's output class (which may be wider than the
/// parser-narrowed class from MRI), enabling copy extension.
void RegAllocProblem::createVariables() {
  unsigned NumVRegs = MRI.getNumVirtRegs();
  RegVar.grow(Register::index2VirtReg(NumVRegs - 1));
  Solution.grow(Register::index2VirtReg(NumVRegs - 1));

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg) || MRI.def_empty(VReg))
      continue;

    const TargetRegisterClass *RC = getDefClass(VReg);
    RegVar[VReg] = makeRegVar(RC);
    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " ("
                      << TRI.getRegClassName(RC) << "): "
                      << getClassPhysRegs(RC).size() << " phys regs\n");
  }
}

/// For each vreg, check if any use operand's instruction requires a
/// narrower register class than the vreg's (widened) definition class.
/// For each such use, create a CopyOp with a CP variable in the
/// required class and a BoolVar controlling whether the copy is active.
void RegAllocProblem::identifyCopyOpportunities() {
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I < E; ++I) {
    Register Reg = Register::index2VirtReg(I);
    if (!RegVar[Reg])
      continue;
    const TargetRegisterClass *DefRC = getDefClass(Reg);

    for (MachineOperand &MO : MRI.use_nodbg_operands(Reg)) {
      if (MO.isUndef())
        continue;
      MachineInstr &MI = *MO.getParent();
      unsigned OpIdx = MO.getOperandNo();

      const auto *RequiredRC = MI.getRegClassConstraint(OpIdx, &TII, &TRI);
      if (!RequiredRC)
        continue;
      // If the def class already fits within the required class, no copy
      // is ever needed — every register the def can produce is acceptable.
      if (DefRC->hasSuperClassEq(RequiredRC))
        continue;

      IntVar *DstVar = makeRegVar(RequiredRC);
      BoolView Active = newBoolVar();

      // Build per-alternative cost table and clobber list.
      // For each source physreg, compute cost and clobber for the best
      // copy to any dest in RequiredRC. Pick the cheapest dest.
      const auto &MOSTRI = static_cast<const MOSRegisterInfo &>(TRI);
      SmallVector<MCPhysReg> SrcRegs = getClassPhysRegs(DefRC);
      int SrcLo = *llvm::min_element(SrcRegs);
      int SrcHi = *llvm::max_element(SrcRegs);

      // Cost table indexed by (src_physreg - SrcLo).
      vec<int> CostTable;
      CostTable.growTo(SrcHi - SrcLo + 1, 0);

      SmallVector<CopyOp::Clobber, 1> Clobbers;
      SmallPtrSet<const TargetRegisterClass *, 2> SeenClobberRCs;

      for (MCPhysReg Src : *DefRC) {
        if (Reserved[Src])
          continue;
        // Find cheapest copy from Src to any register in RequiredRC.
        int BestCost = INT_MAX;
        const TargetRegisterClass *BestClobber = nullptr;
        for (MCPhysReg Dst : *RequiredRC) {
          if (Reserved[Dst])
            continue;
          if (Src == Dst) {
            BestCost = 0;
            BestClobber = nullptr;
            break; // Can't do better than free.
          }
          const TargetRegisterClass *ThisClobber = nullptr;
          int Cost = MOSTRI.copyCost(Dst, Src, STI, &ThisClobber)
                         .value(CostMode);
          if (Cost < BestCost) {
            BestCost = Cost;
            BestClobber = ThisClobber;
          }
        }
        CostTable[Src - SrcLo] = BestCost == INT_MAX ? 0 : BestCost;
        if (BestClobber && SeenClobberRCs.insert(BestClobber).second)
          Clobbers.push_back({Src, makeRegVar(BestClobber), BestClobber});
      }

      // CostVar = CostTable[RegVar[SrcVReg] - SrcLo]
      int MaxCost = 0;
      for (unsigned I = 0; I < CostTable.size(); ++I)
        MaxCost = std::max(MaxCost, CostTable[I]);
      IntVar *CostVar = newIntVar(0, MaxCost);
      array_int_element(RegVar[Reg], CostTable, CostVar, SrcLo);

      CopyOps.push_back({Reg, DstVar, Active, RequiredRC, &MI, OpIdx,
                          CostVar, std::move(Clobbers)});
      CopyOpForOperand[&MO] = CopyOps.size() - 1;

      LLVM_DEBUG({
        dbgs() << "  copy opportunity: " << printReg(Reg, &TRI) << " ("
               << TRI.getRegClassName(DefRC) << " -> "
               << TRI.getRegClassName(RequiredRC) << ")";
        for (const auto &C : CopyOps.back().Clobbers)
          dbgs() << " clobber(" << TRI.getName(C.SrcPhysReg) << "):"
                 << TRI.getRegClassName(C.RC);
        dbgs() << " at " << MI;
      });
    }
  }
}

/// Walk the block backwards to compute liveness and post interference
/// constraints. Also handles tied operands, earlyclobber, physical
/// register defs/uses, regmasks, and block liveins.
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
          unsigned TiedIdx = MI.findTiedOperandIdx(MO.getOperandNo());
          const MachineOperand &TiedUse = MI.getOperand(TiedIdx);
          Register UseReg = TiedUse.getReg();
          assert(UseReg.isVirtual() &&
                 "Expected tied use to be virtual before regalloc");
          auto It = CopyOpForOperand.find(&TiedUse);
          if (It != CopyOpForOperand.end()) {
            // Tied use has a copy opportunity: conditional tie.
            // active  → def = copy dest (the copy output is what the
            //           instruction reads/writes)
            // !active → def = vreg (direct use, no copy)
            CopyOp &CO = CopyOps[It->second];
            int_rel_half_reif(RegVar[Reg], IRT_EQ, CO.DstRegVar, CO.Active);
            int_rel_half_reif(RegVar[Reg], IRT_EQ, RegVar[UseReg],
                              ~CO.Active);
          } else {
            int_rel(RegVar[Reg], IRT_EQ, RegVar[UseReg], 0);
          }
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
        for (MCPhysReg PhysReg : *getDefClass(VReg))
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

    // Phase 3: Uses and earlyclobber defs (simultaneous — earlyclobber
    // writes before reads, so they interfere with each other).
    // Collect copy ops; their interference is posted at the end of the
    // phase so the Live set includes ALL uses of this instruction.
    SmallVector<CopyOp *> PendingCopyOps;
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
          auto It = CopyOpForOperand.find(&MO);
          if (It != CopyOpForOperand.end())
            PendingCopyOps.push_back(&CopyOps[It->second]);
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

    // Post copy temp and clobber interference now that Live includes
    // all uses.
    for (CopyOp *CO : PendingCopyOps) {
      // Helper: post conditional interference for a copy-related IntVar.
      auto PostConditionalInterference = [&](IntVar *Var,
                                             const TargetRegisterClass *RC) {
        for (Register LiveReg : Live) {
          const TargetRegisterClass *LiveRC = getDefClass(LiveReg);
          if (!classesCanOverlap(RC, LiveRC))
            continue;
          int_rel_half_reif(Var, IRT_NE, RegVar[LiveReg], CO->Active);
          for (auto [P1, P2] : getAliasingPairs(RC, LiveRC)) {
            BoolView BEq = newBoolVar();
            int_rel_reif(Var, IRT_EQ, static_cast<int>(P1), BEq);
            BoolView BConj = newBoolVar();
            bool_rel(CO->Active, BRT_AND, BEq, BConj);
            int_rel_half_reif(RegVar[LiveReg], IRT_NE,
                              static_cast<int>(P2), BConj);
          }
        }
        for (MCPhysReg ClassReg : *RC) {
          if (Reserved[ClassReg])
            continue;
          if (!physRegConflictsWithLive(ClassReg))
            continue;
          int_rel_half_reif(Var, IRT_NE, static_cast<int>(ClassReg),
                            CO->Active);
        }
      };

      PostConditionalInterference(CO->DstRegVar, CO->DstRC);
      // Clobber interference: conditional on Active AND vreg == SrcPhysReg.
      for (const auto &Clob : CO->Clobbers) {
        BoolView IsSrc = newBoolVar();
        int_rel_reif(RegVar[CO->SrcVReg], IRT_EQ,
                     static_cast<int>(Clob.SrcPhysReg), IsSrc);
        BoolView ClobActive = newBoolVar();
        bool_rel(CO->Active, BRT_AND, IsSrc, ClobActive);
        // Use ClobActive instead of CO->Active for this clobber.
        for (Register LiveReg : Live) {
          const TargetRegisterClass *LiveRC = getDefClass(LiveReg);
          if (!classesCanOverlap(Clob.RC, LiveRC))
            continue;
          int_rel_half_reif(Clob.Var, IRT_NE, RegVar[LiveReg], ClobActive);
          for (auto [P1, P2] : getAliasingPairs(Clob.RC, LiveRC)) {
            BoolView BEq = newBoolVar();
            int_rel_reif(Clob.Var, IRT_EQ, static_cast<int>(P1), BEq);
            BoolView BConj = newBoolVar();
            bool_rel(ClobActive, BRT_AND, BEq, BConj);
            int_rel_half_reif(RegVar[LiveReg], IRT_NE,
                              static_cast<int>(P2), BConj);
          }
        }
        for (MCPhysReg ClassReg : *Clob.RC) {
          if (Reserved[ClassReg])
            continue;
          if (!physRegConflictsWithLive(ClassReg))
            continue;
          int_rel_half_reif(Clob.Var, IRT_NE, static_cast<int>(ClassReg),
                            ClobActive);
        }
      }
    }
  }
}

void RegAllocProblem::configureBranching() {
  vec<IntVar *> BranchVars;
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (RegVar[VReg])
      BranchVars.push(RegVar[VReg]);
  }
  for (const CopyOp &CO : CopyOps) {
    BranchVars.push(CO.DstRegVar);
    for (const auto &Clob : CO.Clobbers)
      BranchVars.push(Clob.Var);
  }
  branch(BranchVars, VAR_SIZE_MIN, VAL_MIN);
}

/// Minimize total copy cost across all copy operations.
void RegAllocProblem::configureObjective() {
  if (CopyOps.empty())
    return;
  vec<IntVar *> CostVars;
  for (const CopyOp &CO : CopyOps)
    CostVars.push(CO.CostVar);
  IntVar *TotalCost = newIntVar(0, 10000);
  int_linear(CostVars, IRT_EQ, TotalCost);
  optimize(TotalCost, OPT_MIN);
}

// --- Helpers ---

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
  const TargetRegisterClass *RC = getDefClass(VReg);
  for (Register LiveReg : Live) {
    if (LiveReg == VReg)
      continue;
    const TargetRegisterClass *LiveRC = getDefClass(LiveReg);
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

/// Post constraints for each copy operation that don't depend on the
/// backwards walk: conditional class constraints and dominance breaking.
void RegAllocProblem::postCopyConstraints() {
  for (CopyOp &CO : CopyOps) {
    const TargetRegisterClass *DefRC = getDefClass(CO.SrcVReg);

    // When copy is inactive, the vreg itself must be in the required class.
    // For each phys reg in the vreg's domain that's NOT in the required class:
    //   !active → vreg != that_phys_reg
    for (MCPhysReg PhysReg : *DefRC)
      if (!CO.DstRC->contains(PhysReg) && !Reserved[PhysReg])
        int_rel_half_reif(RegVar[CO.SrcVReg], IRT_NE,
                          static_cast<int>(PhysReg), ~CO.Active);

    // Dominance breaking: no-op copies must be inactive.
    //   active → vreg != copy_dest
    int_rel_half_reif(RegVar[CO.SrcVReg], IRT_NE, CO.DstRegVar, CO.Active);
  }
}

/// Return the physical register for an operand, considering active copies.
/// If the operand has an active copy, return the copy destination register;
/// otherwise return the vreg's assigned register.
MCPhysReg RegAllocProblem::getOperandReg(const MachineOperand &MO) const {
  auto It = CopyOpForOperand.find(&MO);
  if (It != CopyOpForOperand.end()) {
    const CopyOp &CO = CopyOps[It->second];
    if (CO.SolvedActive)
      return CO.SolvedDstReg;
  }
  return getAssignment(MO.getReg());
}

/// Emit copy instructions for all active copy operations. When
/// copyPhysReg creates intermediate vregs (e.g., A for X→Y on vanilla
/// 6502), assign them the physical registers from the clobber model.
void RegAllocProblem::emitCopies() {
  for (const CopyOp &CO : CopyOps) {
    if (!CO.SolvedActive)
      continue;

    MCPhysReg SrcPhysReg = Solution[CO.SrcVReg];
    MCPhysReg DstPhysReg = CO.SolvedDstReg;

    unsigned VRegsBefore = MRI.getNumVirtRegs();
    MachineBasicBlock &MBB = *CO.MI->getParent();
    TII.copyPhysReg(MBB, CO.MI->getIterator(), CO.MI->getDebugLoc(),
                    DstPhysReg, SrcPhysReg, /*KillSrc=*/false);

    // Assign physical registers to any vregs that copyPhysReg created
    // for intermediates, using values recorded by recordSolution.
    unsigned NewVRegs = MRI.getNumVirtRegs() - VRegsBefore;
    if (NewVRegs > 0) {
      Solution.grow(Register::index2VirtReg(MRI.getNumVirtRegs() - 1));
      for (unsigned I = 0; I < NewVRegs; ++I) {
        Register VReg = Register::index2VirtReg(VRegsBefore + I);
        const TargetRegisterClass *RC = MRI.getRegClass(VReg);
        MCPhysReg ClobberPhys = 0;
        for (const auto &Clob : CO.Clobbers)
          if (Clob.RC == RC) {
            ClobberPhys = Clob.SolvedReg;
            break;
          }
        assert(ClobberPhys &&
               "copyPhysReg created a vreg with no matching clobber");
        Solution[VReg] = ClobberPhys;
        LLVM_DEBUG(dbgs() << "  clobber vreg " << printReg(VReg, &TRI)
                          << " -> " << TRI.getName(ClobberPhys) << "\n");
      }
    }

    LLVM_DEBUG(dbgs() << "  emit copy: " << TRI.getName(SrcPhysReg) << " -> "
                      << TRI.getName(DstPhysReg) << " before " << *CO.MI);
  }
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
