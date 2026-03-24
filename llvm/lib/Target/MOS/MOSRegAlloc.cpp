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

#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
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
  const TargetRegisterInfo &TRI;
  MachineRegisterInfo &MRI;
  BitVector Reserved;

  // CP variables and solution.
  IndexedMap<IntVar *, VirtReg2IndexFunctor> RegVar;
  IndexedMap<MCPhysReg, VirtReg2IndexFunctor> Solution;
  bool Solved = false;

  // Liveness state for the backwards walk.
  SmallSet<Register, 16> Live;
  BitVector PhysLive;

  // --- Helpers ---

  SmallVector<MCPhysReg> getClassPhysRegs(const TargetRegisterClass *RC);
  bool classesCanOverlap(const TargetRegisterClass *A,
                         const TargetRegisterClass *B);

  // --- Model construction ---

  void postInterference(Register VReg);
  void createVariables();
  void buildConstraints(MachineBasicBlock &MBB);
  void configureBranching();

public:
  RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB);

  void recordSolution();
  bool solved() const { return Solved; }
  MCPhysReg getAssignment(Register VReg) const { return Solution[VReg]; }

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

/// Apply the solved register assignments: replace all virtual register
/// operands with their assigned physical registers.
static void applySolution(MachineBasicBlock &MBB, RegAllocProblem &Problem) {
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
    : TRI(*MF.getSubtarget().getRegisterInfo()), MRI(MF.getRegInfo()),
      Reserved(TRI.getReservedRegs(MF)) {
  createVariables();
  buildConstraints(MBB);
  configureBranching();
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

/// Create one Chuffed IntVar per allocatable vreg, with domain =
/// MCPhysReg values from its register class.
void RegAllocProblem::createVariables() {
  unsigned NumVRegs = MRI.getNumVirtRegs();
  RegVar.grow(Register::index2VirtReg(NumVRegs - 1));
  Solution.grow(Register::index2VirtReg(NumVRegs - 1));

  for (unsigned I = 0; I < NumVRegs; ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(VReg) || MRI.def_empty(VReg))
      continue;

    SmallVector<MCPhysReg> PhysRegs = getClassPhysRegs(MRI.getRegClass(VReg));
    int Lo = *llvm::min_element(PhysRegs);
    int Hi = *llvm::max_element(PhysRegs);
    IntVar *V = newIntVar(Lo, Hi);

    const TargetRegisterClass *RC = MRI.getRegClass(VReg);
    for (int Val = Lo; Val <= Hi; ++Val)
      if (!RC->contains(static_cast<MCPhysReg>(Val)) || Reserved[Val])
        int_rel(V, IRT_NE, Val);

    RegVar[VReg] = V;
    LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " ("
                      << TRI.getRegClassName(RC) << "): " << PhysRegs.size()
                      << " phys regs\n");
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
        // Tied operands should both be virtual before register allocation.
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

    // Phase 3: Uses and earlyclobber defs (simultaneous — earlyclobber
    // writes before reads, so they interfere with each other).
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
  vec<IntVar *> BranchVars;
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register VReg = Register::index2VirtReg(I);
    if (RegVar[VReg])
      BranchVars.push(RegVar[VReg]);
  }
  branch(BranchVars, VAR_SIZE_MIN, VAL_MIN);
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

/// Post != constraints between VReg and all live vregs/physregs it
/// conflicts with.
void RegAllocProblem::postInterference(Register VReg) {
  const TargetRegisterClass *RC = MRI.getRegClass(VReg);
  for (Register LiveReg : Live) {
    if (LiveReg == VReg)
      continue;
    if (!classesCanOverlap(RC, MRI.getRegClass(LiveReg)))
      continue;
    int_rel(RegVar[VReg], IRT_NE, RegVar[LiveReg], 0);
    LLVM_DEBUG(dbgs() << "  interference: " << printReg(VReg, &TRI) << " != "
                      << printReg(LiveReg, &TRI) << "\n");
  }
  for (MCPhysReg PhysReg : *RC) {
    if (!PhysLive[PhysReg])
      continue;
    int_rel(RegVar[VReg], IRT_NE, static_cast<int>(PhysReg));
    LLVM_DEBUG(dbgs() << "  interference: " << printReg(VReg, &TRI) << " != "
                      << TRI.getName(PhysReg) << "\n");
  }
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
