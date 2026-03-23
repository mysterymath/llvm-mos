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
#include "chuffed/vars/modelling.h"


#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

/// Get the allocatable physical registers for a register class.
static SmallVector<MCPhysReg> getClassPhysRegs(const TargetRegisterClass *RC,
                                               const BitVector &Reserved) {
  SmallVector<MCPhysReg> Regs;
  for (MCPhysReg Reg : *RC) {
    if (!Reserved[Reg])
      Regs.push_back(Reg);
  }
  return Regs;
}

/// Check whether two register classes could ever alias.
static bool classesCanOverlap(const TargetRegisterInfo &TRI,
                              const TargetRegisterClass *A,
                              const TargetRegisterClass *B) {
  for (MCPhysReg RegA : *A)
    for (MCPhysReg RegB : *B)
      if (TRI.regsOverlap(RegA, RegB))
        return true;
  return false;
}

/// Chuffed Problem subclass that models register allocation for a single
/// basic block as a constraint satisfaction problem.
///
/// Variables: one IntVar per virtual register, domain = MCPhysReg enum
/// values from the vreg's register class.
///
/// Constraints: interfering vregs (overlapping live ranges whose register
/// classes could alias) must be assigned non-overlapping physical registers.
class RegAllocProblem : public Problem {
  const TargetRegisterInfo &TRI;
  MachineRegisterInfo &MRI;

  IndexedMap<IntVar *, VirtReg2IndexFunctor> RegVar;
  IndexedMap<MCPhysReg, VirtReg2IndexFunctor> Solution;
  bool Solved = false;

public:
  RegAllocProblem(MachineFunction &MF, MachineBasicBlock &MBB)
      : TRI(*MF.getSubtarget().getRegisterInfo()), MRI(MF.getRegInfo()) {

    BitVector Reserved = TRI.getReservedRegs(MF);
    unsigned NumVRegs = MRI.getNumVirtRegs();

    RegVar.grow(Register::index2VirtReg(NumVRegs - 1));
    Solution.grow(Register::index2VirtReg(NumVRegs - 1));

    // --- Variables ---
    // Create a CP variable for each vreg that has non-debug references
    // and a def. Domain = MCPhysReg values from the register class.
    vec<IntVar *> BranchVars;
    for (unsigned I = 0; I < NumVRegs; ++I) {
      Register VReg = Register::index2VirtReg(I);
      if (MRI.reg_nodbg_empty(VReg) || MRI.def_empty(VReg))
        continue;

      SmallVector<MCPhysReg> PhysRegs =
          getClassPhysRegs(MRI.getRegClass(VReg), Reserved);
      int Lo = *llvm::min_element(PhysRegs);
      int Hi = *llvm::max_element(PhysRegs);
      IntVar *V = newIntVar(Lo, Hi);

      const TargetRegisterClass *RC = MRI.getRegClass(VReg);
      for (int Val = Lo; Val <= Hi; ++Val)
        if (!RC->contains(static_cast<MCPhysReg>(Val)) || Reserved[Val])
          int_rel(V, IRT_NE, Val);

      RegVar[VReg] = V;
      BranchVars.push(V);
      LLVM_DEBUG(dbgs() << "  " << printReg(VReg, &TRI) << " ("
                        << TRI.getRegClassName(MRI.getRegClass(VReg))
                        << "): " << PhysRegs.size() << " phys regs\n");
    }

    // --- Constraints: live range interference ---
    // Walk backwards. Uses make vregs live; defs kill them. A newly-live
    // vreg interferes with everything already in the live set.
    SmallSet<Register, 16> Live;
    for (MachineInstr &MI : reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;

      // Defs kill vregs. Tied defs must match their use's register.
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual() || !MO.isDef())
          continue;
        if (MO.isTied()) {
          Register UseReg =
              MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo())).getReg();
          if (UseReg.isVirtual())
            int_rel(RegVar[MO.getReg()], IRT_EQ, RegVar[UseReg], 0);
        }
        Live.erase(MO.getReg());
      }

      // Uses make vregs live. A newly-live vreg interferes with
      // everything already live.
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg() || !MO.getReg().isVirtual() || !MO.isUse())
          continue;
        Register Use = MO.getReg();
        if (Live.contains(Use))
          continue;
        for (Register LiveReg : Live) {
          if (!classesCanOverlap(TRI, MRI.getRegClass(Use),
                                 MRI.getRegClass(LiveReg)))
            continue;
          // For 8-bit classes (M1), != on MCPhysReg values is correct.
          // 16-bit pairs (M5) will need a more general non-overlap
          // constraint.
          int_rel(RegVar[Use], IRT_NE, RegVar[LiveReg], 0);
          LLVM_DEBUG(dbgs() << "  interference: " << printReg(Use, &TRI)
                            << " != " << printReg(LiveReg, &TRI) << "\n");
        }
        Live.insert(Use);
      }
    }

    branch(BranchVars, VAR_SIZE_MIN, VAL_MIN);
  }

  void recordSolution() {
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

  bool solved() const { return Solved; }

  /// Return the assigned physical register, or 0 if unallocated.
  MCPhysReg getAssignment(Register VReg) const { return Solution[VReg]; }

  void print(std::ostream &) override {}
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

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  if (MF.getRegInfo().getNumVirtRegs() == 0)
    return false;

  LLVM_DEBUG(dbgs() << "MOS RegAlloc: " << MF.getName() << " ("
                    << MF.getRegInfo().getNumVirtRegs() << " vregs)\n");

  for (MachineBasicBlock &MBB : MF) {
    LLVM_DEBUG(dbgs() << "  Block " << MBB.getName() << ": " << MBB.size()
                      << " instrs\n");

    // Chuffed uses global state (engine, sat, ldsb) that accumulates
    // variables and propagators. Reset between solves via placement new.
    engine.~Engine();
    new (&engine) Engine();
    sat.~SAT();
    new (&sat) SAT();
    ldsb.~LDSB();
    new (&ldsb) LDSB();

    so.nof_solutions = 1;
    so.print_sol = false;
    so.verbosity = 0;

    auto *Problem = new RegAllocProblem(MF, MBB);
    engine.setSolutionCallback([](::Problem *P) {
      static_cast<RegAllocProblem *>(P)->recordSolution();
    });
    engine.solve(Problem);

    if (!Problem->solved())
      report_fatal_error("MOS CP register allocator failed");

    for (MachineInstr &MI : MBB)
      for (MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.getReg().isVirtual())
          if (MCPhysReg PhysReg = Problem->getAssignment(MO.getReg()))
            MO.setReg(PhysReg);
  }

  MF.getRegInfo().clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
