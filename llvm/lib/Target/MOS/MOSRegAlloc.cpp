//===-- MOSReglloc.cpp - MOS Register Allocation -------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocator.
//
//===----------------------------------------------------------------------===//
//
// The program is assumed on entry to be in SSA form, with all subregister
// definitions performed by INSERT_SUBREG operations, and with critical edges
// split. To begin, COPY operations between vregs are all forwarded, such
// that the only remaining copy operations are between pregs and vregs.

// This isn't technically legal, as the register classes of definitions are not
// properly constrained by their uses. This is intentional, as vregs are used as
// a global value numbering, not as real temporary variables.

// Tied operands are handles specially; the tied use is required to be killed.
// This is achieved by the earlier SSA pass. The new SSA value formed by the
// tied def is considered an extension of the use; they must be assigned to the
// same location around the instruction. This does not interfere with the SSA
// properties used to produce the initial allocation.

// We consider there to be two "program points" between every two instructions
// and between an instruction and the basic block begin or end. One program
// point immediately follows the instruction, and one immediately preceeds the
// next. This accomidates arbirtrary instruction insertion between the two
// points to transform from the preceeding allocation to the next.

// First, a Hack-style SSA allocation is performed to assign values to imaginary
// registers. These form "backups" for values that cannot be placed in
// architectural registers. Some slight modifications need to be made, since
// instructions may not have imaginary registers in their register classes. For
// such instructions that define values, the live range of the imaginary
// register is considered to begin at the second point after the instruction, to
// allow for the copy. Similarly, for instructions that kill values, the value
// is considered killed after the first of the points beefore the use. This
// again accomodates the copy.

// Now, a cost model is established for tree-width dynamic programming. Given a
// pair of program points, and allocations for each, there is an integral cost.
// Use and earlyclobber def register classes hard-constrain the preceeding
// program point, while def register classes and regmasks hard-constrain the
// following program point.

#include "MOSRegAlloc.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <optional>
#include <vector>

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    llvm::initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  MachineFunctionProperties getRequiredProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  LiveVariables *LV;
  MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineDominanceFrontier *MDF;
  const MachineDominatorTree *MDT;
  const MachineLoopInfo *MLI;
  MachineFunction *MF;
  RegisterClassInfo RCI;

  // Map from value to imaginary register
  DenseMap<Register, Register> ImagAlloc;

  void assignImagRegs(const MachineDomTreeNode &MDTN, SmallSet<Register, 8> DomLiveOutVals = {});

  const TargetRegisterClass *getOperandRegClass(const MachineOperand &MO) const;
  bool isDeadMI(const MachineInstr &MI) const;
};

} // namespace

MachineFunctionProperties MOSRegAlloc::getRequiredProperties() const {
  return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
}

void MOSRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariables>();
  AU.addRequired<MachineDominanceFrontier>();
  AU.addPreserved<MachineDominanceFrontier>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
}

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  // TODO: Subregister liveness.
  this->MF = &MF;

  dbgs() << "\n# MOS Register Allocator: " << MF.getName() << "\n\n";

  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MRI->getTargetRegisterInfo();
  MDF = &getAnalysis<MachineDominanceFrontier>();
  MDT = &getAnalysis<MachineDominatorTree>();
  MLI = &getAnalysis<MachineLoopInfo>();
  LV = &getAnalysis<LiveVariables>();

  MF.dump();

  dbgs() << "## Coalesce away copies.\n";

  // Temporarily coalesce away all copies. This makes the register classes
  // invalid, but true copies will be inserted as allocation proceeds to
  // restore them.
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;
    MachineInstr *MI = MRI->getUniqueVRegDef(R);
    if (!MI->isCopy())
      continue;
    MachineOperand &Src = MI->getOperand(1);
    if (!Src.getReg().isVirtual())
      continue;
    dbgs() << "Coalescing: " << *MI;
    for (MachineOperand &Use :
         make_early_inc_range(MRI->use_nodbg_operands(R))) {
      dbgs() << "Use MI: " << *Use.getParent();
      unsigned SubRegIdx = 0;
      if (Src.getSubReg()) {
        SubRegIdx = Src.getSubReg();
        if (Use.getSubReg())
          SubRegIdx = TRI->composeSubRegIndices(SubRegIdx, Use.getSubReg());
      }
      Use.setReg(Src.getReg());
      Use.setSubReg(SubRegIdx);
    }
    MI->eraseFromParent();
  }

  MF.dump();

  MRI->freezeReservedRegs(MF);
  RCI.runOnMachineFunction(MF);

  ReversePostOrderTraversal<MachineFunction *> RPOT(&MF);

  // TODO!
  ImagAlloc.clear();
  assignImagRegs(*MDT->getRootNode());

  // Recompute liveness and kill dead instructions.
  for (MachineBasicBlock *MBB : post_order(&MF)) {

    bool RemovedAny;
    do {
      recomputeLivenessFlags(*MBB);
      RemovedAny = false;
      for (MachineInstr &MI : make_early_inc_range(*MBB)) {
        if (!isDeadMI(MI))
          continue;
        dbgs() << "Remove dead MI: " << MI;
        MI.eraseFromParent();
        RemovedAny = true;
      }
    } while (RemovedAny);

    recomputeLiveIns(*MBB);
  }

  MRI->leaveSSA();
  MF.getProperties().set(MachineFunctionProperties::Property::NoPHIs);
  MRI->clearVirtRegs();

  return false;
}

void MOSRegAlloc::assignImagRegs(const MachineDomTreeNode &MDTN, SmallSet<Register, 8> DomLiveOutVals) {
  SmallSet<Register, 8> LiveInVals;
  for (Register R : DomLiveOutVals)
    if (LV->isLiveIn(R, *MDTN.getBlock()))
      LiveInVals.insert(R);
}

const TargetRegisterClass *
MOSRegAlloc::getOperandRegClass(const MachineOperand &MO) const {
  const TargetRegisterClass *RC =
      MO.getParent()->getRegClassConstraint(MO.getOperandNo(), TII, TRI);
  if (!RC)
    RC = TRI->getLargestLegalSuperClass(MRI->getRegClass(MO.getReg()),
                                        *MO.getParent()->getMF());
  return RC;
}

/// Returns whether an instruction is dead. Conservative, but depends on
/// liveness flags for good results.
bool MOSRegAlloc::isDeadMI(const MachineInstr &MI) const {
  if (MI.isCall() || MI.isBranch() || MI.isReturn() || MI.mayLoadOrStore())
    return false;

  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && !MO.isDead())
      return false;
  }

  return true;
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocator", false, false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
