//===-- MOSRegAlloc.cpp - MOS Register Allocation -------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocation pass.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MOS.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "mos-reg-alloc"

using namespace llvm;

namespace {

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    llvm::initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineFunction *MF;
  MachineRegisterInfo *MRI;

  // For rewriteSSAValues().
  DenseMap<Register, Register> RewrittenVReg;

  void rewriteSSAValues();
  Register rewriteSSAValue(Register R);
  LLT findRegType(Register R);
};

} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  this->MRI = &MF.getRegInfo();
  MF.dump();
  dbgs() << "Rewriting SSA Values.\n";
  rewriteSSAValues();
  MF.dump();
  return false;
}

// Strip out register classes and copies from virtual regs to establish the
// invariant that each SSA value has exactly one SSA variable.
void MOSRegAlloc::rewriteSSAValues() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->use_nodbg_empty(R))
      rewriteSSAValue(R);
  }
}

Register MOSRegAlloc::rewriteSSAValue(Register R) {
  auto It = RewrittenVReg.find(R);
  if (It != RewrittenVReg.end())
    return It->second;

  MachineInstr *Def = MRI->getUniqueVRegDef(R);
  Register New;
  if (Def->isCopy()) {
    New = rewriteSSAValue(Def->getOperand(1).getReg());
    Def->eraseFromParent();
  } else {
    New = MRI->createGenericVirtualRegister(findRegType(R));
  }
  RewrittenVReg.try_emplace(R, New);
  MRI->replaceRegWith(R, New);
  return New;
}

LLT MOSRegAlloc::findRegType(Register R) {
  LLT Ty = MRI->getType(R);
  if (Ty.isValid())
    return Ty;
  const TargetRegisterInfo *TRI = MF->getSubtarget().getRegisterInfo();
  return LLT::scalar(TRI->getRegSizeInBits(R, *MRI));
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
