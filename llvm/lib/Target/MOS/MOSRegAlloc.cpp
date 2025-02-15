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
#include "llvm/ADT/DenseSet.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "mos-reg-alloc"

using namespace llvm;

namespace {

struct RegFamily {
  DenseSet<Register> Regs;
};

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

  void buildRegFamilies();

private:
  MachineFunction *MF;

  SmallVector<std::unique_ptr<RegFamily>> RegFamilies;
  DenseMap<Register, RegFamily*> FamilyForReg;
};

} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  buildRegFamilies();
  return false;
}

void MOSRegAlloc::buildRegFamilies() {
  MachineRegisterInfo &MRI = MF->getRegInfo();

  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI.reg_nodbg_empty(R))
      continue;
  }
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
