//===-- MOSRegAlloc.cpp - MOS Register Allocator --------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocator.
//
// The usual LLVM pipeline involves phase-separated instruction scheduling and
// register allocation. Contrast the usual 6502 assembly programmer, which can
// solve both problems simultaneously, using information about one problem to
// inform decisions about the other.
//
// The general problem of combined instruction scheduling and register
// allocation is more difficult than either, and both problems are already quite
// difficult. However, the 6502 has much more circularity between the two
// problems, owing to its irregularity. Luckily, that same irregularity makes
// the space of possible solutions have extremely sharp gradients, which allows
// heuristic techniques to work well. This works much less well if the problems
// are considered separately, since a large don't-care region in one problem may
// have an overwhelming preference in the in the other.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MOS.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    llvm::initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) { return true; }

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocator", false, false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
