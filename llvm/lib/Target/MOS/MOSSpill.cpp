//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Split and spill SSA live ranges to permit imaginary register assignment.
///
/// Starting from conventional SSA, reduce pressure until MOSImagRegAssign can
/// assign imaginary storage and repair constraints without further spilling.
/// The capacity guarantee must account for Imag8/Imag16 aliasing, the shared
/// assignments represented by PHI reservations, and instruction constraints.
/// Spill stores and reloads preserve SSA value flow; this pass chooses which
/// ranges need stack storage, but does not choose physical imaginary registers.
///
/// Only the pipeline entry point is implemented. Reject execution until the
/// pass can satisfy its contract; do not feed unallocated MIR to later passes.
///
//===----------------------------------------------------------------------===//

#include "MOSSpill.h"
#include "MOS.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-spill"

using namespace llvm;

namespace {

class MOSSpill : public MachineFunctionPass {
public:
  static char ID;

  MOSSpill() : MachineFunctionPass(ID) {
    initializeMOSSpillPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &) override {
    report_fatal_error("MOSSpill is not implemented", false);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // namespace

char MOSSpill::ID = 0;
INITIALIZE_PASS(MOSSpill, DEBUG_TYPE, "MOS Imaginary Register Spilling", false,
                false)

MachineFunctionPass *llvm::createMOSSpillPass() { return new MOSSpill; }
