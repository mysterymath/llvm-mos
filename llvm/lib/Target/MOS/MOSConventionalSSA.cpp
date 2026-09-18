//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Normalize the SSA into "conventional" form before allocation.
///
/// The shape of SSA IR does not necessarily allow a single register to be
/// assigned to a PHI def and all its uses. This means that coming out of SSA
/// would correspond to inserting copies along the edge from the PHI's
/// predecessor to the PHI block. Unfortunately, in the presence of indirect
/// branches (computed goto), this isn't generally possible without severe
/// performance impact.
///
/// Accordingly, this pass inserts "parallel copies" (PCOPY pseudos) to ensure
/// that every value used or defined by the PHI is unique, and thus trivially
/// assignable to the same register. This is called "conventional SSA form" in
/// the literature.
///
//===----------------------------------------------------------------------===//

#include "MOSConventionalSSA.h"
#include "MOS.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-conventional-ssa"

using namespace llvm;

namespace {

class MOSConventionalSSA : public MachineFunctionPass {
public:
  static char ID;

  MOSConventionalSSA() : MachineFunctionPass(ID) {
    initializeMOSConventionalSSAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &) override {
    report_fatal_error("MOSConventionalSSA is not implemented", false);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // namespace

char MOSConventionalSSA::ID = 0;
INITIALIZE_PASS(MOSConventionalSSA, DEBUG_TYPE, "MOS Conventional SSA", false,
                false)

MachineFunctionPass *llvm::createMOSConventionalSSAPass() {
  return new MOSConventionalSSA;
}
