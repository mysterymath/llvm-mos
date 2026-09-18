//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Choose hardware placements and emit physical instructions from assigned SSA.
///
/// MOSImagRegAssign supplies legal imaginary assignments and the copies needed
/// to satisfy their constraints. Choose where values occupy hardware registers
/// and when to materialize their imaginary storage, inserting transfers or
/// rematerializations to satisfy the remaining instruction constraints.
/// Whenever a live value leaves hardware, it must remain recoverable from its
/// assigned imaginary storage or by rematerialization. Required imaginary
/// accesses must find their operands there, even if copies elsewhere survive.
///
/// Resolve PHIs and parallel copies and replace virtual registers with physical
/// registers. The result is no longer SSA and requires no further allocation of
/// virtual registers; target pseudo expansion and frame lowering follow.
///
/// Only the pipeline entry point is implemented. Reject execution until the
/// pass can satisfy its contract; do not feed unallocated MIR to later passes.
///
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"
#include "MOS.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &) override {
    report_fatal_error("MOSRegAlloc is not implemented", false);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // namespace

char MOSRegAlloc::ID = 0;
INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Hardware Register Allocation",
                false, false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
