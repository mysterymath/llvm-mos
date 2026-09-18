//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Assign imaginary storage and repair instruction constraints in SSA form.
///
/// MOSSpill guarantees sufficient capacity. Assign each range needing imaginary
/// storage a physical imaginary register in VirtRegMap, respecting aliasing and
/// PHI reservations. Repair imaginary register constraints, including physical
/// uses, definitions, clobbers, and ties, with copies and SSA live-range splits.
/// Each resulting range has one assignment throughout its lifetime.
///
/// The output remains SSA: hardware register placement is left to MOSRegAlloc.
/// That pass may omit imaginary definitions and copies when values stay in
/// hardware, but must honor accesses that require particular imaginary storage.
/// It need not choose or repair the imaginary assignments itself.
///
/// Only the pipeline entry point is implemented. Reject execution until the
/// pass can satisfy its contract; do not feed unallocated MIR to later passes.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegAssign.h"
#include "MOS.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-imag-regassign"

using namespace llvm;

namespace {

class MOSImagRegAssign : public MachineFunctionPass {
public:
  static char ID;

  MOSImagRegAssign() : MachineFunctionPass(ID) {
    initializeMOSImagRegAssignPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &) override {
    report_fatal_error("MOSImagRegAssign is not implemented", false);
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
};

} // namespace

char MOSImagRegAssign::ID = 0;
INITIALIZE_PASS(MOSImagRegAssign, DEBUG_TYPE,
                "MOS Imaginary Register Assignment", false, false)

MachineFunctionPass *llvm::createMOSImagRegAssignPass() {
  return new MOSImagRegAssign;
}
