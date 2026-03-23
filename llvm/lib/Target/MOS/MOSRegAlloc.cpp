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

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#include "chuffed/core/engine.h"
#include "chuffed/core/options.h"
#include "chuffed/globals/globals.h"
#include "chuffed/primitives/primitives.h"
#include "chuffed/vars/modelling.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

/// Smoke test: create a trivial Chuffed CP problem and solve it.
/// This verifies that Chuffed headers, linking, and runtime all work
/// from within the LLVM pass.
namespace {
class ChuffedSmokeTest : public Problem {
public:
  IntVar *x, *y;

  ChuffedSmokeTest() {
    x = newIntVar(1, 5);
    y = newIntVar(1, 5);
    int_rel(x, IRT_NE, y, 0);
    vec<IntVar *> vars;
    vars.push(x);
    vars.push(y);
    branch(vars, VAR_SIZE_MIN, VAL_MIN);
    optimize(x, OPT_MIN);
  }

  void print(std::ostream &os) override {
    os << "x=" << x->getVal() << " y=" << y->getVal() << "\n";
  }
};
} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: Chuffed smoke test\n");
  LLVM_DEBUG({
    so.nof_solutions = 1;
    so.print_sol = false;
    engine.solve(new ChuffedSmokeTest());
    dbgs() << "MOS RegAlloc: Chuffed OK\n";
  });
  return false;
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
