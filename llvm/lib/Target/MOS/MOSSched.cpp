//===-- MOSSched.cpp - MOS Register Allocator --------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS instruction scheduler.
//
// Unlike MachineScheduler, this operates on SSA, since the MOS register
// allocator takes SSA. This pass also performs two-address instruction
// chaining and commutation.
//
//===----------------------------------------------------------------------===//

#include "MOSSched.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define DEBUG_TYPE "mos-sched"

using namespace llvm;

namespace {

struct Node {
  unsigned Idx;
  MachineInstr *Instr;
};

struct SchedulingDAG {
  SmallVector<Node, 0> Nodes;
  unsigned NextIdx;
  void clear();
};

class MOSSched : public MachineFunctionPass {
public:
  static char ID;

  MOSSched() : MachineFunctionPass(ID) {
    llvm::initializeMOSSchedPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void scheduleBlock(MachineBasicBlock &MBB);
  void buildDAG();

  MachineBasicBlock *MBB;
  SchedulingDAG DAG;
};

bool MOSSched::runOnMachineFunction(MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF)
    scheduleBlock(MBB);

  return true;
}

void MOSSched::scheduleBlock(MachineBasicBlock &MBB) {
  dbgs() << "Scheduling block " << MBB.getName() << '\n';
  this->MBB = &MBB;
  buildDAG();
  dbgs() << '\n';
}

void MOSSched::buildDAG() {
  DAG.clear();

  for (MachineInstr &MI : *MBB)
    DAG.Nodes.push_back({DAG.NextIdx++, &MI});

  for (const Node &N : DAG.Nodes) {
    dbgs() << "Node " << N.Idx << ": " << *N.Instr;
    dbgs() << '\n';
  }
}

void SchedulingDAG::clear() {
  Nodes.clear();
  NextIdx = 0;
}

} // namespace

char MOSSched::ID = 0;

INITIALIZE_PASS(MOSSched, DEBUG_TYPE, "MOS Instruction Scheduler", false, false)

MachineFunctionPass *llvm::createMOSSchedPass() { return new MOSSched(); }
