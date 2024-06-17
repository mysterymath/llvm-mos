//===-- MOSSched.cpp - MOS Register Allocator --------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// ===----------------------------------------------------------------------===//
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
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define DEBUG_TYPE "mos-sched"

using namespace llvm;

namespace {

struct Node {
  unsigned Idx;
  MachineInstr *MI;
  SmallSet<Node *, 4> Predecessors;
};

struct SchedulingDAG {
  SmallVector<Node, 10> Nodes;
  DenseMap<const MachineInstr *, Node *> MINodes;
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

  for (MachineInstr &MI : *MBB) {
    if (MI.isPHI() || MI.isTerminator())
      continue;
    DAG.Nodes.push_back({DAG.NextIdx++, &MI, {}});
  }
  for (Node &N : DAG.Nodes)
    DAG.MINodes.try_emplace(N.MI, &N);

  const MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  for (Node &N : DAG.Nodes) {
    for (MachineOperand &MO : N.MI->explicit_uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      MachineInstr *Def = MRI.getUniqueVRegDef(MO.getReg());
      if (Def->getParent() != MBB)
        continue;
      auto It = DAG.MINodes.find(Def);
      if (It == DAG.MINodes.end())
        continue;
      N.Predecessors.insert(It->second);
    }
  }

  for (const Node &N : DAG.Nodes) {
    dbgs() << "Node " << N.Idx << ": " << *N.MI;
    if (!N.Predecessors.empty()) {
      dbgs() << "Predecessors:";
      for (Node *P : N.Predecessors)
        dbgs() << ' ' << P->Idx;
      dbgs() << '\n';
    }
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
