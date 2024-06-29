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
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define DEBUG_TYPE "mos-sched"

using namespace llvm;

namespace {

struct Node {
  unsigned Idx;
  SmallVector<MachineInstr *, 0> MIs;

  void addPredecessor(Node *N);

  SmallSet<Node *, 4> Predecessors;
  SmallSet<Node *, 4> Successors;
};

struct SchedulingDAG {
  SmallVector<Node, 0> Nodes;
  unsigned NextIdx;

  DenseMap<const MachineInstr *, Node *> MINodes;

  // Nodes that can be safely scheduled after the forward frontier (starting
  // from BB entry).
  SmallVector<Node *, 4> ForwardAvail;

  // Nodes that can be safely scheduled before the backwards frontier (starting
  // from BB exit).
  SmallVector<Node *, 4> BackwardAvail;
};

class MOSSched : public MachineFunctionPass {
public:
  static char ID;

  MOSSched() : MachineFunctionPass(ID) {
    llvm::initializeMOSSchedPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
  void buildDAGs();

  MachineFunction *MF;
  DenseMap<MachineBasicBlock *, SchedulingDAG> DAGs;
};

void MOSSched::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MOSSched::runOnMachineFunction(MachineFunction &MF) {
  getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree().print(dbgs());
  getAnalysis<MachineLoopInfo>().getBase().print(dbgs());
  this->MF = &MF;
  buildDAGs();
  return true;
}

void MOSSched::buildDAGs() {
  DAGs.clear();
  for (MachineBasicBlock &MBB : *MF) {
    // Build nodes.
    SchedulingDAG &DAG = DAGs[&MBB];
    for (MachineInstr &MI : MBB) {
      if (MI.isPHI() || MI.isTerminator())
        continue;
      DAG.Nodes.push_back(Node{DAG.NextIdx++, {&MI}, {}, {}});
    }

    // Build map from MI to node.
    for (Node &N : DAG.Nodes)
      for (MachineInstr *MI : N.MIs)
        DAG.MINodes.try_emplace(MI, &N);

    // Find predecessors.
    const MachineRegisterInfo &MRI = MBB.getParent()->getRegInfo();
    for (Node &N : DAG.Nodes) {
      for (MachineInstr *MI : N.MIs) {
        for (MachineOperand &MO : MI->explicit_uses()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          MachineInstr *Def = MRI.getUniqueVRegDef(MO.getReg());
          if (Def->getParent() != &MBB)
            continue;
          auto It = DAG.MINodes.find(Def);
          if (It == DAG.MINodes.end())
            continue;
          N.addPredecessor(It->second);
        }

        if ((MI->isCopy() && MI->getOperand(0).getReg().isPhysical()) ||
            MI->getOpcode() == MOS::ADJCALLSTACKDOWN)
          for (MachineInstr &Succ : llvm::make_range(
                   std::next(MachineBasicBlock::iterator(MI)), MBB.end()))
            if (Succ.isCall())
              DAG.MINodes.find(&Succ)->second->addPredecessor(&N);

        if (MI->getOpcode() == MOS::ADJCALLSTACKUP)
          for (MachineInstr &Pred :
               llvm::make_range(MBB.begin(), MachineBasicBlock::iterator(MI)))
            if (Pred.isCall())
              N.addPredecessor(DAG.MINodes.find(&Pred)->second);
      }
    }

    // Find available nodes.
    for (Node &N : DAG.Nodes) {
      if (N.Predecessors.empty())
        DAG.ForwardAvail.push_back(&N);
      if (N.Successors.empty())
        DAG.BackwardAvail.push_back(&N);
    }

    dbgs() << "\n\nMBB: " << MBB.getName() << '\n';
    for (const Node &N : DAG.Nodes) {
      dbgs() << "\nNode " << N.Idx << ":\n";
      for (MachineInstr *MI : N.MIs)
        dbgs() << *MI;
      if (!N.Predecessors.empty()) {
        dbgs() << "Predecessors:";
        for (Node *P : N.Predecessors)
          dbgs() << ' ' << P->Idx;
        dbgs() << '\n';
      }
      if (!N.Successors.empty()) {
        dbgs() << "Successors:";
        for (Node *P : N.Successors)
          dbgs() << ' ' << P->Idx;
        dbgs() << '\n';
      }
    }

    if (!DAG.ForwardAvail.empty()) {
      dbgs() << "\nForward Avail:";
      for (const Node *N : DAG.ForwardAvail)
        dbgs() << ' ' << N->Idx;
      dbgs() << '\n';
    }
    if (!DAG.BackwardAvail.empty()) {
      dbgs() << "\nBackward Avail:";
      for (const Node *N : DAG.BackwardAvail)
        dbgs() << ' ' << N->Idx;
      dbgs() << '\n';
    }
  }
}

void Node::addPredecessor(Node *N) {
  Predecessors.insert(N);
  N->Successors.insert(this);
}

} // namespace

char MOSSched::ID = 0;

INITIALIZE_PASS(MOSSched, DEBUG_TYPE, "MOS Instruction Scheduler", false, false)

MachineFunctionPass *llvm::createMOSSchedPass() { return new MOSSched(); }
