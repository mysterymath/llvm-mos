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

struct Node : public ilist_node<Node> {
  Node(unsigned Idx, ArrayRef<MachineInstr *> MIs);
  unsigned Idx;
  SmallVector<MachineInstr *, 0> MIs;

  void addPredecessor(Node &N);

  SmallSet<Node *, 4> Predecessors;
  SmallSet<Node *, 4> Successors;
};

struct Frontier {
  // Position to which nodes may be scheduled.
  MachineBasicBlock::iterator Pos;

  // Nodes that can be safely scheduled.
  SmallSetVector<Node *, 4> Avail;
};

struct SchedulingDAG {
  ilist<Node> Nodes;
  unsigned NextIdx;

  DenseMap<const MachineInstr *, Node *> MINodes;

  Frontier ForwardFrontier;
  Frontier BackwardFrontier;
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
  void scheduleTrivialNodes();
  void scheduleNode(Node &N, Frontier &F, SchedulingDAG &DAG,
                    MachineBasicBlock &MBB);
  void dump();

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
  dump();
  scheduleTrivialNodes();
  dump();
  return true;
}

template <typename T> void dumpNodes(StringRef Name, const T &Nodes) {
  if (Nodes.empty())
    return;
  dbgs() << Name << ':';
  for (Node *N : Nodes)
    dbgs() << ' ' << N->Idx;
  dbgs() << '\n';
}

void MOSSched::buildDAGs() {
  DAGs.clear();
  for (MachineBasicBlock &MBB : *MF) {
    // Build nodes.
    SchedulingDAG &DAG = DAGs[&MBB];
    for (MachineInstr &MI : MBB) {
      if (MI.isPHI() || MI.isTerminator())
        continue;
      DAG.Nodes.push_back(new Node(DAG.NextIdx++, {&MI}));
      DAG.MINodes.try_emplace(&MI, &DAG.Nodes.back());
    }

    // Set initial frontier positions.
    DAG.ForwardFrontier.Pos = MBB.getFirstNonPHI();
    DAG.BackwardFrontier.Pos = MBB.getFirstTerminator();

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
          N.addPredecessor(*It->second);
        }

        if ((MI->isCopy() && MI->getOperand(0).getReg().isPhysical()) ||
            MI->getOpcode() == MOS::ADJCALLSTACKDOWN)
          for (MachineInstr &Succ : llvm::make_range(
                   std::next(MachineBasicBlock::iterator(MI)), MBB.end()))
            if (Succ.isCall())
              DAG.MINodes.find(&Succ)->second->addPredecessor(N);

        if (MI->getOpcode() == MOS::ADJCALLSTACKUP)
          for (MachineInstr &Pred :
               llvm::make_range(MBB.begin(), MachineBasicBlock::iterator(MI)))
            if (Pred.isCall())
              N.addPredecessor(*DAG.MINodes.find(&Pred)->second);
      }
    }

    // Find available nodes.
    for (auto &N : DAG.Nodes) {
      if (N.Predecessors.empty())
        DAG.ForwardFrontier.Avail.insert(&N);
      if (N.Successors.empty())
        DAG.BackwardFrontier.Avail.insert(&N);
    }
  }
}

void MOSSched::scheduleTrivialNodes() {
  for (auto &[MBB, DAG] : DAGs) {
    while (DAG.ForwardFrontier.Avail.size() == 1 ||
           DAG.BackwardFrontier.Avail.size() == 1) {
      if (DAG.ForwardFrontier.Avail.size() == 1)
        scheduleNode(*DAG.ForwardFrontier.Avail.front(), DAG.ForwardFrontier,
                     DAG, *MBB);
      else if (DAG.BackwardFrontier.Avail.size() == 1)
        scheduleNode(*DAG.BackwardFrontier.Avail.front(), DAG.BackwardFrontier,
                     DAG, *MBB);
    }
  }
}

void MOSSched::scheduleNode(Node &N, Frontier &F, SchedulingDAG &DAG,
                            MachineBasicBlock &MBB) {
  // Move MIs to frontier position and advance it.
  for (MachineInstr *MI : N.MIs)
    MBB.insert(F.Pos, MI->removeFromParent());
  if (&F == &DAG.BackwardFrontier)
    F.Pos = N.MIs.front();

  // Remove the node from containing lists and sets.
  for (Node *P : N.Predecessors)
    P->Successors.erase(&N);
  for (Node *S : N.Successors)
    S->Predecessors.erase(&N);
  DAG.ForwardFrontier.Avail.remove(&N);
  DAG.BackwardFrontier.Avail.remove(&N);
  DAG.Nodes.erase(&N);
}

void MOSSched::dump() {
  for (const auto &[MBB, DAG] : DAGs) {
    dbgs() << "\n\nMBB: " << MBB->getName() << '\n';
    for (const auto &N : DAG.Nodes) {
      dbgs() << "\nNode " << N.Idx << ":\n";
      for (MachineInstr *MI : N.MIs)
        dbgs() << *MI;
      dumpNodes("Predecessors", N.Predecessors);
      dumpNodes("Successors", N.Successors);
    }

    dumpNodes("Forward Avail", DAG.ForwardFrontier.Avail);
    dumpNodes("Backward Avail", DAG.ForwardFrontier.Avail);
  }
}

Node::Node(unsigned Idx, ArrayRef<MachineInstr *> MIs) : Idx(Idx), MIs(MIs) {}

void Node::addPredecessor(Node &N) {
  Predecessors.insert(&N);
  N.Successors.insert(this);
}

} // namespace

char MOSSched::ID = 0;

INITIALIZE_PASS(MOSSched, DEBUG_TYPE, "MOS Instruction Scheduler", false, false)

MachineFunctionPass *llvm::createMOSSchedPass() { return new MOSSched(); }
