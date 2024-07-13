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
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/ISDOpcodes.h"
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

struct SchedulingDAG {
  ilist<Node> Nodes;
  unsigned NextIdx;

  DenseMap<const MachineInstr *, Node *> MINodes;

  MachineBasicBlock::iterator FrontierPos;
  SmallSetVector<Node *, 4> ForwardAvail;
  SmallSetVector<Node *, 4> BackwardAvail;
};

class MOSSched : public MachineFunctionPass {
public:
  static char ID;

  MOSSched() : MachineFunctionPass(ID) {
    llvm::initializeMOSSchedPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
  void widenRCs();
  void buildDAGs();
  void scheduleTrivialNodes();
  void scheduleNode(Node &N, bool Forward, SchedulingDAG &DAG,
                    MachineBasicBlock &MBB);
  void dump();

  MachineFunction *MF;
  MachineRegisterInfo *MRI;
  const TargetRegisterInfo *TRI;
  const TargetInstrInfo *TII;
  DenseMap<MachineBasicBlock *, SchedulingDAG> DAGs;
};

void MOSSched::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineLoopInfoWrapperPass>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MOSSched::runOnMachineFunction(MachineFunction &MF) {
  getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree().print(dbgs());
  getAnalysis<MachineLoopInfoWrapperPass>().getLI().print(dbgs());
  this->MF = &MF;
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  TII = MF.getSubtarget().getInstrInfo();
  MF.dump();
  widenRCs();
  MF.dump();
  buildDAGs();
  dump();
  scheduleTrivialNodes();
  dump();
  MF.dump();
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

void MOSSched::widenRCs() {
  dbgs() << "Inserting copies to fully widen RCs.\n";

  DenseMap<Register, Register> CarryingRegister;

  // Establish a maximal RC carrying register for each def.
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : make_early_inc_range(MBB)) {
      // Trivially forward copies.
      if (MI.isCopy() && MI.getOperand(0).getReg().isVirtual()) {
        MRI->replaceRegWith(MI.getOperand(0).getReg(),
                            MI.getOperand(1).getReg());
        MI.eraseFromParent();
        continue;
      }
      for (MachineOperand &MO : MI.defs()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        Register R = MO.getReg();
        const TargetRegisterClass *RC = MRI->getRegClass(R);
        const TargetRegisterClass *NewRC =
            TRI->getLargestLegalSuperClass(RC, *MF);
        if (NewRC == RC) {
          CarryingRegister[R] = R;
          continue;
        }
        Register New = MRI->createVirtualRegister(NewRC);
        const TargetRegisterClass *Constraint =
            TII->getRegClass(MI.getDesc(), MO.getOperandNo(), TRI, *MF);
        if (Constraint && Constraint != NewRC) {
          MachineIRBuilder Builder(MBB,
                                   std::next(MachineBasicBlock::iterator(MI)));
          Builder.buildCopy(New, R);
        } else {
          MO.setReg(New);
        }
        CarryingRegister[R] = New;
      }
    }
  }

  // Replace uses with the carrying register, inserting copies to resolve RC.
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineInstr &MI : MBB) {
      for (MachineOperand &MO : MI.uses()) {
        if (!MO.isReg() || !MO.getReg().isVirtual())
          continue;
        Register New = CarryingRegister[MO.getReg()];
        const TargetRegisterClass *RC =
            TII->getRegClass(MI.getDesc(), MO.getOperandNo(), TRI, *MF);
        if (RC && RC != MRI->getRegClass(New)) {
          Register CopyReg = MRI->createVirtualRegister(RC);
          MachineIRBuilder Builder(MBB, MI);
          Builder.buildCopy(CopyReg, New);
          New = CopyReg;
        }
        MO.setReg(New);
      }
    }
  }
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

    // Set initial frontier position.
    DAG.FrontierPos = MBB.getFirstTerminator();

    // Find predecessors.
    for (Node &N : DAG.Nodes) {
      for (MachineInstr *MI : N.MIs) {
        for (MachineOperand &MO : MI->explicit_uses()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          MachineInstr *Def = MRI->getUniqueVRegDef(MO.getReg());
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
        DAG.ForwardAvail.insert(&N);
      if (N.Successors.empty())
        DAG.BackwardAvail.insert(&N);
    }
  }
}

void MOSSched::scheduleTrivialNodes() {
  for (MachineBasicBlock &MBB : *MF) {
    SchedulingDAG &DAG = DAGs[&MBB];
    if (DAG.ForwardAvail.size() != 1 && DAG.BackwardAvail.size() != 1)
      continue;
    dbgs() << "\nTrivial scheduling for %bb." << MBB.getNumber() << '\n';
    do {
      if (DAG.ForwardAvail.size() == 1)
        scheduleNode(*DAG.ForwardAvail.front(), /*Forward=*/true, DAG, MBB);
      else if (DAG.BackwardAvail.size() == 1)
        scheduleNode(*DAG.BackwardAvail.front(), /*Forward=*/false, DAG, MBB);
    } while (DAG.ForwardAvail.size() == 1 || DAG.BackwardAvail.size() == 1);
  }
}

void MOSSched::scheduleNode(Node &N, bool Forward, SchedulingDAG &DAG,
                            MachineBasicBlock &MBB) {
  dbgs() << "\nScheduling Node " << N.Idx
         << (Forward ? " forward\n" : " backward\n");

  // Move MIs to frontier position and update it.
  for (MachineInstr *MI : N.MIs) {
    dbgs() << *MI;
    MBB.insert(DAG.FrontierPos, MI->removeFromParent());
  }
  if (!Forward)
    DAG.FrontierPos = N.MIs.front();

  // Remove the node from containing sets, update Avail, and erase the node.
  for (Node *P : N.Predecessors) {
    P->Successors.erase(&N);
    if (P->Successors.empty())
      DAG.BackwardAvail.insert(P);
  }
  for (Node *S : N.Successors) {
    S->Predecessors.erase(&N);
    if (S->Predecessors.empty())
      DAG.ForwardAvail.insert(S);
  }
  DAG.ForwardAvail.remove(&N);
  DAG.BackwardAvail.remove(&N);
  DAG.Nodes.erase(&N);
}

void MOSSched::dump() {
  for (MachineBasicBlock &MBB : *MF) {
    SchedulingDAG &DAG = DAGs[&MBB];
    if (DAG.Nodes.empty())
      continue;
    dbgs() << "\n%bb." << MBB.getNumber() << '\n';
    dumpNodes("Forward Avail", DAG.ForwardAvail);
    dumpNodes("Backward Avail", DAG.BackwardAvail);

    for (const auto &N : DAG.Nodes) {
      dbgs() << "\nNode " << N.Idx << ":\n";
      for (MachineInstr *MI : N.MIs)
        dbgs() << *MI;
      dumpNodes("Predecessors", N.Predecessors);
      dumpNodes("Successors", N.Successors);
    }
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
