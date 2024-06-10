//===-- MOSRegAlloc.cpp - MOS Register Allocator --------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocator.
//
// The usual LLVM pipeline involves phase-separated instruction scheduling and
// register allocation. Contrast the usual 6502 assembly programmer, which can
// solve both problems simultaneously, using information about one problem to
// inform decisions about the other.
//
// The general problem of combined instruction scheduling and register
// allocation is more difficult than either, and both problems are already quite
// difficult. However, the 6502 has much more circularity between the two
// problems, owing to its irregularity. Luckily, that same irregularity makes
// the space of possible solutions have extremely sharp gradients, which allows
// heuristic techniques to work well. This works much less well if the problems
// are considered separately, since a large don't-care region in one problem may
// have an overwhelming preference in the in the other.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MOS.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

struct DAG;
struct Node;

struct SchedulingDAG {
  SmallVector<std::unique_ptr<Node>, 0> Nodes;
  unsigned NextIdx;
};

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    llvm::initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void buildDAG(MachineFunction &MF);

  SchedulingDAG DAG;
};

struct Edge {
  Register Reg;
  const TargetRegisterClass *RC;
  Node *Node;
};

struct Node {
  unsigned Idx;
  SmallVector<MachineInstr *> Instrs;
  SmallVector<Edge> Defs;
  SmallVector<Edge> EarlyClobberDefs;
  SmallVector<Edge> Uses;

  void printEdges(StringRef Name, const SmallVector<Edge> &Edges) const;
};

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  buildDAG(MF);

  return true;
}

void MOSRegAlloc::buildDAG(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  DAG.Nodes.clear();
  DAG.NextIdx = 0;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      auto N = std::make_unique<Node>();
      N->Idx = DAG.NextIdx++;

      N->Instrs.push_back(&MI);
      for (MachineOperand &MO : MI.operands()) {
        if (!MO.isReg())
          continue;
        SmallVector<Edge> *Edges;
        if (MO.isDef())
          Edges = MO.isEarlyClobber() ? &N->EarlyClobberDefs : &N->Defs;
        else
          Edges = &N->Uses;
        Edges->push_back({MO.getReg(),
                          TII.getRegClass(TII.get(MI.getOpcode()),
                                          MO.getOperandNo(), &TRI, MF),
                          /*Node=*/nullptr});
      }
      DAG.Nodes.push_back(std::move(N));
    }
  }

  for (const std::unique_ptr<Node> &N : DAG.Nodes) {
    dbgs() << "Node " << N->Idx << ":\n";

    N->printEdges("Defs", N->Defs);
    N->printEdges("Early Clobber Defs", N->EarlyClobberDefs);
    N->printEdges("Uses", N->Uses);

    dbgs() << "Instrs:\n";
    for (const MachineInstr *MI : N->Instrs)
      dbgs() << *MI;

    dbgs() << '\n';
  }
}

void Node::printEdges(StringRef Name, const SmallVector<Edge> &Edges) const {
  if (Edges.empty())
    return;
  const TargetRegisterInfo &TRI =
      *Instrs.front()->getMF()->getSubtarget().getRegisterInfo();
  dbgs() << Name << ": ";
  for (const Edge &E : Edges) {
    dbgs() << printReg(E.Reg);
    if (E.RC)
      dbgs() << "(" << TRI.getRegClassName(E.RC) << ")";
    dbgs() << ' ';
  }
  dbgs() << '\n';
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocator", false, false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
