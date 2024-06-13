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
// chaining and commutation. Constants, vreg-vreg copies, and REG_SEQUENCE are
// left alone, since they are handled symbolically by the register allocator.
//
//===----------------------------------------------------------------------===//

#include "MOSSched.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

#define DEBUG_TYPE "mos-sched"

using namespace llvm;

namespace {

struct DAG;
struct Node;

struct SchedulingDAG {
  DenseMap<Register, const MachineInstr *> Constants;
  SmallVector<std::unique_ptr<Node>, 0> Nodes;
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

bool MOSSched::runOnMachineFunction(MachineFunction &MF) {
  buildDAG(MF);

  return true;
}

void MOSSched::buildDAG(MachineFunction &MF) {
  const TargetInstrInfo &TII = *MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();

  DAG.clear();
  DAG.NextIdx = 0;

  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      switch (MI.getOpcode()) {
      case MOS::LDImm:
      case MOS::LDImm1:
      case MOS::LDImm16:
        DAG.Constants[MI.getOperand(0).getReg()] = &MI;
        continue;
      default:
        break;
      }

      if (MI.isPHI() || MI.isCopy() || MI.isRegSequence())
        continue;

      DAG.Nodes.push_back(std::make_unique<Node>());
      Node *N = DAG.Nodes.back().get();
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
    }
  }

  if (!DAG.Constants.empty()) {
    dbgs() << "Constants:\n";
    for (const auto &[Reg, MI] : DAG.Constants)
      dbgs() << printReg(Reg, &TRI) << ": " << *MI;
    dbgs() << '\n';
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

void SchedulingDAG::clear() {
  Constants.clear();
  Nodes.clear();
}

void Node::printEdges(StringRef Name, const SmallVector<Edge> &Edges) const {
  if (Edges.empty())
    return;
  const TargetRegisterInfo &TRI =
      *Instrs.front()->getMF()->getSubtarget().getRegisterInfo();
  dbgs() << Name << ": ";
  for (const Edge &E : Edges) {
    dbgs() << printReg(E.Reg, &TRI);
    if (E.RC)
      dbgs() << "(" << TRI.getRegClassName(E.RC) << ")";
    dbgs() << ' ';
  }
  dbgs() << '\n';
}

} // namespace

char MOSSched::ID = 0;

INITIALIZE_PASS(MOSSched, DEBUG_TYPE, "MOS Instruction Scheduler", false, false)

MachineFunctionPass *llvm::createMOSSchedPass() { return new MOSSched(); }
