//===-- MOSMachineSSA.cpp - MOS Machine SSA -------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS machine SSA pass.
//
// The LLVM-MOS register allocator operates over SSA form, since it works in
// terms of values, not temporaries. Accordingly, this pass converts from MIR
// back to machine SSA. Ideally, the machine SSA would simply be preserved until
// this point, but too many prior passes are written against non-SSA IR.
//
//===----------------------------------------------------------------------===//

#include "MOSMachineSSA.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

#define DEBUG_TYPE "mos-machine-ssa"

using namespace llvm;

namespace {

class MOSMachineSSA : public MachineFunctionPass {
public:
  static char ID;

  MOSMachineSSA() : MachineFunctionPass(ID) {
    llvm::initializeMOSMachineSSAPass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  // The dominating defining SSA vreg at the current point for a given original
  // VReg. Given as a stack, since returning up the dominator tree should peel
  // back defs.
  DenseMap<Register, SmallVector<Register>> DominatingDefs;

  void insertPHIs(MachineFunction &MF);
  void renameRegs(MachineDomTreeNode &MDTN);
  void pruneDeadPHIs(MachineFunction &MF);
};

void MOSMachineSSA::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<MachineDominanceFrontier>();
  AU.addPreserved<MachineDominanceFrontier>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
}

bool MOSMachineSSA::runOnMachineFunction(MachineFunction &MF) {
  MachineDominatorTree &MDT = getAnalysis<MachineDominatorTree>();

  MF.getProperties().reset(MachineFunctionProperties::Property::NoPHIs);
  MF.getProperties().reset(
      MachineFunctionProperties::Property::TiedOpsRewritten);
  MF.getProperties().set(MachineFunctionProperties::Property::IsSSA);

  insertPHIs(MF);

  DominatingDefs.clear();
  renameRegs(*MDT.getRootNode());

  pruneDeadPHIs(MF);
  return true;
}

// Insert PHI instructions in the iterated dominance frontier of each
// definition.
void MOSMachineSSA::insertPHIs(MachineFunction &MF) {
  MachineDominanceFrontier &MDF = getAnalysis<MachineDominanceFrontier>();
  MachineRegisterInfo &MRI = MF.getRegInfo();
  for (unsigned I = 0, E = MRI.getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    SetVector<MachineBasicBlock *> Worklist;
    for (MachineInstr &MI : MRI.def_instructions(R)) {
      const auto &DomSet = MDF.find(MI.getParent())->second;
      for (MachineBasicBlock *MBB : DomSet)
        Worklist.insert(MBB);
    }
    for (unsigned I = 0; I < Worklist.size(); I++) {
      MachineBasicBlock *MBB = Worklist[I];
      MachineIRBuilder Builder(*MBB, MBB->begin());
      auto Phi = Builder.buildInstr(MOS::PHI).addDef(R);
      for (MachineBasicBlock *Pred : MBB->predecessors())
        Phi.addUse(R).addMBB(Pred);
      const auto &DomSet = MDF.find(MBB)->second;
      for (MachineBasicBlock *MBB : DomSet)
        Worklist.insert(MBB);
    }
  }
}

// Walk instructions and create new vregs for each definition.
void MOSMachineSSA::renameRegs(MachineDomTreeNode &MDTN) {
  MachineBasicBlock *MBB = MDTN.getBlock();
  MachineRegisterInfo &MRI = MBB->getParent()->getRegInfo();
  const TargetRegisterInfo &TRI = *MRI.getTargetRegisterInfo();

  // Keep track of the registers with new defs; these must be popped upon
  // return.
  DenseSet<Register> NewDefs;

  for (MachineInstr &MI : make_early_inc_range(*MBB)) {
    if (!MI.isPHI()) {
      for (MachineOperand &MO : MI.uses()) {
        if (!MO.isReg())
          continue;
        Register R = MO.getReg();
        if (!R.isVirtual())
          continue;
        MO.setReg(DominatingDefs[R].back());
      }
    }
    for (MachineOperand &MO : MI.defs()) {
      Register Orig = MO.getReg();
      if (!Orig.isVirtual())
        continue;

      SmallVector<Register> &Stack = DominatingDefs[Orig];
      Register New = MRI.cloneVirtualRegister(Orig);

      unsigned SubReg = MO.getSubReg();
      if (SubReg) {
        // Instead of having the destination be a subregister of a wide register
        // class, make it a full use of a narrow register class.
        Register Narrow = MRI.createVirtualRegister(
            TRI.getSubRegisterClass(MRI.getRegClass(Orig), SubReg));
        MO.setReg(Narrow);
        MO.setSubReg(0);

        // Generate the new vreg by inserting the narrow reg.
        MachineIRBuilder Builder(MI);
        Builder.setInsertPt(*MI.getParent(), std::next(MI.getIterator()));
        Register Prev;
        if (MO.isUndef()) {
          MO.setIsUndef(false);
          Prev = MRI.cloneVirtualRegister(Orig);
          Builder.buildInstr(MOS::IMPLICIT_DEF).addDef(Prev);
        } else {
          assert(!Stack.empty() &&
                 "Non-undef subregister def requires previous value.");
          Prev = Stack.back();
        }
        Builder.buildInstr(MOS::INSERT_SUBREG)
            .addDef(New)
            .addUse(Prev)
            .addUse(Narrow)
            .addImm(SubReg);
      } else {
        MO.setReg(New);
      }

      if (NewDefs.insert(Orig).second)
        Stack.push_back(New);
      else
        Stack.back() = New;
    }
  }

  for (MachineBasicBlock *SuccMBB : MBB->successors()) {
    for (MachineInstr &MI : SuccMBB->phis()) {
      for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
        if (MI.getOperand(I + 1).getMBB() == MBB) {
          SmallVector<Register> &Stack =
              DominatingDefs[MI.getOperand(I).getReg()];
          MI.getOperand(I).setReg(!Stack.empty() ? Stack.back() : Register());
        }
      }
    }
  }

  for (MachineDomTreeNode *C : MDTN.children())
    renameRegs(*C);

  for (Register R : NewDefs)
    DominatingDefs[R].pop_back();
}

void MOSMachineSSA::pruneDeadPHIs(MachineFunction &MF) {
  MachineRegisterInfo &MRI = MF.getRegInfo();

  SetVector<MachineInstr *> LivePHIs;

  // Ground the live PHIs with the PHIs that are used by at least one non-PHI.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB.phis()) {
      Register R = MI.getOperand(0).getReg();
      if (!MRI.use_nodbg_empty(R) &&
          llvm::any_of(MRI.use_nodbg_instructions(R),
                       [](const MachineInstr &MI) { return !MI.isPHI(); })) {
        LivePHIs.insert(&MI);
      }
    }
  }

  // Add any PHIs used by a live PHI.
  for (unsigned I = 0; I < LivePHIs.size(); ++I) {
    MachineInstr *MI = LivePHIs[I];
    for (unsigned I = 1, E = MI->getNumOperands(); I != E; I += 2) {
      Register R = MI->getOperand(I).getReg();
      if (!R.isVirtual())
        continue;
      MachineInstr *DefMI = MRI.getUniqueVRegDef(R);
      if (DefMI->isPHI())
        LivePHIs.insert(DefMI);
    }
  }

  // Delete any PHIs that don't meet the above criteria; they're dead.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : make_early_inc_range(MBB.phis()))
      if (!LivePHIs.contains(&MI))
        MI.eraseFromParent();
}

} // namespace

char MOSMachineSSA::ID = 0;

INITIALIZE_PASS(MOSMachineSSA, DEBUG_TYPE, "Convert MIR to SSA form", false,
                false)

MachineFunctionPass *llvm::createMOSMachineSSAPass() {
  return new MOSMachineSSA();
}
