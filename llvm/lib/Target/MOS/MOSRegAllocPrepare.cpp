//===-- MOSRegAllocPrepare.cpp - Prepare for MOS RegAlloc -----------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocation preparation pass.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAllocPrepare.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-regalloc-prepare"

using namespace llvm;

namespace {

class MOSRegAllocPrepare : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAllocPrepare() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPreparePass(*PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTreeWrapperPass>();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineRegisterInfo *MRI;
  const MachineDominatorTree *MDT;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  void expandRegSequences();
  void ensureEqualTieRCs();
  void killTieUses();
  void splitPHIEdges(MachineFunction &MF);
};

bool MOSRegAllocPrepare::runOnMachineFunction(MachineFunction &MF) {
  MRI = &MF.getRegInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();

  killTieUses();
  ensureEqualTieRCs();
  expandRegSequences();
  // Do this last, since it invalidates MDT.
  splitPHIEdges(MF);

  return true;
}

// Expand away REG_SEQUENCE operations. This makes a pair of 8-bit values
// available in an imaginary register pair. It's important that the two values
// needn't be alive at the same time in their source registers; that would
// unnecessarily increase register pressure. Instead, the sequence operation can
// kill each half separately. This is done by allocating the pair immediately
// after the first def, then redefining it after the second def. Note that this
// presumes that imag16 pairs are easier to come by than the sources, which
// should usually be true. There's a lot more optimization we can do here, but
// this should suffice for a first pass.
void MOSRegAllocPrepare::expandRegSequences() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->def_empty(R))
      continue;
    MachineInstr &MI = *MRI->getVRegDef(R);
    if (!MI.isRegSequence())
      continue;

    MachineOperand *EarlierDefMO = nullptr;
    unsigned EarlierSubReg;
    MachineOperand *LaterDefMO;
    unsigned LaterSubReg;

    for (unsigned J = 1, JE = MI.getNumOperands(); J != JE; J += 2) {
      assert(MI.getOperand(J + 1).getImm() == MOS::sublo ||
             MI.getOperand(J + 1).getImm() == MOS::subhi &&
                 "only support imag reg pairs");
      Register Child = MI.getOperand(J).getReg();
      MachineOperand *DefMO = MRI->getOneDef(Child);
      if (!EarlierDefMO ||
          MDT->dominates(DefMO->getParent(), EarlierDefMO->getParent())) {
        LaterDefMO = EarlierDefMO;
        LaterSubReg = EarlierSubReg;
        EarlierDefMO = DefMO;
        EarlierSubReg = MI.getOperand(J + 1).getImm();
      } else {
        LaterDefMO = DefMO;
        LaterSubReg = MI.getOperand(J + 1).getImm();
      }
    }
    assert(EarlierDefMO && LaterDefMO &&
           MDT->dominates(EarlierDefMO->getParent(), LaterDefMO->getParent()) &&
           "earlier REG_SEQUENCE input def does not dominate later");

    MachineIRBuilder Builder(
        *EarlierDefMO->getParent()->getParent(),
        std::next(MachineBasicBlock::iterator(EarlierDefMO->getParent())));
    auto Earlier = Builder.buildInstr(
        MOS::INSERT_SUBREG, {&MOS::Imag16RegClass},
        {Builder.buildInstr(MOS::INIT_UNDEF, {&MOS::Imag16RegClass}, {}),
         EarlierDefMO->getReg(), int64_t{EarlierSubReg}});
    Builder.setInsertPt(
        *LaterDefMO->getParent()->getParent(),
        std::next(MachineBasicBlock::iterator(LaterDefMO->getParent())));
    Builder.buildInstr(MOS::INSERT_SUBREG, {R},
                       {Earlier, LaterDefMO->getReg(), int64_t{LaterSubReg}});

    MI.eraseFromParent();
  }
}

// It's not sensible when allocating directly in SSA form for a tied RC def to
// have a different register class from its corresponding tied use, since the
// allocation of the use must be directly reused for the def. Accordingly, this
// establishes the invariant that tie uses and tie defs always have exactly the
// same RC.
void MOSRegAllocPrepare::ensureEqualTieRCs() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    MachineOperand *MO = MRI->getOneDef(R);
    if (!MO || !MO->isTied())
      continue;
    MachineInstr &MI = *MO->getParent();
    const TargetRegisterClass *DstRC = MRI->getRegClass(R);
    MachineOperand &SrcMO =
        MI.getOperand(MI.findTiedOperandIdx(MO->getOperandNo()));
    Register SrcR = SrcMO.getReg();
    if (DstRC == MRI->getRegClass(SrcR))
      continue;
    Register Tmp = MRI->createVirtualRegister(DstRC);
    MachineIRBuilder Builder(MI);
    Builder.buildCopy(Tmp, SrcR);
    SrcMO.setReg(Tmp);
  }
}

// Tied defs will reuse the allocation of their cooresponding uses. This is only
// possible if tied uses are always kills. Accordingly, insert copies to ensure
// this.
void MOSRegAllocPrepare::killTieUses() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    MachineOperand *MO = MRI->getOneDef(R);
    if (!MO || !MO->isTied())
      continue;
    MachineInstr &MI = *MO->getParent();
    MachineOperand &SrcMO =
        MI.getOperand(MI.findTiedOperandIdx(MO->getOperandNo()));
    if (MRI->hasOneNonDBGUse(SrcMO.getReg()))
      continue;
    MachineIRBuilder Builder(MI);
    const TargetRegisterClass *RC =
        MI.getRegClassConstraint(SrcMO.getOperandNo(), TII, TRI);
    if (!RC)
      RC = TRI->getLargestLegalSuperClass(MRI->getRegClass(R), MRI->getMF());
    SrcMO.setReg(Builder.buildCopy(RC, R).getReg(0));
  }
}

void MOSRegAllocPrepare::splitPHIEdges(MachineFunction &MF) {
  // TODO: Jump tables
  //
  // Note: We shouldn't need to do anything to handle indirect branches, since
  // CodeGenPrepare splits them. This should mean that every indirect branch
  // targets only PHI-less blocks.

  for (MachineBasicBlock &MBB : MF) {
    MapVector<const MachineBasicBlock *, MachineBasicBlock *> PHIBlocks;
    for (MachineInstr &MI : make_early_inc_range(MBB.phis())) {
      for (unsigned I = 1, E = MI.getNumOperands(); I < E; I += 2) {
        Register Use = MI.getOperand(I).getReg();
        MachineBasicBlock &UseMBB = *MI.getOperand(I + 1).getMBB();

        Register UseCopy = MRI->createVirtualRegister(
            TRI->getLargestLegalSuperClass(MRI->getRegClass(Use), MF));
        MI.getOperand(I).setReg(UseCopy);

        auto It = PHIBlocks.find(&UseMBB);
        if (It == PHIBlocks.end()) {
          MachineBasicBlock &PHIBlock =
              *MF.CreateMachineBasicBlock(MBB.getBasicBlock());
          It = PHIBlocks.try_emplace(&UseMBB, &PHIBlock).first;
          MF.insert(MBB.getIterator(), &PHIBlock);
          TII->insertUnconditionalBranch(PHIBlock, &MBB, MI.getDebugLoc());
          PHIBlock.addSuccessor(&MBB);
          MachineIRBuilder Builder(PHIBlock, PHIBlock.begin());
          Builder.buildInstr(MOS::PCOPY,
                             {&MOS::Imag16RegClass, &MOS::Imag16RegClass}, {});
          MachineBasicBlock *TBB;
          MachineBasicBlock *FBB;
          SmallVector<MachineOperand, 2> Cond;
          bool Failure = TII->analyzeBranch(UseMBB, TBB, FBB, Cond);
          if (Failure)
            report_fatal_error("could not split PHI edges in regalloc");
          DebugLoc DL = UseMBB.findBranchDebugLoc();
          TII->removeBranch(UseMBB);
          if (TBB == &MBB) {
            UseMBB.replaceSuccessor(TBB, &PHIBlock);
            TBB = &PHIBlock;
          }
          if (FBB == &MBB) {
            UseMBB.replaceSuccessor(FBB, &PHIBlock);
            FBB = &PHIBlock;
          }
          if (!TBB && !FBB)
            UseMBB.replaceSuccessor(&MBB, &PHIBlock);
          TII->insertBranch(UseMBB, TBB, FBB, Cond, DL);
        }
        MachineBasicBlock &PHIBlock = *It->second;
        auto &PCopy = *PHIBlock.begin();
        PCopy.addOperand(MachineOperand::CreateReg(Use, /*isDef=*/false));
        PCopy.insert(PCopy.operands_begin() + PCopy.getNumExplicitDefs(),
                     {MachineOperand::CreateReg(UseCopy, /*isDef=*/true)});
        MI.getOperand(I + 1).setMBB(&PHIBlock);
      }
    }

    // Ensure that the PCOPY scratch registers conflict with the PCOPY defs,
    // that is, the PHI defs. (The actual PCOPY defs are mere artifacts.)
    if (!MBB.phis().empty()) {
      MachineIRBuilder Builder(MBB, MBB.getFirstNonPHI());
      auto Phi1 = Builder.buildInstr(MOS::PHI, {&MOS::Imag16RegClass}, {});
      auto Phi2 = Builder.buildInstr(MOS::PHI, {&MOS::Imag16RegClass}, {});
      for (auto &[UseMBB, PHIBlock] : PHIBlocks) {
        auto &PCopy = *PHIBlock->begin();
        Phi1.addUse(PCopy.getOperand(0).getReg());
        Phi1.addMBB(PHIBlock);
        Phi2.addUse(PCopy.getOperand(1).getReg());
        Phi2.addMBB(PHIBlock);
      }
    }
  }
}

} // namespace

char MOSRegAllocPrepare::ID = 0;

INITIALIZE_PASS_BEGIN(MOSRegAllocPrepare, DEBUG_TYPE,
                      "MOS Register Allocation Preparation", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(MOSRegAllocPrepare, DEBUG_TYPE,
                    "MOS Register Allocation Preparation", false, false)

MachineFunctionPass *llvm::createMOSRegAllocPreparePass() {
  return new MOSRegAllocPrepare;
}
