//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Isolate PHIs to put machine IR in conventional SSA form. Each PHI's inputs
/// and result are fresh registers with mutually disjoint live ranges, allowing
/// the allocator to give them a common backing location. Parallel copies before
/// predecessor terminators and after PHI bundles connect these registers to the
/// original values. All PHIs remain in SSA form, and the CFG is unchanged.
///
/// This implements the unoptimized copy insertion construction: it does not
/// coalesce copies. PHI inputs defined at or after the predecessor's first
/// terminator are not yet supported, since the exit copy would read them too
/// early.
///
//===----------------------------------------------------------------------===//

#include "MOSConventionalSSA.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-conventional-ssa"

using namespace llvm;

namespace {

class MOSConventionalSSA : public MachineFunctionPass {
public:
  static char ID;

  MOSConventionalSSA() : MachineFunctionPass(ID) {
    initializeMOSConventionalSSAPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  // Keep operand flags and subregisters when moving PHI operands into copies.
  struct Copy {
    MachineOperand Def;
    MachineOperand Use;
  };

  void isolatePHIs(MachineBasicBlock &MBB);
  void insertExitCopies(MachineBasicBlock &MBB, ArrayRef<Copy> Copies);
  void insertParallelCopy(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator InsertPt,
                          ArrayRef<Copy> Copies, const DebugLoc &DL);

  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  // One exit PCOPY per predecessor, shared by all its successors.
  IndexedMap<SmallVector<Copy, 0>, MBB2NumberFunctor> ExitCopies;
};

bool MOSConventionalSSA::runOnMachineFunction(MachineFunction &MF) {
  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  ExitCopies.clear();
  ExitCopies.resize(MF.getNumBlockIDs());

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF) {
    if (MBB.phis().empty())
      continue;
    isolatePHIs(MBB);
    Changed = true;
  }

  for (MachineBasicBlock &MBB : MF) {
    ArrayRef<Copy> Copies = ExitCopies[&MBB];
    if (!Copies.empty())
      insertExitCopies(MBB, Copies);
  }
  return Changed;
}

void MOSConventionalSSA::isolatePHIs(MachineBasicBlock &MBB) {
  SmallVector<Copy> EntryCopies;
  for (MachineInstr &PHI : MBB.phis()) {
    MachineOperand &Def = PHI.getOperand(0);
    Register Result = MRI->cloneVirtualRegister(Def.getReg());
    EntryCopies.push_back({Def, MachineOperand::CreateReg(Result, false)});
    Def.setReg(Result);
    Def.setIsDead(false);
  }
  // Keep result copies parallel, including when PHIs exchange their previous
  // results on a backedge.
  insertParallelCopy(MBB, MBB.getFirstNonPHI(), EntryCopies,
                     MBB.front().getDebugLoc());

  for (MachineInstr &PHI : MBB.phis()) {
    assert(PHI.getNumOperands() == 1 + 2 * MBB.pred_size() &&
           "expected one PHI input per predecessor");
    for (unsigned I = 1, E = PHI.getNumOperands(); I != E; I += 2) {
      MachineOperand &Use = PHI.getOperand(I);
      MachineBasicBlock *Pred = PHI.getOperand(I + 1).getMBB();
      Register Input = MRI->cloneVirtualRegister(PHI.getOperand(0).getReg());
      MRI->clearKillFlags(Use.getReg());
      ExitCopies[Pred].push_back({MachineOperand::CreateReg(Input, true), Use});
      Use.setReg(Input);
      Use.setSubReg(0);
      Use.setIsUndef(false);
      Use.setIsKill(false);
    }
  }
}

void MOSConventionalSSA::insertExitCopies(MachineBasicBlock &MBB,
                                          ArrayRef<Copy> Copies) {
  assert(llvm::none_of(MBB,
                       [](const MachineInstr &MI) {
                         return MI.getOpcode() == TargetOpcode::INLINEASM_BR;
                       }) &&
         "inline asm callbr is unsupported by MOS GlobalISel");
  auto InsertPt = MBB.getFirstTerminator();
  // PHI inputs are logically read on edges, but these copies execute before the
  // first terminator. Check that moving each read earlier does not precede its
  // definition. Undef operands do not read their named register.
  for (const Copy &C : Copies) {
    if (C.Use.isUndef())
      continue;
    const MachineInstr *Source = MRI->getVRegDef(C.Use.getReg());
    if (Source->getParent() == &MBB &&
        llvm::any_of(make_range(InsertPt, MBB.end()),
                     [&](const MachineInstr &MI) { return &MI == Source; }))
      report_fatal_error("MOSConventionalSSA cannot copy a PHI input defined "
                         "at or after the first terminator in its source block",
                         /*gen_crash_diag=*/false);
  }
  insertParallelCopy(MBB, InsertPt, Copies, DebugLoc());
}

void MOSConventionalSSA::insertParallelCopy(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator InsertPt,
    ArrayRef<Copy> Copies, const DebugLoc &DL) {
  MachineInstrBuilder MIB = BuildMI(MBB, InsertPt, DL, TII->get(MOS::PCOPY));
  for (const Copy &C : Copies)
    MIB.add(C.Def);
  for (const Copy &C : Copies)
    MIB.add(C.Use);
}

} // namespace

char MOSConventionalSSA::ID = 0;
INITIALIZE_PASS(MOSConventionalSSA, DEBUG_TYPE, "MOS Conventional SSA", false,
                false)

MachineFunctionPass *llvm::createMOSConventionalSSAPass() {
  return new MOSConventionalSSA;
}
