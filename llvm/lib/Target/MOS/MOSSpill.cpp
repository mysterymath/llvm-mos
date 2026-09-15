//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Prepare conventional SSA live ranges for imaginary register assignment.
///
/// MOSSpill is responsible for splitting and spilling ranges until conservative
/// imaginary demand permits treescan assignment. Demand ignores fixed physical
/// locations, which MOSImagRegAssign handles through local repair. Both passes
/// use the same value and reservation interference rules. The boundary between
/// them is SSA MIR; this pass does not assign locations in VirtRegMap.
///
/// Currently this pass validates demand and diagnoses cases requiring spills.
/// It does not insert spills or guarantee scratch space for every local repair;
/// MOSImagRegAssign still diagnoses unsupported repairs requiring extra
/// storage.
///
//===----------------------------------------------------------------------===//

#include "MOSSpill.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSImagRegAllocUtils.h"
#include "MOSRegisterInfo.h"
#include "MOSValueNumbering.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-spill"

using namespace llvm;

namespace {

class MOSSpill : public MachineFunctionPass {
public:
  static char ID;
  MOSSpill();
  bool runOnMachineFunction(MachineFunction &F) override;
  MachineFunctionProperties getRequiredProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  void checkPressure();
  void checkMBBPressure(MachineBasicBlock &MBB);
  void checkMIPressure(MachineInstr &MI);
  void checkDefPressure(MachineInstr &MI, Register R);
  void checkClobbers(MachineInstr &MI);

  // Test demand before inserting R. Failure leaves the live set unchanged.
  // Reservation members inherit the capacity guaranteed at their root.
  [[nodiscard]] bool tryAddLiveReg(Register R);
  void removeLiveReg(Register R);
  void releaseKilledUses(const MachineInstr &MI);
  void removeClobbers(const MachineOperand &RegMask);
  void releaseDeadDefs(const MachineInstr &MI);
  // With B conflicting Imag8s and P Imag16s, require 1+B+2P locations
  // for an Imag8, or 1+B+P for an Imag16. Bytes can block separate pairs.
  unsigned requiredCapacity(Register R) const;
  unsigned capacity(Register R) const {
    return RCI
        ->getOrder(TRI->getRegSizeInBits(R, *MRI) == 16 ? &MOS::Imag16RegClass
                                                        : &MOS::Imag8RegClass)
        .size();
  }

  MachineFunction *MF = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  const MachineDominatorTree *MDT = nullptr;
  const MOSValueNumbering *ValueNumbers = nullptr;

  // Live ranges requiring imaginary storage, before any locations are chosen.
  // Physical imaginary ranges contribute anonymous demand. An Imag16 is one
  // entry; a partial physical kill leaves its surviving Imag8 byte instead.
  SparseBitVector<> LiveRegs;
};

MOSSpill::MOSSpill() : MachineFunctionPass(ID) {
  initializeMOSSpillPass(*PassRegistry::getPassRegistry());
}

bool MOSSpill::runOnMachineFunction(MachineFunction &F) {
  MF = &F;
  MRI = &F.getRegInfo();
  TRI = F.getSubtarget().getRegisterInfo();
  F.getRegInfo().freezeReservedRegs();
  RCI = &getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI();
  LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  ValueNumbers = &getAnalysis<MOSValueNumberingWrapperPass>().valueNumbers();
  bool Changed = recomputeLiveIns(F.front());
  checkPressure();
  return Changed;
}

MachineFunctionProperties MOSSpill::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

void MOSSpill::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineRegisterClassInfoWrapperPass>();
  AU.addRequired<MOSValueNumberingWrapperPass>();
  // Only the entry live-in metadata changes while spill insertion is absent.
  AU.setPreservesAll();
}

void MOSSpill::checkPressure() {
  // Physical live-ins are confined to the entry block at this stage.
  LivePhysRegs EntryLiveRegs(*TRI);
  EntryLiveRegs.addLiveInsNoPristines(MF->front());
  LiveRegs.clear();
  // Seed pairs first so that the conservative bound does not assume that the
  // incoming bytes were scattered across otherwise available pairs.
  for (const TargetRegisterClass *RC :
       {&MOS::Imag16RegClass, &MOS::Imag8RegClass}) {
    for (MCPhysReg R : RCI->getOrder(RC)) {
      // LivePhysRegs includes the subregisters of each live Imag16.
      if (!EntryLiveRegs.contains(R) ||
          llvm::any_of(TRI->superregs(R), [&](MCPhysReg Super) {
            return EntryLiveRegs.contains(Super) &&
                   mos::needsImagReg(Super, *MF, *ValueNumbers);
          }))
        continue;
      if (!tryAddLiveReg(R))
        report_fatal_error("MOSSpill cannot accommodate entry live-ins",
                           /*GenCrashDiag=*/false);
    }
  }
  SmallVector<std::pair<const MachineDomTreeNode *, SparseBitVector<>>>
      LiveOuts;
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock &MBB = *Node->getBlock();
    while (!LiveOuts.empty() && LiveOuts.back().first != Node->getIDom())
      LiveOuts.pop_back();
    if (!LiveOuts.empty()) {
      LiveRegs = LiveOuts.back().second;
      for (auto I = LiveRegs.begin(), E = LiveRegs.end(); I != E;) {
        Register R = *I++;
        if (!LV->isLiveIn(R, MBB))
          LiveRegs.reset(R);
      }
    }
    checkMBBPressure(MBB);
    assert(llvm::all_of(LiveRegs, [](Register R) { return R.isVirtual(); }) &&
           "physical register live out of basic block");
    LiveOuts.emplace_back(Node, LiveRegs);
  }
}

void MOSSpill::checkMBBPressure(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    checkMIPressure(MI);
  }
}

void MOSSpill::checkMIPressure(MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isEarlyClobber())
      checkDefPressure(MI, MO.getReg());
  releaseKilledUses(MI);

  checkClobbers(MI);
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      removeClobbers(MO);

  for (const MachineOperand &MO : MI.all_defs())
    if (!MO.isEarlyClobber())
      checkDefPressure(MI, MO.getReg());

  releaseDeadDefs(MI);
}

void MOSSpill::checkDefPressure(MachineInstr &MI, Register R) {
  if (!mos::needsImagReg(R, *MF, *ValueNumbers))
    return;
  if (!tryAddLiveReg(R)) {
    bool IsImag16 = TRI->getRegSizeInBits(R, MF->getRegInfo()) == 16;
    errs() << "MOSSpill: cannot prove imaginary register assignability in "
           << MF->getName() << ", bb." << MI.getParent()->getNumber() << " for "
           << (IsImag16 ? "Imag16" : "Imag8") << ": requires "
           << requiredCapacity(R) << ", " << capacity(R)
           << " available locations\n"
           << MI;
    report_fatal_error("MOSSpill spill insertion is not implemented",
                       /*GenCrashDiag=*/false);
  }
}

void MOSSpill::checkClobbers(MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
      if (MO.clobbersPhysReg(R))
        checkDefPressure(MI, R);
  }
}

bool MOSSpill::tryAddLiveReg(Register R) {
  if (LiveRegs.test(R))
    return true;
  // PHI inputs and results inherit the assignment guaranteed at their root's
  // definition; they introduce no new imaginary assignment to check.
  Register Root = mos::getImagReservationRoot(R, *MRI);
  if ((!Root || R == Root) && requiredCapacity(R) > capacity(R))
    return false;
  if (R.isPhysical())
    removeLiveReg(R);
  LiveRegs.set(R);
  return true;
}

void MOSSpill::removeLiveReg(Register R) {
  if (LiveRegs.test(R)) {
    LiveRegs.reset(R);
    return;
  }
  if (R.isVirtual() || !R)
    return;
  if (MOS::Imag16RegClass.contains(R)) {
    removeLiveReg(TRI->getSubReg(R, MOS::sublo));
    removeLiveReg(TRI->getSubReg(R, MOS::subhi));
  } else if (MOS::Imag8RegClass.contains(R)) {
    for (MCPhysReg Super : TRI->superregs(R)) {
      if (!MOS::Imag16RegClass.contains(Super) || !LiveRegs.test(Super))
        continue;
      // Only this byte died or was overwritten. Preserve the other byte's
      // lifetime, now independently of its former pair.
      removeLiveReg(Super);
      Register Lo = TRI->getSubReg(Super, MOS::sublo);
      Register Hi = TRI->getSubReg(Super, MOS::subhi);
      LiveRegs.set(R == Lo ? Hi : Lo);
      break;
    }
  }
}

void MOSSpill::releaseKilledUses(const MachineInstr &MI) {
  // LiveVariables accounts for PHI edge uses in its kill flags.
  if (MI.isPHI())
    return;
  for (const MachineOperand &MO : MI.all_uses())
    if (MO.isKill())
      removeLiveReg(MO.getReg());
}

void MOSSpill::removeClobbers(const MachineOperand &RegMask) {
  for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
    if (RegMask.clobbersPhysReg(R))
      removeLiveReg(R);
}

void MOSSpill::releaseDeadDefs(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isDead())
      removeLiveReg(MO.getReg());
}

unsigned MOSSpill::requiredCapacity(Register R) const {
  bool IsImag16 = TRI->getRegSizeInBits(R, *MRI) == 16;
  unsigned Required = 1;
  for (Register LiveReg : LiveRegs) {
    if (R == LiveReg)
      continue;
    // Physical demand can block any location. For virtual ranges, reservations
    // only block their required exits; equal whole values can share anywhere.
    if (R.isVirtual() && LiveReg.isVirtual()) {
      if (R == mos::getImagReservationRoot(R, *MRI)) {
        if (!mos::overlapsImagReservation(R, LiveReg, *MRI, *LV))
          continue;
      } else if (LiveReg == mos::getImagReservationRoot(LiveReg, *MRI)) {
        if (!mos::overlapsImagReservation(LiveReg, R, *MRI, *LV))
          continue;
      } else if (ValueNumbers->getValueNumber(R) ==
                 ValueNumbers->getValueNumber(LiveReg)) {
        continue;
      }
    }
    unsigned Occupied =
        !IsImag16 && TRI->getRegSizeInBits(LiveReg, *MRI) == 16 ? 2 : 1;
    if (R.isPhysical() && LiveReg.isPhysical() && TRI->regsOverlap(R, LiveReg))
      // A pair definition replaces its byte aliases. A byte definition leaves
      // the other byte of an overlapping pair live.
      Occupied = IsImag16 ? 0 : 1;
    Required += Occupied;
  }
  return Required;
}

} // namespace

char MOSSpill::ID = 0;
INITIALIZE_PASS_BEGIN(MOSSpill, DEBUG_TYPE, "MOS imaginary register spilling",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MOSValueNumberingWrapperPass)
INITIALIZE_PASS_END(MOSSpill, DEBUG_TYPE, "MOS imaginary register spilling",
                    false, false)

MachineFunctionPass *llvm::createMOSSpillPass() { return new MOSSpill; }
