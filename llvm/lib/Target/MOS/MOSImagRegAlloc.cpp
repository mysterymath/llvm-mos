//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Allocate imaginary registers before MOS register allocation.
///
/// MOSImagRegAlloc operates on SSA machine IR. Its contract is to make
/// imaginary backing registers assignable by inserting spills and reloads where
/// necessary, while preserving SSA form. Pressure is assessed assuming that
/// values need backing even if subsequent hardware register allocation may
/// eliminate that need.
///
/// Backing registers are assigned by a dominance-order treescan, independently
/// of operand constraints. MOSRegAlloc may keep copies elsewhere and repairs
/// constraints locally; live values return to their backing registers at block
/// boundaries. VirtRegMap carries these assignments to MOSRegAlloc. They may be
/// outside the vregs' operand classes; MOSRegAlloc realizes those constraints
/// instead of the ordinary virtual-register rewriter.
/// VirtRegMap's split ancestry also records whole-value equality for COPYs and
/// value-preserving splits. These roots do not merge backing assignments or
/// live ranges; MOSRegAlloc uses them to identify equal register contents.
///
/// This implementation handles Imag8 and Imag16 backing registers, with Imag8
/// backing for flags. Physical imaginary definitions contribute ordinary
/// demand; call clobbers contribute simultaneous dead definitions. Their
/// required physical locations are left to MOSRegAlloc. This pass does not yet
/// insert spills; failure of the conservative pressure test is diagnosed.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegAlloc.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-imag-regalloc"

using namespace llvm;

namespace {

// Counts of Imag8 and Imag16 values or locations. Flags need Imag8 backing.
struct ImagCounts {
  unsigned Imag8 = 0;
  unsigned Imag16 = 0;
};

// Active backing demand, independent of placement constraints. Both virtual
// and physical Imag16 definitions count as one Imag16. A partial kill of a
// physical Imag16 leaves its surviving byte as Imag8 demand. The caller selects
// values needing backing and advances their lifetimes.
class LiveRegisters {
public:
  void init(const MachineFunction &MF, const RegisterClassInfo &RCI) {
    MRI = &MF.getRegInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    this->RCI = &RCI;
    clear();
  }

  // Incorporate a definition, normalizing overlapping physical aliases.
  // Fail without changing the live set if the capacity bound cannot guarantee
  // placement. A register already present succeeds without changing the set.
  [[nodiscard]] bool insert(Register R);
  void erase(Register R);
  void clear() {
    Regs.clear();
    Counts = {};
  }

  auto begin() const { return Regs.begin(); }
  auto end() const { return Regs.end(); }

  // Sufficient total capacity for the live set including R, in locations of
  // R's backing size. With B Imag8s and P Imag16s, an Imag8 needs B + 2P;
  // an Imag16 needs B + P, since each Imag8 could occupy a different pair.
  unsigned getRequiredCapacity(Register R) const;
  unsigned getCapacity(Register R) const {
    return RCI
        ->getOrder(TRI->getRegSizeInBits(R, *MRI) == 16 ? &MOS::Imag16RegClass
                                                        : &MOS::Imag8RegClass)
        .size();
  }

private:
  unsigned &count(Register R) {
    return TRI->getRegSizeInBits(R, *MRI) == 16 ? Counts.Imag16 : Counts.Imag8;
  }

  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  SparseBitVector<> Regs;
  ImagCounts Counts;
};

class MOSImagRegAlloc : public MachineFunctionPass {
public:
  static char ID;
  MOSImagRegAlloc();

  bool runOnMachineFunction(MachineFunction &F) override;
  MachineFunctionProperties getRequiredProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  void recordCopyOrigins();
  // Check unconstrained demand. Failure is fatal until spill insertion exists.
  void checkPressure();
  void checkMBBPressure(MachineBasicBlock &MBB);
  void checkMIPressure(MachineInstr &MI);
  void assignBackingRegisters();
  void assignMBBBackingRegisters(MachineBasicBlock &MBB);

  void
  walkDominatorTree(void (MOSImagRegAlloc::*VisitBlock)(MachineBasicBlock &));

  void checkDefPressure(MachineInstr &MI, Register R);
  void assignBackingRegister(Register R);

  // Unused values and trivially rematerializable values need no backing.
  bool needsReg(Register R) const;
  // LiveVariables accounts for PHI edge uses in its kill flags.
  void removeKilledUses(const MachineInstr &MI);
  void removeDeadDefs(const MachineInstr &MI);
  void checkClobbers(MachineInstr &MI);
  void removeClobbers(const MachineInstr &MI);

  MachineFunction *MF = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  VirtRegMap *VRM = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  const MachineDominatorTree *MDT = nullptr;

  LiveRegisters LiveRegs;
};

MOSImagRegAlloc::MOSImagRegAlloc() : MachineFunctionPass(ID) {
  initializeMOSImagRegAllocPass(*PassRegistry::getPassRegistry());
}

bool MOSImagRegAlloc::runOnMachineFunction(MachineFunction &F) {
  MF = &F;
  TRI = F.getSubtarget().getRegisterInfo();
  F.getRegInfo().freezeReservedRegs();
  RCI = &getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  LiveRegs.init(F, *RCI);
  recordCopyOrigins();
  checkPressure();
  assignBackingRegisters();
  return false;
}

MachineFunctionProperties MOSImagRegAlloc::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

void MOSImagRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineRegisterClassInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addPreserved<LiveVariablesWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
  AU.setPreservesCFG();
}

// Record in VirtRegMap that COPY destinations are split from sources.  This is
// maintained transitively, such that the destination of a chain of copies is
// considered split from its ultimate source.  The mapping must be maintained
// for use by MOSRegAlloc.
void MOSImagRegAlloc::recordCopyOrigins() {
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  EquivalenceClasses<Register> ECs;
  // Registers that are not full copies of other registers.
  SmallVector<Register> Roots;
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register R = Register::index2VirtReg(I);
    Register Source = R;
    const MachineInstr *Def = MRI.getVRegDef(R);
    if (Def && Def->isFullCopy() && !Def->getOperand(1).isUndef()) {
      Register CopySource = Def->getOperand(1).getReg();
      if (CopySource.isVirtual() &&
          TRI->getRegSizeInBits(*MRI.getRegClass(R)) ==
              TRI->getRegSizeInBits(*MRI.getRegClass(CopySource)))
        Source = CopySource;
    }
    if (Source == R)
      Roots.push_back(R);
    ECs.unionSets(Source, R);
  }
  for (Register Root : Roots)
    for (Register R : ECs.members(Root))
      if (R != Root)
        VRM->setIsSplitFromReg(R, Root);
}

void MOSImagRegAlloc::checkPressure() {
  // Physical live-ins are confined to the entry block at this stage.
  LivePhysRegs EntryLiveRegs;
  computeLiveIns(EntryLiveRegs, MF->front());
  LiveRegs.clear();
  // Seed pairs first so that the conservative bound does not assume that the
  // incoming bytes were scattered across otherwise available pairs.
  for (const TargetRegisterClass *RC :
       {&MOS::Imag16RegClass, &MOS::Imag8RegClass}) {
    for (MCPhysReg R : RCI->getOrder(RC)) {
      // LivePhysRegs includes the subregisters of each live Imag16.
      if (!EntryLiveRegs.contains(R) ||
          llvm::any_of(TRI->superregs(R), [&](MCPhysReg Super) {
            return EntryLiveRegs.contains(Super) && needsReg(Super);
          }))
        continue;
      if (!LiveRegs.insert(R))
        report_fatal_error("MOSImagRegAlloc cannot accommodate entry live-ins",
                           /*GenCrashDiag=*/false);
    }
  }
  walkDominatorTree(&MOSImagRegAlloc::checkMBBPressure);
}

void MOSImagRegAlloc::checkMBBPressure(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    checkMIPressure(MI);
  }
}

void MOSImagRegAlloc::checkMIPressure(MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isEarlyClobber())
      checkDefPressure(MI, MO.getReg());
  removeKilledUses(MI);

  checkClobbers(MI);
  removeClobbers(MI);

  for (const MachineOperand &MO : MI.all_defs())
    if (!MO.isEarlyClobber())
      checkDefPressure(MI, MO.getReg());

  removeDeadDefs(MI);
}

void MOSImagRegAlloc::checkDefPressure(MachineInstr &MI, Register R) {
  if (!needsReg(R))
    return;
  if (!LiveRegs.insert(R)) {
    bool IsImag16 = TRI->getRegSizeInBits(R, MF->getRegInfo()) == 16;
    errs() << "MOSImagRegAlloc: cannot prove backing assignability in "
           << MF->getName() << ", bb." << MI.getParent()->getNumber() << " for "
           << (IsImag16 ? "Imag16" : "Imag8") << " backing: requires "
           << LiveRegs.getRequiredCapacity(R) << ", " << LiveRegs.getCapacity(R)
           << " available locations\n"
           << MI;
    report_fatal_error("MOSImagRegAlloc spill insertion is not implemented",
                       /*GenCrashDiag=*/false);
  }
}

void MOSImagRegAlloc::checkClobbers(MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
      if (MO.clobbersPhysReg(R))
        checkDefPressure(MI, R);
  }
}

void MOSImagRegAlloc::removeClobbers(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
      if (MO.clobbersPhysReg(R))
        LiveRegs.erase(R);
  }
}

void MOSImagRegAlloc::assignBackingRegisters() {
  LiveRegs.clear();
  walkDominatorTree(&MOSImagRegAlloc::assignMBBBackingRegisters);
}

void MOSImagRegAlloc::assignMBBBackingRegisters(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    if (MI.isDebugInstr())
      continue;
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.isEarlyClobber())
        assignBackingRegister(MO.getReg());
    removeKilledUses(MI);
    for (const MachineOperand &MO : MI.all_defs())
      if (!MO.isEarlyClobber())
        assignBackingRegister(MO.getReg());
    removeDeadDefs(MI);
  }
}

void MOSImagRegAlloc::walkDominatorTree(
    void (MOSImagRegAlloc::*VisitBlock)(MachineBasicBlock &)) {
  SmallVector<std::pair<const MachineDomTreeNode *, LiveRegisters>> DomLiveRegs;
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock &MBB = *Node->getBlock();
    while (!DomLiveRegs.empty() && DomLiveRegs.back().first != Node->getIDom())
      DomLiveRegs.pop_back();
    if (!DomLiveRegs.empty()) {
      LiveRegs = DomLiveRegs.back().second;
      // Trim the inherited virtual live ranges to this block's live-ins.
      for (auto I = LiveRegs.begin(), E = LiveRegs.end(); I != E;) {
        Register R = *I;
        ++I;
        if (!LV->isLiveIn(R, MBB))
          LiveRegs.erase(R);
      }
    }

    (this->*VisitBlock)(MBB);
    assert(llvm::all_of(LiveRegs, [](Register R) { return R.isVirtual(); }) &&
           "physical register live out of basic block");
    DomLiveRegs.emplace_back(Node, LiveRegs);
  }
}

void MOSImagRegAlloc::assignBackingRegister(Register R) {
  if (!R.isVirtual() || !needsReg(R))
    return;
  auto Order = RCI->getOrder(
      TRI->getRegSizeInBits(*MF->getRegInfo().getRegClass(R)) == 16
          ? &MOS::Imag16RegClass
          : &MOS::Imag8RegClass);
  auto Available = llvm::find_if(Order, [&](MCPhysReg Candidate) {
    return llvm::none_of(LiveRegs, [&](Register LiveReg) {
      MCPhysReg BackingReg = VRM->getPhys(LiveReg);
      return BackingReg && TRI->regsOverlap(Candidate, BackingReg);
    });
  });
  assert(Available != Order.end() &&
         "pressure check guaranteed a backing register");
  bool Inserted = LiveRegs.insert(R);
  assert(Inserted && "pressure check guaranteed insertion during assignment");
  (void)Inserted;
  VRM->assignVirt2Phys(R, *Available);
}

bool MOSImagRegAlloc::needsReg(Register R) const {
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  if (R.isPhysical())
    return R &&
           (MOS::Imag8RegClass.contains(R) ||
            MOS::Imag16RegClass.contains(R)) &&
           !MRI.isReserved(R);
  if (MRI.use_nodbg_empty(R))
    return false;
  unsigned Bits = TRI->getRegSizeInBits(*MRI.getRegClass(R)).getFixedValue();
  if (Bits != 1 && Bits != 8 && Bits != 16)
    report_fatal_error(
        "MOSImagRegAlloc only supports Imag8 and Imag16 backing registers",
        /*GenCrashDiag=*/false);
  const MachineInstr *Def = MRI.getVRegDef(VRM->getOriginal(R));
  return !Def ||
         !MF->getSubtarget().getInstrInfo()->isTriviallyReMaterializable(*Def);
}

void MOSImagRegAlloc::removeKilledUses(const MachineInstr &MI) {
  if (MI.isPHI())
    return;
  for (const MachineOperand &MO : MI.all_uses())
    if (MO.isKill())
      LiveRegs.erase(MO.getReg());
}

void MOSImagRegAlloc::removeDeadDefs(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isDead())
      LiveRegs.erase(MO.getReg());
}

bool LiveRegisters::insert(Register R) {
  if (Regs.test(R))
    return true;
  if (getRequiredCapacity(R) > getCapacity(R))
    return false;
  if (R.isPhysical())
    erase(R);
  Regs.set(R);
  ++count(R);
  return true;
}

unsigned LiveRegisters::getRequiredCapacity(Register R) const {
  bool IsImag16 = TRI->getRegSizeInBits(R, *MRI) == 16;
  unsigned Required =
      Counts.Imag8 + (IsImag16 ? Counts.Imag16 : 2 * Counts.Imag16);
  if (Regs.test(R))
    return Required;
  ++Required;
  if (R.isPhysical()) {
    if (IsImag16) {
      // The pair replaces any live byte aliases.
      Required -= Regs.test(TRI->getSubReg(R, MOS::sublo));
      Required -= Regs.test(TRI->getSubReg(R, MOS::subhi));
    } else {
      // Replacing one byte of a live pair does not add byte demand.
      for (MCPhysReg Super : TRI->superregs(R))
        if (MOS::Imag16RegClass.contains(Super) && Regs.test(Super)) {
          --Required;
          break;
        }
    }
  }
  return Required;
}

void LiveRegisters::erase(Register R) {
  if (Regs.test(R)) {
    Regs.reset(R);
    --count(R);
    return;
  }
  if (R.isVirtual() || !R)
    return;
  if (MOS::Imag16RegClass.contains(R)) {
    erase(TRI->getSubReg(R, MOS::sublo));
    erase(TRI->getSubReg(R, MOS::subhi));
  } else if (MOS::Imag8RegClass.contains(R)) {
    for (MCPhysReg Super : TRI->superregs(R)) {
      if (!MOS::Imag16RegClass.contains(Super) || !Regs.test(Super))
        continue;
      // Only this byte died or was overwritten. Preserve the other byte's
      // lifetime, now independently of its former pair.
      erase(Super);
      Register Lo = TRI->getSubReg(Super, MOS::sublo);
      Register Hi = TRI->getSubReg(Super, MOS::subhi);
      Regs.set(R == Lo ? Hi : Lo);
      ++Counts.Imag8;
      break;
    }
  }
}

} // namespace

char MOSImagRegAlloc::ID = 0;
INITIALIZE_PASS_BEGIN(MOSImagRegAlloc, DEBUG_TYPE,
                      "MOS imaginary register allocation", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_END(MOSImagRegAlloc, DEBUG_TYPE,
                    "MOS imaginary register allocation", false, false)
MachineFunctionPass *llvm::createMOSImagRegAllocPass() {
  return new MOSImagRegAlloc;
}
