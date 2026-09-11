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
/// MOSImagRegAlloc operates on conventional SSA machine IR. Its contract is to
/// make imaginary backing registers assignable by inserting spills and reloads
/// where necessary, while preserving SSA form. Pressure is assessed assuming
/// that values need backing even if subsequent hardware register allocation may
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
/// Each isolated PHI and its inputs share the backing register of the explicit
/// IMPLICIT_DEF reservation supplied by MOSConventionalSSA. This reserves a
/// location before the PHI's incoming values are defined. The location needs to
/// hold those values only from the predecessor exit PCOPYs through the PHI to
/// the entry PCOPY. Other values may reuse it if their lifetimes do not overlap
/// those intervals.
///
/// Before assigning new backing, the pass scans live registers to check that
/// enough locations are available. Reservation IMPLICIT_DEFs predict conflicts
/// at those future boundaries; other registers use their ordinary SSA lifetimes.
/// Assignment uses the same conflict rule. Conflicts are conservative: distinct
/// SSA registers may interfere even when they contain copies of the same value.
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

// Active backing demand, independent of placement constraints. Both virtual
// and physical Imag16 definitions count as one Imag16. A partial kill of a
// physical Imag16 leaves its surviving byte as Imag8 demand. The caller selects
// values needing backing and advances their lifetimes.
class LiveRegisters {
public:
  void init(const MachineFunction &MF, const RegisterClassInfo &RCI,
            LiveVariables &LV) {
    MRI = &MF.getRegInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    this->RCI = &RCI;
    this->LV = &LV;
    clear();
  }

  // Incorporate a definition, normalizing overlapping physical aliases.
  // Fail without changing the live set if the capacity bound cannot guarantee
  // placement. A register already present succeeds without changing the set.
  [[nodiscard]] bool insert(Register R);
  void erase(Register R);
  void clear() { Regs.clear(); }

  auto begin() const { return Regs.begin(); }
  auto end() const { return Regs.end(); }

  // Return the backing reservation's IMPLICIT_DEF, or zero if there is none.
  Register getReservationRoot(Register R) const;

  // Whether a new backing assignment conflicts with an earlier live register.
  // PHI inputs and results inherit backing and never initiate this query.
  // Only reservation roots predict conflicts at future boundaries.
  bool conflict(Register R, Register LiveReg) const;

  // Capacity sufficient for R, measured in locations of its backing size.
  // With B conflicting live Imag8s and P Imag16s, the bound is 1 + B + 2P for
  // Imag8 or 1 + B + P for Imag16: each Imag8 could block a different pair.
  // Physical redefinitions discount the aliases they replace.
  unsigned getRequiredCapacity(Register R) const;
  unsigned getCapacity(Register R) const {
    return RCI
        ->getOrder(TRI->getRegSizeInBits(R, *MRI) == 16 ? &MOS::Imag16RegClass
                                                        : &MOS::Imag8RegClass)
        .size();
  }

private:
  bool conflictsWithReservation(Register Root, Register R) const;
  bool overlapsExit(Register R, const MachineInstr &Copy) const;

  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  SparseBitVector<> Regs;
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
  LiveRegs.init(F, *RCI, *LV);
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
    const MachineOperand *Copy = nullptr;
    if (Def && Def->isFullCopy())
      Copy = &Def->getOperand(1);
    else if (Def && Def->getOpcode() == MOS::PCOPY)
      for (unsigned I = 0, E = Def->getNumExplicitDefs(); I != E; ++I)
        if (Def->getOperand(I).getReg() == R) {
          Copy = &Def->getOperand(E + I);
          break;
        }
    if (Copy && !Copy->isUndef() && !Copy->getSubReg()) {
      Register CopySource = Copy->getReg();
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
  Register Root = LiveRegs.getReservationRoot(R);
  if (Root && Root != R) {
    assert(VRM->hasPhys(Root) && "reservation must be assigned first");
    VRM->assignVirt2Phys(R, VRM->getPhys(Root));
    bool Inserted = LiveRegs.insert(R);
    assert(Inserted && "reservation guaranteed insertion during assignment");
    (void)Inserted;
    return;
  }
  auto Order = RCI->getOrder(
      TRI->getRegSizeInBits(*MF->getRegInfo().getRegClass(R)) == 16
          ? &MOS::Imag16RegClass
          : &MOS::Imag8RegClass);
  auto Available = llvm::find_if(Order, [&](MCPhysReg Candidate) {
    return llvm::none_of(LiveRegs, [&](Register LiveReg) {
      Register Root = LiveRegs.getReservationRoot(LiveReg);
      MCPhysReg BackingReg = VRM->getPhys(Root ? Root : LiveReg);
      return BackingReg && TRI->regsOverlap(Candidate, BackingReg) &&
             LiveRegs.conflict(R, LiveReg);
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
  // A reservation's IMPLICIT_DEF is rematerializable, but reserves backing for
  // its PHI. Its incoming copies must also inherit that backing even when their
  // sources are rematerializable.
  if (LiveRegs.getReservationRoot(R))
    return true;
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

Register LiveRegisters::getReservationRoot(Register R) const {
  if (!R.isVirtual())
    return Register();
  const MachineOperand *Def = &*MRI->def_begin(R);
  const MachineInstr *MI = Def->getParent();
  if (MI->isImplicitDef()) {
    auto Uses = MRI->use_nodbg_operands(R);
    if (Uses.empty())
      return Register();
    // Every use of a reservation is an implicit exit PCOPY operand. An
    // ordinary undefined value may instead be an explicit copy source.
    const MachineOperand &Use = *Uses.begin();
    return Use.isImplicit() && Use.getParent()->getOpcode() == MOS::PCOPY
               ? R
               : Register();
  }
  if (MI->isPHI()) {
    // All incoming copies carry the same reservation; any input identifies it.
    Def = &*MRI->def_begin(MI->getOperand(1).getReg());
    MI = Def->getParent();
    assert(MI->getOpcode() == MOS::PCOPY && "expected an isolated PHI input");
  }
  if (MI->getOpcode() != MOS::PCOPY)
    return Register();
  unsigned NumDefs = MI->getNumExplicitDefs();
  if (MI->getNumOperands() == 2 * NumDefs)
    return Register(); // Entry PCOPY destinations have independent backing.
  assert(MI->getNumOperands() == 3 * NumDefs &&
         "expected exit PCOPY reservation operands");
  Register Root = MI->getOperand(2 * NumDefs + Def->getOperandNo()).getReg();
  assert(MRI->getVRegDef(Root)->isImplicitDef() &&
         "expected a reservation root");
  return Root;
}

bool LiveRegisters::insert(Register R) {
  if (Regs.test(R))
    return true;
  // PHI inputs and results inherit the assignment guaranteed at their root's
  // definition; they introduce no new backing assignment to check.
  Register Root = getReservationRoot(R);
  if ((!Root || R == Root) && getRequiredCapacity(R) > getCapacity(R))
    return false;
  if (R.isPhysical())
    erase(R);
  Regs.set(R);
  return true;
}

bool LiveRegisters::conflict(Register R, Register LiveReg) const {
  if (R == LiveReg)
    return false;
  // Physical demand is repaired later; conservatively allow it to displace any
  // backing register, including reservations.
  if (R.isPhysical() || LiveReg.isPhysical())
    return true;
  if (R == getReservationRoot(R))
    return conflictsWithReservation(R, LiveReg);
  if (LiveReg == getReservationRoot(LiveReg))
    return conflictsWithReservation(LiveReg, R);
  // All other registers are queried only while simultaneously live.
  return true;
}

bool LiveRegisters::conflictsWithReservation(Register Root, Register R) const {
  bool IsReservation = R == getReservationRoot(R);
  for (const MachineInstr &Copy : MRI->use_nodbg_instructions(Root)) {
    assert(Copy.getOpcode() == MOS::PCOPY && "unexpected reservation use");
    // Roots conflict at shared exit PCOPYs. Other registers conflict if they
    // need their backing anywhere from the exit PCOPY through the terminators.
    if (IsReservation ? Copy.readsRegister(R, /*TRI=*/nullptr)
                      : overlapsExit(R, Copy))
      return true;
  }
  return false;
}

bool LiveRegisters::overlapsExit(Register R, const MachineInstr &Copy) const {
  const MachineBasicBlock &MBB = *Copy.getParent();
  // CSSA edge operands inherit backing rather than initiating allocation.
  // Other values surviving the exit region are live in to a successor.
  if (llvm::any_of(MBB.successors(), [&](const MachineBasicBlock *Succ) {
        return LV->isLiveIn(R, *Succ);
      }))
    return true;

  // The exit PCOPY precedes the terminators. A value killed there, or even a
  // dead definition there, still needs storage alongside its destinations.
  // Sources killed by the PCOPY itself can reuse their locations.
  for (const MachineInstr &MI :
       make_range(std::next(Copy.getIterator()), MBB.instr_end())) {
    if (MI.isDebugInstr())
      continue;
    for (const MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.getReg() == R && (MO.isDef() || MO.readsReg()))
        return true;
  }
  return false;
}

unsigned LiveRegisters::getRequiredCapacity(Register R) const {
  bool IsImag16 = TRI->getRegSizeInBits(R, *MRI) == 16;
  unsigned Required = 1;
  for (Register LiveReg : Regs) {
    if (!conflict(R, LiveReg))
      continue;
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

void LiveRegisters::erase(Register R) {
  if (Regs.test(R)) {
    Regs.reset(R);
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
