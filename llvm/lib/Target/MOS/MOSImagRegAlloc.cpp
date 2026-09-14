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
/// A dominance-order treescan chooses global backing locations and repairs
/// imaginary constraints in the same scan, following Colombet et al.,
/// "Graph-Coloring and Treescan Register Allocation Using Repairing", section
/// 3.2. Global locations respect interference; temporary local locations
/// satisfy instruction constraints. Parallel copies restore global locations at
/// block boundaries, and fresh SSA names represent every change of location.
/// Physical definitions establish fixed-location constraints lasting through
/// their physical live ranges. A COPY to a physical imaginary register places
/// its source there, sharing that location with the physical range. Physical
/// liveness protects the location even after the virtual source dies. A later
/// overwrite preserves any surviving virtual value in a new SSA range.
/// Each VRM assignment belongs to one concrete SSA range and remains fixed.
/// Pending restorations record temporary departures from global assignments.
/// VirtRegMap carries the assignments between passes. Assignments may be
/// outside the vregs' operand classes; MOSRegAlloc handles hardware
/// register constraints and may eliminate backing operations by retaining
/// values in hardware registers. VirtRegMap's split ancestry also records
/// whole-value equality for COPYs and value-preserving splits. These roots do
/// not merge backing assignments or live ranges; MOSRegAlloc uses them to
/// identify equal register contents.
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
/// at those future boundaries; other registers use their ordinary SSA
/// lifetimes. Assignment uses the same conflict rule. Conflicts are
/// conservative: distinct SSA registers may interfere even when they contain
/// copies of the same value.
///
/// This implementation handles Imag8 and Imag16 backing registers, with Imag8
/// backing for flags. Physical imaginary definitions contribute ordinary
/// demand; call clobbers contribute simultaneous dead definitions. Their
/// required physical locations and whole-register ties are handled by local
/// repairing. Repairing omits optimized restore placement and does not move
/// repairs across terminators. This pass does not yet insert spills; failure
/// of the conservative pressure test or local assignment is diagnosed.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegAlloc.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterContents.h"
#include "MOSRegisterInfo.h"
#include "MOSValueNumbering.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveRegUnits.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAUpdater.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

#define DEBUG_TYPE "mos-imag-regalloc"

using namespace llvm;

namespace {

// A physical COPY connects its virtual operand to a fixed location. Virtual
// COPYs need no location constraint: value equality permits ordinary sharing.
static MachineOperand *tiedOperand(MachineOperand &MO) {
  MachineInstr &MI = *MO.getParent();
  if (MI.isFullCopy() && MO.getOperandNo() < 2 &&
      (MI.getOperand(0).getReg().isPhysical() ||
       MI.getOperand(1).getReg().isPhysical()))
    return &MI.getOperand(1 - MO.getOperandNo());
  if (MO.isTied())
    return &MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo()));
  return nullptr;
}

// A local range must return to Phys if it survives the block. Original is
// the SSA name before its first split in this block, or the definition itself
// when it needed a temporary assignment. Outgoing uses still name Original
// until SSA repair connects them to the restored range.
struct Restoration {
  Register Original;
  MCPhysReg Phys;
};

struct Copy {
  Register Def;
  Register Use;
};

// Active backing demand, independent of placement constraints. Both virtual
// and physical Imag16 definitions count as one Imag16. A partial kill of a
// physical Imag16 leaves its surviving byte as Imag8 demand. The caller selects
// values needing backing and advances their lifetimes.
class LiveRegisters {
public:
  void init(const MachineFunction &MF, const RegisterClassInfo &RCI,
            LiveVariables &LV, const MOSValueNumbering &ValueNumbers) {
    MRI = &MF.getRegInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    this->RCI = &RCI;
    this->LV = &LV;
    this->ValueNumbers = &ValueNumbers;
    clear();
  }

  // Incorporate a definition, normalizing overlapping physical aliases.
  // Fail without changing the live set if the capacity bound cannot guarantee
  // placement. A register already present succeeds without changing the set.
  [[nodiscard]] bool insert(Register R);
  // Track an already chosen placement without repeating the anonymous
  // pressure bound. Local repairing checks actual physical availability.
  void insertAssigned(Register R) {
    assert(R.isVirtual());
    Regs.set(R);
  }
  bool contains(Register R) const { return Regs.test(R); }
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
  const MOSValueNumbering *ValueNumbers = nullptr;
  SparseBitVector<> Regs;
};

class MOSImagRegAlloc : public MachineFunctionPass {
public:
  static char ID;
  MOSImagRegAlloc();

  bool runOnMachineFunction(MachineFunction &F) override;
  MachineFunctionProperties getRequiredProperties() const override;
  MachineFunctionProperties getClearedProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  // Check unconstrained demand. Failure is fatal until spill insertion exists.
  void checkPressure();
  void checkMBBPressure(MachineBasicBlock &MBB);
  void checkMIPressure(MachineInstr &MI);
  void assign();
  void assignMBB(MachineBasicBlock &MBB);
  void assignMI(MachineInstr &MI);
  void recordCopySourceValue(const MachineInstr &MI);
  void recordPhysicalDefinition(const MachineOperand &MO);
  bool isPhysicalRegisterAvailable(MCPhysReg Phys, Register Value) const;
  bool swapInto(MachineInstr &MI, Register R, MCPhysReg Phys);
  void prepareTiedUses(MachineInstr &MI);
  void assignDefs(MachineInstr &MI, bool Early);
  void displace(MachineInstr &MI, MCPhysReg Phys, Register Value = Register(),
                bool BeforeUses = false);
  // Storage is available if its occupants hold Value or are being moved/killed
  // (Vacated). With no Value, require empty storage rather than equal contents.
  bool backingAvailable(MCPhysReg Phys, Register Value = Register(),
                        Register Vacated = Register()) const;
  bool isPhysicalInput(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isPhysicalOutput(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isRepairRegisterAvailable(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isResultRegisterAvailable(const MachineInstr &MI, MCPhysReg Phys,
                                 Register R) const;
  MCPhysReg chooseRepairRegister(const MachineInstr &MI, Register R) const;
  MCPhysReg chooseResultRegister(const MachineInstr &MI, Register R) const;
  void releaseInputsAndClobbers(MachineInstr &MI);
  void releaseDeadResults(const MachineInstr &MI);
  Register moveBeforeInstruction(MachineInstr &MI, Register R, MCPhysReg Phys,
                                 bool ReplaceUses);

  void
  walkDominatorTree(void (MOSImagRegAlloc::*VisitBlock)(MachineBasicBlock &));

  void checkDefPressure(MachineInstr &MI, Register R);
  MCPhysReg chooseGlobalRegister(const MachineInstr &MI, Register R) const;

  void restoreRegisters(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator InsertPt);
  void repairOutgoingUses(MachineBasicBlock &MBB);
  void updateLiveness(MachineBasicBlock &MBB);
  void updateLiveOuts(LiveRegisters &Regs, const MachineBasicBlock &MBB);
  bool isLiveOut(Register R, const MachineBasicBlock &MBB) const;

  const TargetRegisterClass *backingClass(Register R) const {
    return TRI->getRegSizeInBits(R, *MRI) == 16 ? &MOS::Imag16RegClass
                                                : &MOS::Imag8RegClass;
  }
  bool hasImaginaryOption(Register R) const {
    return TRI->getCommonSubClass(MRI->getRegClass(R), backingClass(R));
  }
  bool isReservation(Register R) const {
    const MachineInstr *Def = MRI->getVRegDef(R);
    return Def && Def->isImplicitDef();
  }
  MCPhysReg globalBacking(Register R) const {
    auto I = Restorations.find(R);
    return I == Restorations.end() ? MCPhysReg(VRM->getPhys(R))
                                   : I->second.Phys;
  }
  // LiveVariables still describes this unsplit range until block-end repair.
  Register originalRange(Register R) const {
    auto I = Restorations.find(R);
    return I == Restorations.end() ? R : I->second.Original;
  }
  void assignDefinition(Register R, MCPhysReg Local, MCPhysReg Global);
  Register split(Register R, MCPhysReg Phys);
  void replaceLocalUses(Register R, Register New,
                        MachineBasicBlock::iterator Begin);
  void insertCopies(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator InsertPt,
                    ArrayRef<Copy> Copies);

  // Unused values and trivially rematerializable values need no backing.
  bool needsReg(Register R) const;
  // LiveVariables accounts for PHI edge uses in its kill flags.
  void removeKilledUses(const MachineInstr &MI);
  void removeDeadDefs(const MachineInstr &MI);
  void checkClobbers(MachineInstr &MI);
  void removeClobbers(const MachineInstr &MI);

  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  VirtRegMap *VRM = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  const MachineDominatorTree *MDT = nullptr;

  std::optional<MOSValueNumbering> ValueNumbers;
  LiveRegisters LiveRegs;

  // Maps local SSA names to their pending restorations. Records follow local
  // splits and remain until block-end SSA repair, even if the physical
  // placement has already been restored. Kill flags on rewritten uses retain
  // the original range's local lifetime, including outstanding outgoing uses;
  // LiveVariables is recomputed for the affected ranges at the end of the
  // block.
  MapVector<Register, Restoration> Restorations;
  SparseBitVector<> ChangedRegs;
  bool Changed = false;

  // Completed ancestors' live-outs for the dominance walk. SSA repair can add
  // PHIs in these blocks; update their snapshots along with LiveVariables.
  SmallVector<std::pair<const MachineDomTreeNode *, LiveRegisters>> DomLiveRegs;
  // Fixed physical lifetimes protect their locations independently of virtual
  // liveness. A virtual range may share its backing with a physical COPY of
  // the same value, but cannot overwrite it while this constraint remains.
  LivePhysRegs PhysRegs;
  std::optional<MOSRegisterContents> PhysContents;

  // Physical liveness at instruction entry, captured by assignMI.
  // Repair copies are emitted before the instruction, even when we discover
  // the need for one while processing its clobbers or definitions. They must
  // preserve these incoming values.
  LiveRegUnits IncomingPhysRegs;
};

MOSImagRegAlloc::MOSImagRegAlloc() : MachineFunctionPass(ID) {
  initializeMOSImagRegAllocPass(*PassRegistry::getPassRegistry());
}

bool MOSImagRegAlloc::runOnMachineFunction(MachineFunction &F) {
  MF = &F;
  MRI = &F.getRegInfo();
  TII = F.getSubtarget().getInstrInfo();
  TRI = F.getSubtarget().getRegisterInfo();
  F.getRegInfo().freezeReservedRegs();
  RCI = &getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  ValueNumbers.emplace(*VRM);
  LiveRegs.init(F, *RCI, *LV, *ValueNumbers);
  PhysRegs.init(*TRI);
  PhysContents.emplace(*TRI, *ValueNumbers);
  Changed = recomputeLiveIns(F.front());

  ValueNumbers->recordCopyOrigins();
  checkPressure();
  assign();
  return Changed;
}

MachineFunctionProperties MOSImagRegAlloc::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

MachineFunctionProperties MOSImagRegAlloc::getClearedProperties() const {
  return MachineFunctionProperties().setNoPHIs();
}

void MOSImagRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addPreserved<LiveVariablesWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineRegisterClassInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
  AU.addPreservedID(UnreachableMachineBlockElimID);
  AU.setPreservesCFG();
}

void MOSImagRegAlloc::checkPressure() {
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

void MOSImagRegAlloc::assign() {
  LiveRegs.clear();
  walkDominatorTree(&MOSImagRegAlloc::assignMBB);
}

void MOSImagRegAlloc::assignMBB(MachineBasicBlock &MBB) {
  Restorations.clear();
  ChangedRegs.clear();
  PhysRegs.clear();
  PhysContents->clear();
  if (MBB.isEntryBlock())
    PhysRegs.addLiveInsNoPristines(MBB);
  // Restore before the first terminator so every outgoing edge sees global
  // backing assignments. Repairing a terminator must not move a value needed
  // after that instruction away from its global backing assignment.
  auto FirstTerminator = MBB.getFirstTerminator();
  for (MachineInstr &MI : make_early_inc_range(MBB)) {
    if (MI.isDebugInstr())
      continue;
    // The existing PCOPYs come from CSSA. Restore before them so reservation
    // transfers start from global placements.
    // Copies inserted before MI by repairing are not revisited by this scan.
    if (MI.getIterator() == FirstTerminator || MI.getOpcode() == MOS::PCOPY)
      restoreRegisters(MBB, MI.getIterator());
    assignMI(MI);
  }
  // A block without terminators falls through; restore at its end instead.
  if (MBB.terminators().empty())
    restoreRegisters(MBB, MBB.end());
  // Restoration creates fresh SSA names. Connect outgoing uses to them before
  // recomputing liveness for the split ranges.
  repairOutgoingUses(MBB);
  updateLiveness(MBB);
}

void MOSImagRegAlloc::walkDominatorTree(
    void (MOSImagRegAlloc::*VisitBlock)(MachineBasicBlock &)) {
  DomLiveRegs.clear();
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

MCPhysReg MOSImagRegAlloc::chooseGlobalRegister(const MachineInstr &MI,
                                                Register R) const {
  Register Root = LiveRegs.getReservationRoot(R);
  if (Root && Root != R) {
    assert(globalBacking(Root) && "reservation must be assigned first");
    return globalBacking(Root);
  }
  auto Available = [&](MCPhysReg Candidate) {
    return llvm::none_of(LiveRegs, [&](Register LiveReg) {
      Register Root = LiveRegs.getReservationRoot(LiveReg);
      MCPhysReg Backing = globalBacking(Root ? Root : LiveReg);
      return Backing && TRI->regsOverlap(Candidate, Backing) &&
             LiveRegs.conflict(R, originalRange(LiveReg));
    });
  };
  auto Order = RCI->getOrder(backingClass(R));
  auto Chosen = llvm::find_if(Order, [&](MCPhysReg Candidate) {
    return Available(Candidate) &&
           (Root == R || isResultRegisterAvailable(MI, Candidate, R));
  });
  if (Chosen == Order.end())
    Chosen = llvm::find_if(Order, Available);
  assert(Chosen != Order.end() &&
         "pressure check guaranteed a backing register");
  return *Chosen;
}

void MOSImagRegAlloc::assignDefinition(Register R, MCPhysReg Local,
                                       MCPhysReg Global) {
  if (!VRM->hasPhys(R))
    VRM->assignVirt2Phys(R, Local);
  else
    assert(VRM->getPhys(R) == Local && "changed an existing assignment");
  if (Global && Local != Global)
    Restorations[R] = {R, Global};
}

Register MOSImagRegAlloc::split(Register R, MCPhysReg Phys) {
  Register New = MRI->cloneVirtualRegister(R);
  VRM->grow();
  VRM->setIsSplitFromReg(New, VRM->getOriginal(R));
  VRM->assignVirt2Phys(New, Phys);
  ChangedRegs.set(R);
  ChangedRegs.set(New);
  Changed = true;
  return New;
}

void MOSImagRegAlloc::replaceLocalUses(Register R, Register New,
                                       MachineBasicBlock::iterator Begin) {
  Restoration Restore = {R, MCPhysReg(VRM->getPhys(R))};
  auto I = Restorations.find(R);
  if (I != Restorations.end()) {
    Restore = I->second;
    Restorations.erase(I);
  }
  Restorations[New] = Restore;
  for (MachineInstr &MI :
       make_range(Begin, MRI->getVRegDef(New)->getParent()->end()))
    for (MachineOperand &MO : MI.operands())
      if (MO.isReg() && MO.isUse() && MO.getReg() == R)
        MO.setReg(New);
  if (LiveRegs.contains(R)) {
    LiveRegs.erase(R);
    LiveRegs.insertAssigned(New);
  }
}

void MOSImagRegAlloc::insertCopies(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator InsertPt,
                                   ArrayRef<Copy> Copies) {
  if (Copies.empty())
    return;
  MachineInstrBuilder MIB =
      BuildMI(MBB, InsertPt, DebugLoc(), TII->get(MOS::PCOPY));
  for (const Copy &C : Copies)
    MIB.addReg(C.Def, RegState::Define);
  for (const Copy &C : Copies)
    MIB.addReg(C.Use);
}

void MOSImagRegAlloc::assignMI(MachineInstr &MI) {
  recordCopySourceValue(MI);
  IncomingPhysRegs = LiveRegUnits(*TRI);
  for (MCPhysReg R : PhysRegs)
    IncomingPhysRegs.addReg(R);

  prepareTiedUses(MI);
  assignDefs(MI, true);
  releaseInputsAndClobbers(MI);
  assignDefs(MI, false);
  releaseDeadResults(MI);
}

void MOSImagRegAlloc::recordCopySourceValue(const MachineInstr &MI) {
  if (!MI.isFullCopy() || !MI.getOperand(0).getReg().isVirtual() ||
      !MI.getOperand(1).getReg().isPhysical() || MI.getOperand(1).isUndef())
    return;
  // Before allocating %v = COPY $phys, name unknown source contents with %v's
  // identity. The new virtual range can then share its still-live constraint.
  Register Source = MI.getOperand(1).getReg();
  if (!PhysContents->read(Source).Reg)
    PhysContents->define(Source,
                         ValueNumbers->getDefValueNumber(MI.getOperand(0)));
}

void MOSImagRegAlloc::recordPhysicalDefinition(const MachineOperand &MO) {
  MCPhysReg Phys = MO.getReg();
  PhysRegs.removeReg(Phys);
  PhysRegs.addReg(Phys);
  PhysContents->define(Phys, ValueNumbers->getDefValueNumber(MO));
}

bool MOSImagRegAlloc::isPhysicalRegisterAvailable(MCPhysReg Phys,
                                                  Register Value) const {
  if (PhysRegs.available(*MRI, Phys) ||
      PhysContents->contains(Phys, ValueNumbers->getValueNumber(Value)))
    return true;
  for (unsigned SubReg : ValueNumbers->subRegIndices(Phys)) {
    MCPhysReg Part = Phys;
    if (SubReg)
      Part = TRI->getSubReg(Phys, SubReg);
    if (!PhysRegs.available(*MRI, Part) &&
        PhysContents->read(Part) != ValueNumbers->getValueNumber(Value, SubReg))
      return false;
  }
  return true;
}

void MOSImagRegAlloc::releaseInputsAndClobbers(MachineInstr &MI) {
  removeKilledUses(MI);
  for (const MachineOperand &MO : MI.all_uses())
    if (MO.getReg().isPhysical() && MO.isKill()) {
      PhysRegs.removeReg(MO.getReg());
      PhysContents->clobber(MO.getReg());
    }

  // Inputs are consumed before register-mask clobbers take effect. Preserve
  // surviving virtual values before releasing the clobbered physical ranges.
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
      if (MO.clobbersPhysReg(R))
        displace(MI, R);
    PhysRegs.removeRegsInMask(MO);
    PhysContents->clobber(MO.getRegMask());
  }

  // Kills of tied inputs can erase aliases of early results, and explicit
  // early results survive register masks. Reestablish those new definitions.
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.getReg().isPhysical() && MO.isEarlyClobber())
      recordPhysicalDefinition(MO);
}

void MOSImagRegAlloc::releaseDeadResults(const MachineInstr &MI) {
  removeDeadDefs(MI);
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.getReg().isPhysical() && MO.isDead())
      PhysRegs.removeReg(MO.getReg());
  // LiveVariables can mark a whole result dead and describe its surviving
  // bytes with implicit defs. Keep both their lifetimes and known contents.
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.getReg().isPhysical() && !MO.isDead())
      PhysRegs.addReg(MO.getReg());
  PhysContents->forgetIf([&](MCPhysReg R, MOSValueNumbering::ValueNumber) {
    return PhysRegs.available(*MRI, R);
  });
}

bool MOSImagRegAlloc::backingAvailable(MCPhysReg Phys, Register Value,
                                       Register Vacated) const {
  return llvm::none_of(LiveRegs, [&](Register R) {
    return R != Vacated && !isReservation(R) &&
           TRI->regsOverlap(Phys, VRM->getPhys(R)) &&
           !ValueNumbers->sameValue(R, Value);
  });
}

bool MOSImagRegAlloc::isPhysicalInput(const MachineInstr &MI,
                                      MCPhysReg Phys) const {
  return llvm::any_of(MI.all_uses(), [&](const MachineOperand &MO) {
    return MO.getReg().isPhysical() && MO.readsReg() &&
           TRI->regsOverlap(Phys, MO.getReg());
  });
}

bool MOSImagRegAlloc::isPhysicalOutput(const MachineInstr &MI,
                                       MCPhysReg Phys) const {
  return llvm::any_of(MI.all_defs(), [&](const MachineOperand &MO) {
    return MO.getReg().isPhysical() && TRI->regsOverlap(Phys, MO.getReg());
  });
}

bool MOSImagRegAlloc::isRepairRegisterAvailable(const MachineInstr &MI,
                                                MCPhysReg Phys) const {
  if (!IncomingPhysRegs.available(Phys) || !backingAvailable(Phys) ||
      isPhysicalInput(MI, Phys) || isPhysicalOutput(MI, Phys))
    return false;
  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isRegMask() && MO.clobbersPhysReg(Phys))
      return false;
    // Inputs already removed from LiveRegs must still survive repair copies
    // placed before this instruction, including inputs to tied definitions.
    if (MO.isReg() && MO.isUse() && MO.readsReg() && MO.getReg().isVirtual() &&
        VRM->hasPhys(MO.getReg()) &&
        TRI->regsOverlap(Phys, VRM->getPhys(MO.getReg())))
      return false;
  }
  return true;
}

bool MOSImagRegAlloc::isResultRegisterAvailable(const MachineInstr &MI,
                                                MCPhysReg Phys,
                                                Register R) const {
  // Account for fixed outputs not yet visited, including ordinary physical
  // definitions when choosing an early-clobber result's location.
  return isPhysicalRegisterAvailable(Phys, R) && !isPhysicalOutput(MI, Phys) &&
         backingAvailable(Phys, R);
}

MCPhysReg MOSImagRegAlloc::chooseRepairRegister(const MachineInstr &MI,
                                                Register R) const {
  for (MCPhysReg Phys : RCI->getOrder(backingClass(R)))
    if (isRepairRegisterAvailable(MI, Phys))
      return Phys;
  report_fatal_error("MOS imaginary repairing requires spill insertion",
                     /*gen_crash_diag=*/false);
}

MCPhysReg MOSImagRegAlloc::chooseResultRegister(const MachineInstr &MI,
                                                Register R) const {
  for (MCPhysReg Phys : RCI->getOrder(backingClass(R)))
    if (isResultRegisterAvailable(MI, Phys, R))
      return Phys;
  report_fatal_error("MOS imaginary repairing requires spill insertion",
                     /*gen_crash_diag=*/false);
}

Register MOSImagRegAlloc::moveBeforeInstruction(MachineInstr &MI, Register R,
                                                MCPhysReg Phys,
                                                bool ReplaceUses) {
  Register New = split(R, Phys);
  Copy C{New, R};
  insertCopies(*MI.getParent(), MI.getIterator(), C);
  if (ReplaceUses)
    replaceLocalUses(R, New, MI.getIterator());
  return New;
}

void MOSImagRegAlloc::displace(MachineInstr &MI, MCPhysReg Phys, Register Value,
                               bool BeforeUses) {
  SmallVector<Register, 2> Occupants;
  // Preserve incoming values, not results produced by MI itself. In
  // particular, a regmask does not invalidate a new early-clobber result.
  for (Register R : LiveRegs)
    if (!ValueNumbers->sameValue(R, Value) && !isReservation(R) &&
        MRI->getVRegDef(R) != &MI && TRI->regsOverlap(Phys, VRM->getPhys(R)))
      Occupants.push_back(R);
  for (Register R : Occupants) {
    bool Pinned = isPhysicalInput(MI, VRM->getPhys(R));
    if (Pinned && BeforeUses)
      report_fatal_error("MOS imaginary constraints overlap a live fixed use",
                         /*gen_crash_diag=*/false);
    if (MI.isTerminator() && !MI.killsRegister(R, nullptr))
      report_fatal_error("MOS imaginary repairing cannot split a live-through "
                         "value at a terminator",
                         /*gen_crash_diag=*/false);
    Register New =
        moveBeforeInstruction(MI, R, chooseRepairRegister(MI, R), !Pinned);
    if (Pinned)
      replaceLocalUses(R, New, std::next(MI.getIterator()));
  }
}

bool MOSImagRegAlloc::swapInto(MachineInstr &MI, Register R, MCPhysReg Phys) {
  MCPhysReg Old = VRM->getPhys(R);
  if (!Old || !backingAvailable(Old, Register(), R) ||
      !PhysRegs.available(*MRI, Old) || isPhysicalInput(MI, Old) ||
      isPhysicalInput(MI, Phys))
    return false;
  SmallVector<Copy, 3> Copies;
  SmallVector<std::pair<Register, MCPhysReg>, 2> Occupants;
  for (Register Other : LiveRegs) {
    if (Other == R || isReservation(Other) ||
        !TRI->regsOverlap(Phys, VRM->getPhys(Other)))
      continue;
    MCPhysReg Destination = Old;
    if (backingClass(Other) != backingClass(R)) {
      if (backingClass(R) != &MOS::Imag16RegClass)
        return false;
      unsigned Lane = TRI->getSubReg(Phys, MOS::sublo) == VRM->getPhys(Other)
                          ? MOS::sublo
                          : MOS::subhi;
      Destination = TRI->getSubReg(Old, Lane);
    }
    Occupants.emplace_back(Other, Destination);
  }
  if (Occupants.empty())
    return false;
  for (auto [Other, Destination] : Occupants)
    Copies.push_back({split(Other, Destination), Other});
  Copies.push_back({split(R, Phys), R});
  insertCopies(*MI.getParent(), MI.getIterator(), Copies);
  for (Copy C : Copies)
    replaceLocalUses(C.Use, C.Def, MI.getIterator());
  return true;
}

void MOSImagRegAlloc::prepareTiedUses(MachineInstr &MI) {
  for (MachineOperand &Use : MI.all_uses()) {
    MachineOperand *Def = tiedOperand(Use);
    if (!Def || !Use.getReg().isVirtual() || Use.isUndef())
      continue;
    Register R = Use.getReg(), D = Def->getReg();
    bool PreservesValue = MI.isCopy();
    MCPhysReg Phys;
    if (D.isPhysical()) {
      if (!PreservesValue || !needsReg(D))
        continue;
      Phys = D;
      if (VRM->getPhys(R) == Phys)
        continue;
    } else {
      // Destructive ties need a separate occurrence if either another virtual
      // copy or a physical constraint still needs the old value.
      if (!hasImaginaryOption(R) || !hasImaginaryOption(D))
        continue;
      if (Use.getSubReg() || Def->getSubReg())
        report_fatal_error(
            "MOS imaginary repairing requires whole-register ties",
            /*gen_crash_diag=*/false);
      if (Use.isKill() && VRM->hasPhys(R) &&
          backingAvailable(VRM->getPhys(R), Register(), R) &&
          PhysRegs.available(*MRI, VRM->getPhys(R)) &&
          !isPhysicalOutput(MI, VRM->getPhys(R)))
        continue;
      Phys = chooseRepairRegister(MI, R);
    }

    // A value-preserving tie can relocate the source's remaining uses too,
    // unless a previous physical lifetime still needs it at the old location.
    bool MoveSource = PreservesValue && VRM->hasPhys(R) &&
                      PhysRegs.available(*MRI, VRM->getPhys(R));
    if (MoveSource && swapInto(MI, R, Phys))
      continue;
    displace(MI, Phys, R, true);
    bool Killed = MI.killsRegister(R, nullptr);
    Register New = moveBeforeInstruction(MI, R, Phys, MoveSource);
    if (!MoveSource) {
      Use.setReg(New);
      Use.setIsKill(true);
      LiveRegs.insertAssigned(New);
      if (Killed) {
        if (MI.readsVirtualRegister(R))
          MI.addRegisterKilled(R, nullptr);
        else
          LiveRegs.erase(R);
      }
    }
  }
}

void MOSImagRegAlloc::assignDefs(MachineInstr &MI, bool Early) {
  for (MachineOperand &MO : MI.all_defs()) {
    if (MO.isEarlyClobber() != Early || !MO.getReg().isPhysical())
      continue;
    Register Source;
    // Implicit alias defs describe the same COPY write, so they preserve the
    // source value just as the explicit whole-register definition does.
    if (MI.isFullCopy() && ValueNumbers->getDefValueNumber(MO).Reg)
      Source = MI.getOperand(1).getReg();
    if (needsReg(MO.getReg()))
      displace(MI, MO.getReg(), Source, Early);
    recordPhysicalDefinition(MO);
  }
  // Fixed destinations are now clear. Pairs precede bytes because every
  // virtual register of a given width has the same imaginary domain.
  for (unsigned Bits : {16u, 8u}) {
    for (MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();
      if (MO.isEarlyClobber() != Early || !R.isVirtual() ||
          (backingClass(R) == &MOS::Imag16RegClass ? 16u : 8u) != Bits)
        continue;
      MachineOperand *Input = tiedOperand(MO);
      bool InheritBacking = Input && Input->getReg().isVirtual() &&
                            !Input->isUndef() && hasImaginaryOption(R) &&
                            hasImaginaryOption(Input->getReg());
      if (!needsReg(R) && !InheritBacking)
        continue;
      MCPhysReg Global = !needsReg(R)      ? MCPhysReg()
                         : VRM->hasPhys(R) ? MCPhysReg(VRM->getPhys(R))
                                           : chooseGlobalRegister(MI, R);
      MCPhysReg Local = Global;
      if (InheritBacking)
        Local = VRM->getPhys(Input->getReg());
      else if (MI.isFullCopy() && Input && Input->getReg().isPhysical() &&
               needsReg(Input->getReg()) &&
               isResultRegisterAvailable(MI, Input->getReg(), R))
        // Keep a captured value where it already exists, including when the
        // physical constraint remains live and holds this same value.
        Local = Input->getReg();
      else if (!isReservation(R) && !isResultRegisterAvailable(MI, Local, R))
        Local = chooseResultRegister(MI, R);
      assignDefinition(R, Local, Global);
      LiveRegs.insertAssigned(R);
    }
  }
}

void MOSImagRegAlloc::restoreRegisters(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator InsertPt) {
  SmallVector<Copy> Copies;
  for (Register R : LiveRegs) {
    if (isReservation(R))
      continue;
    MCPhysReg Phys = globalBacking(R);
    if (VRM->getPhys(R) == Phys)
      continue;
    if (!isPhysicalRegisterAvailable(Phys, R))
      report_fatal_error("MOS imaginary repairing cannot restore a backing "
                         "register across a fixed physical lifetime",
                         /*gen_crash_diag=*/false);
    Copies.push_back({split(R, Phys), R});
  }
  insertCopies(MBB, InsertPt, Copies);
  for (const Copy &C : Copies)
    replaceLocalUses(C.Use, C.Def, InsertPt);
}

void MOSImagRegAlloc::repairOutgoingUses(MachineBasicBlock &MBB) {
  for (auto [Current, Restore] : Restorations) {
    if (!LiveRegs.contains(Current))
      continue;
    assert(VRM->getPhys(Current) == Restore.Phys &&
           "live-out range was not restored");
    Register R = Restore.Original;
    MachineBasicBlock *DefBlock = MRI->getVRegDef(R)->getParent();
    // Uses in the original definition's block already see that definition.
    // Rewrite PHI edge uses and uses in other blocks through the updater.
    // This includes incoming uses in MBB: a loop may need a new entry PHI.
    SmallVector<MachineOperand *> Uses;
    for (MachineOperand &MO : MRI->use_operands(R))
      if (MO.getParent()->isPHI() || MO.getParent()->getParent() != DefBlock)
        Uses.push_back(&MO);

    SmallVector<MachineInstr *> PHIs;
    MachineSSAUpdater Updater(*MF, &PHIs);
    Updater.Initialize(R);
    Updater.AddAvailableValue(DefBlock, R);
    Updater.AddAvailableValue(&MBB, Current);
    for (MachineOperand *MO : Uses)
      Updater.RewriteUse(*MO);
    for (MachineInstr *PHI : PHIs) {
      Register Reg = PHI->getOperand(0).getReg();
      VRM->grow();
      VRM->assignVirt2Phys(Reg, Restore.Phys);
      VRM->setIsSplitFromReg(Reg, VRM->getOriginal(R));
      ChangedRegs.set(Reg);
    }
  }
}

bool MOSImagRegAlloc::isLiveOut(Register R,
                                const MachineBasicBlock &MBB) const {
  if (llvm::any_of(MBB.successors(), [&](const MachineBasicBlock *Succ) {
        return LV->isLiveIn(R, *Succ);
      }))
    return true;
  // PHI operands are edge uses, rather than live-ins of their instruction's
  // block, and therefore need a separate check.
  return llvm::any_of(
      MRI->use_nodbg_operands(R), [&](const MachineOperand &MO) {
        const MachineInstr &MI = *MO.getParent();
        return MI.isPHI() &&
               MI.getOperand(MO.getOperandNo() + 1).getMBB() == &MBB;
      });
}

void MOSImagRegAlloc::updateLiveness(MachineBasicBlock &MBB) {
  for (Register R : ChangedRegs)
    LV->recomputeForSingleDefVirtReg(R);
  updateLiveOuts(LiveRegs, MBB);
  for (auto &[Node, Regs] : DomLiveRegs)
    updateLiveOuts(Regs, *Node->getBlock());
}

void MOSImagRegAlloc::updateLiveOuts(LiveRegisters &Regs,
                                     const MachineBasicBlock &MBB) {
  for (Register R : ChangedRegs) {
    Regs.erase(R);
    if (VRM->hasPhys(R) && isLiveOut(R, MBB))
      Regs.insertAssigned(R);
  }
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
    // SSA repair also creates ordinary PHIs whose assignments are inherited
    // from an already colored range, without a CSSA reservation.
    if (MI->getOpcode() != MOS::PCOPY)
      return Register();
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
  // Simultaneously live copies of this value can share its new assignment,
  // regardless of which locations were chosen for those copies.
  return !ValueNumbers->sameValue(R, LiveReg);
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
  // During block-local repair, these are the original range's kill points.
  // Its liveness is recomputed once outgoing uses have been repaired.
  const auto &Kills = LV->getVarInfo(R).Kills;
  const MachineInstr *Def = MRI->getVRegDef(R);
  for (const MachineInstr &MI :
       make_range(std::next(Copy.getIterator()), MBB.instr_end()))
    if (&MI == Def || llvm::is_contained(Kills, &MI))
      return true;
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
