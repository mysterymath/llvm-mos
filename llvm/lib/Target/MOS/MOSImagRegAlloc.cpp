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
/// assign imaginary registers, inserting spills and reloads where necessary
/// while preserving SSA form. Pressure is assessed assuming that values need
/// imaginary registers even if subsequent hardware register allocation may
/// eliminate that need.
///
/// A dominance-order treescan chooses global imaginary registers and repairs
/// imaginary constraints in the same scan, following Colombet et al.,
/// "Graph-Coloring and Treescan Register Allocation Using Repairing", section
/// 3.2. Global locations respect interference; temporary local locations
/// satisfy instruction constraints. Parallel copies restore global locations at
/// block boundaries, and fresh SSA names represent every change of location.
/// Physical definitions establish fixed-location constraints lasting through
/// their physical live ranges. A COPY to a physical imaginary register records
/// the source value there; its virtual source need not be assigned to that
/// location. Physical liveness protects the location even after the virtual
/// source dies. A later overwrite preserves any surviving virtual value in a
/// new SSA range. Each VRM
/// assignment belongs to one concrete SSA range and remains fixed. Pending
/// restorations record temporary departures from global assignments. VirtRegMap
/// carries the assignments between passes. Assignments may be outside the
/// vregs' operand classes; MOSRegAlloc handles hardware register constraints
/// and may eliminate transfers to and from imaginary registers by retaining
/// values in hardware registers. MOSValueNumbering identifies equal contents
/// through copies and REG_SEQUENCE components. This pass records new splits in
/// that shared analysis, so both allocators agree on value identities
/// independently of imaginary assignments and live ranges.
///
/// Each isolated PHI and its inputs share the imaginary register of the
/// explicit IMPLICIT_DEF reservation supplied by MOSConventionalSSA. This
/// reserves a location before the PHI's incoming values are defined. The
/// location needs to hold those values only from the predecessor exit PCOPYs
/// through the PHI to the entry PCOPY. Other values may reuse it if their
/// lifetimes do not overlap those intervals.
///
/// Before assigning an imaginary register, the pass scans live registers to
/// check that enough locations are available. Reservation IMPLICIT_DEFs predict
/// conflicts at those future boundaries; other registers use their ordinary SSA
/// lifetimes. Assignment uses the same conflict rule. Ordinary SSA registers
/// with equal value numbers do not conflict and may share imaginary registers.
///
/// This implementation handles Imag8 and Imag16 registers, with Imag8
/// registers for flags. Physical imaginary definitions contribute ordinary
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

// The common interference rule for the pressure proof and global assignment.
// Queries use SSA liveness, value identities, and CSSA reservations,
// independently of either scan's current live set or register placements.
class ImagInterference {
public:
  void init(const MachineRegisterInfo &MRI, LiveVariables &LV,
            const MOSValueNumbering &ValueNumbers) {
    this->MRI = &MRI;
    this->LV = &LV;
    this->ValueNumbers = &ValueNumbers;
  }

  // Whether a new imaginary assignment conflicts with an earlier live register.
  // PHI inputs and results inherit imaginary assignments and never initiate
  // this query. Only reservation roots predict conflicts at future boundaries.
  bool conflict(Register R, Register LiveReg) const;
  // Return the imaginary register reservation's IMPLICIT_DEF, or zero if there
  // is none.
  Register getReservationRoot(Register R) const;

private:
  bool conflictsWithReservation(Register Root, Register R) const;
  bool overlapsExit(Register R, const MachineInstr &Copy) const;

  const MachineRegisterInfo *MRI = nullptr;
  LiveVariables *LV = nullptr;
  const MOSValueNumbering *ValueNumbers = nullptr;
};

// Live imaginary demand before locations are assigned. Physical imaginary
// definitions count as anonymous demand, just like virtual definitions. An
// Imag16 occupies one entry; killing one of its bytes leaves Imag8 demand.
// The capacity bound guarantees that the later treescan can choose locations.
class ImagPressure {
public:
  void init(const MachineFunction &MF, const RegisterClassInfo &RCI,
            const ImagInterference &Interference) {
    MRI = &MF.getRegInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    this->RCI = &RCI;
    this->Interference = &Interference;
    clear();
  }

  // Incorporate a definition, normalizing overlapping physical aliases.
  // Fail without changing the live set if the capacity bound cannot guarantee
  // placement. A register already present succeeds without changing the set.
  [[nodiscard]] bool insert(Register R);
  void erase(Register R);
  void clear() { ImagRanges.clear(); }
  void releaseKilledUses(const MachineInstr &MI);
  void clobber(const MachineOperand &RegMask);
  void releaseDeadDefs(const MachineInstr &MI);

  // Capacity sufficient for R, measured in locations of its imaginary register
  // size. With B conflicting live Imag8s and P Imag16s, the bound is 1 + B + 2P
  // for Imag8 or 1 + B + P for Imag16: each Imag8 could block a different pair.
  // Physical redefinitions discount the aliases they replace.
  unsigned getRequiredCapacity(Register R) const;
  unsigned getCapacity(Register R) const {
    return RCI
        ->getOrder(TRI->getRegSizeInBits(R, *MRI) == 16 ? &MOS::Imag16RegClass
                                                        : &MOS::Imag8RegClass)
        .size();
  }
  const SparseBitVector<> &imagRanges() const { return ImagRanges; }
  void inherit(const SparseBitVector<> &LiveOuts) { ImagRanges = LiveOuts; }

private:
  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  const ImagInterference *Interference = nullptr;
  SparseBitVector<> ImagRanges;
};

// Live assignments and the contents they must preserve. Virtual ranges have
// concrete imaginary locations in VRM; physical ranges retain their explicit
// locations. Placement queries account for both and permit equal values to
// share.
class LiveRegisters {
public:
  using ValueNumber = MOSValueNumbering::ValueNumber;

  void init(const MachineFunction &MF, const MOSValueNumbering &ValueNumbers,
            const VirtRegMap &VRM) {
    MRI = &MF.getRegInfo();
    TRI = MF.getSubtarget().getRegisterInfo();
    this->ValueNumbers = &ValueNumbers;
    this->VRM = &VRM;
    PhysRegs.init(*TRI);
    PhysContents.emplace(*TRI, ValueNumbers);
    clear();
  }

  // The caller has chosen and recorded this virtual range's assignment in VRM.
  void insert(Register R) {
    assert(R.isVirtual() && VRM->hasPhys(R));
    ImagRanges.set(R);
  }
  void erase(Register R);
  void clear() {
    ImagRanges.clear();
    PhysRegs.clear();
    PhysContents->clear();
  }

  // Physical lifetimes are block-local. Keep the virtual live-ins inherited
  // from the dominance walk and initialize this block's physical registers.
  void beginBlock(const MachineBasicBlock &MBB);
  void definePhysical(const MachineOperand &Def);
  // A COPY from a physical register can identify its previously unknown value
  // through the virtual result, without starting another physical lifetime.
  void recordPhysicalValue(MCPhysReg R, ValueNumber Value);
  void releaseKilledUses(const MachineInstr &MI);
  void clobber(const MachineOperand &RegMask);
  void releaseDeadDefs(const MachineInstr &MI);

  // Writing Value must preserve all surviving virtual and physical contents.
  bool canPlace(MCPhysReg Phys, ValueNumber Value) const;
  // Input is a dying virtual range; its location may hold the new value once
  // all other surviving contents have been accounted for.
  bool canReuse(Register Input, ValueNumber Value) const;
  // Check one register's contents, including overlapping byte identities.
  // Existing undef bytes impose no preservation requirement. Reservation
  // conflicts at future boundaries are handled by global assignment instead.
  bool preservesValue(Register R, MCPhysReg Phys, ValueNumber Value) const;
  // Parallel restoration moves virtual ranges together; only physical
  // lifetimes must retain their existing locations throughout that shuffle.
  bool preservesPhysicalValues(MCPhysReg Phys, ValueNumber Value) const;
  // Snapshot for repair copies inserted before the current instruction.
  LiveRegUnits physicalUnits() const;

  // Only virtual ranges cross block boundaries; physical lifetimes stay local.
  const SparseBitVector<> &imagRanges() const { return ImagRanges; }
  void inherit(const SparseBitVector<> &LiveOuts) { ImagRanges = LiveOuts; }

private:
  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MOSValueNumbering *ValueNumbers = nullptr;
  const VirtRegMap *VRM = nullptr;
  SparseBitVector<> ImagRanges;
  LivePhysRegs PhysRegs;
  std::optional<MOSRegisterContents> PhysContents;
};

class MOSImagRegAlloc : public MachineFunctionPass {
public:
  using ValueNumber = MOSValueNumbering::ValueNumber;

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
  void checkDefPressure(MachineInstr &MI, Register R);
  void checkClobbers(MachineInstr &MI);

  void assign();
  void assignMBB(MachineBasicBlock &MBB);
  void assignMI(MachineInstr &MI);

  void assignUses(MachineInstr &MI);
  void assignDefs(MachineInstr &MI, bool Early);
  void releaseInputsAndClobbers(MachineInstr &MI);

  MCPhysReg chooseGlobalRegister(const MachineInstr &MI, Register R) const;
  MCPhysReg chooseRepairRegister(const MachineInstr &MI, Register R) const;
  MCPhysReg chooseResultRegister(const MachineInstr &MI, Register R) const;
  void assignDefinition(Register R, MCPhysReg Local, MCPhysReg Global);

  void displace(MachineInstr &MI, MCPhysReg Phys, ValueNumber Value = {},
                bool BeforeUses = false);
  Register moveBeforeInstruction(MachineInstr &MI, Register R, MCPhysReg Phys,
                                 bool ReplaceUses);
  Register split(Register R, MCPhysReg Phys);
  void insertCopies(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator InsertPt,
                    ArrayRef<Copy> Copies);
  void replaceLocalUses(Register R, Register New,
                        MachineBasicBlock::iterator Begin);

  void restoreRegisters(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator InsertPt);
  void repairOutgoingUses(MachineBasicBlock &MBB);
  void updateLiveness(MachineBasicBlock &MBB);
  bool isLiveOut(Register R, const MachineBasicBlock &MBB) const;

  template <typename LiveSet>
  void
  walkDominatorTree(LiveSet &Regs,
                    void (MOSImagRegAlloc::*VisitBlock)(MachineBasicBlock &));

  bool isPhysicalInput(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isPhysicalOutput(const MachineInstr &MI, MCPhysReg Phys) const;
  // Reuse the dying input's assignment if writing the tied result preserves
  // the contents required by other virtual and physical lifetimes.
  bool canReuseTiedInput(const MachineOperand &Use) const;
  bool isRepairRegisterAvailable(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isResultRegisterAvailable(const MachineInstr &MI, MCPhysReg Phys,
                                 ValueNumber Value) const;

  // Unused values and trivially rematerializable values need no imaginary
  // register.
  bool needsReg(Register R) const;
  const TargetRegisterClass *imagRegClass(Register R) const {
    return TRI->getRegSizeInBits(R, *MRI) == 16 ? &MOS::Imag16RegClass
                                                : &MOS::Imag8RegClass;
  }
  bool hasImaginaryOption(Register R) const {
    return TRI->getCommonSubClass(MRI->getRegClass(R), imagRegClass(R));
  }
  MCPhysReg globalImagReg(Register R) const {
    auto I = Restorations.find(R);
    return I == Restorations.end() ? MCPhysReg(VRM->getPhys(R))
                                   : I->second.Phys;
  }
  // LiveVariables still describes this unsplit range until block-end repair.
  Register originalRange(Register R) const {
    auto I = Restorations.find(R);
    return I == Restorations.end() ? R : I->second.Original;
  }

  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  VirtRegMap *VRM = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  const MachineDominatorTree *MDT = nullptr;

  MOSValueNumbering *ValueNumbers = nullptr;
  ImagInterference Interference;
  ImagPressure Pressure;
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
  SmallVector<std::pair<const MachineDomTreeNode *, SparseBitVector<>>>
      DomLiveRegs;

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
  ValueNumbers = &getAnalysis<MOSValueNumberingWrapperPass>().valueNumbers();
  Interference.init(*MRI, *LV, *ValueNumbers);
  Pressure.init(F, *RCI, Interference);
  LiveRegs.init(F, *ValueNumbers, *VRM);
  Changed = recomputeLiveIns(F.front());

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
  AU.addRequired<MOSValueNumberingWrapperPass>();
  AU.addPreserved<MOSValueNumberingWrapperPass>();
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
  Pressure.clear();
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
      if (!Pressure.insert(R))
        report_fatal_error("MOSImagRegAlloc cannot accommodate entry live-ins",
                           /*GenCrashDiag=*/false);
    }
  }
  walkDominatorTree(Pressure, &MOSImagRegAlloc::checkMBBPressure);
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
  Pressure.releaseKilledUses(MI);

  checkClobbers(MI);
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      Pressure.clobber(MO);

  for (const MachineOperand &MO : MI.all_defs())
    if (!MO.isEarlyClobber())
      checkDefPressure(MI, MO.getReg());

  Pressure.releaseDeadDefs(MI);
}

void MOSImagRegAlloc::checkDefPressure(MachineInstr &MI, Register R) {
  if (!needsReg(R))
    return;
  if (!Pressure.insert(R)) {
    bool IsImag16 = TRI->getRegSizeInBits(R, MF->getRegInfo()) == 16;
    errs()
        << "MOSImagRegAlloc: cannot prove imaginary register assignability in "
        << MF->getName() << ", bb." << MI.getParent()->getNumber() << " for "
        << (IsImag16 ? "Imag16" : "Imag8") << ": requires "
        << Pressure.getRequiredCapacity(R) << ", " << Pressure.getCapacity(R)
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

void MOSImagRegAlloc::assign() {
  LiveRegs.clear();
  walkDominatorTree(LiveRegs, &MOSImagRegAlloc::assignMBB);
}

void MOSImagRegAlloc::assignMBB(MachineBasicBlock &MBB) {
  Restorations.clear();
  ChangedRegs.clear();
  LiveRegs.beginBlock(MBB);
  // Restore before the first terminator so every outgoing edge sees global
  // imaginary assignments. Repairing a terminator must not move a value needed
  // after that instruction away from its global imaginary assignment.
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

void MOSImagRegAlloc::assignMI(MachineInstr &MI) {
  IncomingPhysRegs = LiveRegs.physicalUnits();

  assignUses(MI);
  assignDefs(MI, true);
  releaseInputsAndClobbers(MI);
  assignDefs(MI, false);
  LiveRegs.releaseDeadDefs(MI);
}

void MOSImagRegAlloc::assignUses(MachineInstr &MI) {
  for (MachineOperand &Use : MI.all_uses()) {
    if (!Use.isTied() || !Use.getReg().isVirtual() || Use.isUndef())
      continue;
    const MachineOperand &Def =
        MI.getOperand(MI.findTiedOperandIdx(Use.getOperandNo()));
    Register R = Use.getReg(), D = Def.getReg();
    // Destructive ties need a separate occurrence if either another virtual
    // copy or a physical constraint still needs the old value.
    if (!D.isVirtual() || !hasImaginaryOption(R) || !hasImaginaryOption(D))
      continue;
    assert(!Use.getSubReg() && !Def.getSubReg() &&
           "MOSConventionalSSA must extract tied subregister uses");
    if (canReuseTiedInput(Use))
      continue;

    MCPhysReg Phys = chooseRepairRegister(MI, R);
    bool Killed = MI.killsRegister(R, nullptr);
    Register New = moveBeforeInstruction(MI, R, Phys, false);
    Use.setReg(New);
    Use.setIsKill(true);
    LiveRegs.insert(New);
    if (Killed) {
      if (MI.readsVirtualRegister(R))
        MI.addRegisterKilled(R, nullptr);
      else
        LiveRegs.erase(R);
    }
  }
}

void MOSImagRegAlloc::assignDefs(MachineInstr &MI, bool Early) {
  for (MachineOperand &MO : MI.all_defs()) {
    if (MO.isEarlyClobber() != Early || !MO.getReg().isPhysical())
      continue;
    // Implicit alias defs describe the same COPY write, so they preserve the
    // source value just as the explicit whole-register definition does.
    if (needsReg(MO.getReg()))
      displace(MI, MO.getReg(), ValueNumbers->getValueNumber(MO), Early);
    LiveRegs.definePhysical(MO);
  }
  // Fixed destinations are now clear. Pairs precede bytes because every
  // virtual register of a given width has the same imaginary domain.
  for (unsigned Bits : {16u, 8u}) {
    for (MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();
      if (MO.isEarlyClobber() != Early || !R.isVirtual() ||
          (imagRegClass(R) == &MOS::Imag16RegClass ? 16u : 8u) != Bits)
        continue;
      const MachineOperand *CopySource =
          MI.isCopy() && MO.getOperandNo() == 0 ? &MI.getOperand(1) : nullptr;
      bool CopiesPhysicalReg = CopySource &&
                               CopySource->getReg().isPhysical() &&
                               !CopySource->isUndef();
      // The definition in %v = COPY $phys names the source's unknown contents.
      // Record that identity before choosing %v's assignment, so it can share
      // the source location even while the physical constraint remains live.
      if (CopiesPhysicalReg)
        LiveRegs.recordPhysicalValue(CopySource->getReg(),
                                     ValueNumbers->getValueNumber(MO));
      const MachineOperand *TiedUse =
          MO.isTied() ? &MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo()))
                      : nullptr;
      bool InheritImagReg = TiedUse && TiedUse->getReg().isVirtual() &&
                            !TiedUse->isUndef() && hasImaginaryOption(R) &&
                            hasImaginaryOption(TiedUse->getReg());
      if (!needsReg(R) && !InheritImagReg)
        continue;
      ValueNumber Value = ValueNumbers->getValueNumber(MO);
      MCPhysReg Global = !needsReg(R)      ? MCPhysReg()
                         : VRM->hasPhys(R) ? MCPhysReg(VRM->getPhys(R))
                                           : chooseGlobalRegister(MI, R);
      MCPhysReg Local = Global;
      if (InheritImagReg)
        Local = VRM->getPhys(TiedUse->getReg());
      else if (CopiesPhysicalReg && needsReg(CopySource->getReg()) &&
               isResultRegisterAvailable(MI, CopySource->getReg(), Value))
        // Keep a captured value where it already exists, including when the
        // physical constraint remains live and holds this same value.
        Local = CopySource->getReg();
      else if (!isResultRegisterAvailable(MI, Local, Value))
        Local = chooseResultRegister(MI, R);
      assignDefinition(R, Local, Global);
      LiveRegs.insert(R);
    }
  }
}

void MOSImagRegAlloc::releaseInputsAndClobbers(MachineInstr &MI) {
  LiveRegs.releaseKilledUses(MI);

  // Inputs are consumed before register-mask clobbers take effect. Preserve
  // surviving virtual values before releasing the clobbered physical ranges.
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
      if (MO.clobbersPhysReg(R))
        displace(MI, R);
    LiveRegs.clobber(MO);
  }

  // Kills of tied inputs can erase aliases of early results, and explicit
  // early results survive register masks. Reestablish those new definitions.
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.getReg().isPhysical() && MO.isEarlyClobber())
      LiveRegs.definePhysical(MO);
}

MCPhysReg MOSImagRegAlloc::chooseGlobalRegister(const MachineInstr &MI,
                                                Register R) const {
  Register Root = Interference.getReservationRoot(R);
  if (Root && Root != R) {
    assert(globalImagReg(Root) && "reservation must be assigned first");
    return globalImagReg(Root);
  }
  auto Available = [&](MCPhysReg Candidate) {
    return llvm::none_of(LiveRegs.imagRanges(), [&](Register LiveReg) {
      Register Root = Interference.getReservationRoot(LiveReg);
      MCPhysReg ImagReg = globalImagReg(Root ? Root : LiveReg);
      return ImagReg && TRI->regsOverlap(Candidate, ImagReg) &&
             Interference.conflict(R, originalRange(LiveReg));
    });
  };
  auto Order = RCI->getOrder(imagRegClass(R));
  ValueNumber Value = ValueNumbers->getValueNumber(R);
  auto Chosen = llvm::find_if(Order, [&](MCPhysReg Candidate) {
    return Available(Candidate) &&
           (Root == R || isResultRegisterAvailable(MI, Candidate, Value));
  });
  if (Chosen == Order.end())
    Chosen = llvm::find_if(Order, Available);
  assert(Chosen != Order.end() &&
         "pressure check guaranteed an imaginary register");
  return *Chosen;
}

MCPhysReg MOSImagRegAlloc::chooseRepairRegister(const MachineInstr &MI,
                                                Register R) const {
  for (MCPhysReg Phys : RCI->getOrder(imagRegClass(R)))
    if (isRepairRegisterAvailable(MI, Phys))
      return Phys;
  report_fatal_error("MOS imaginary repairing requires spill insertion",
                     /*gen_crash_diag=*/false);
}

MCPhysReg MOSImagRegAlloc::chooseResultRegister(const MachineInstr &MI,
                                                Register R) const {
  ValueNumber Value = ValueNumbers->getValueNumber(R);
  for (MCPhysReg Phys : RCI->getOrder(imagRegClass(R)))
    if (isResultRegisterAvailable(MI, Phys, Value))
      return Phys;
  report_fatal_error("MOS imaginary repairing requires spill insertion",
                     /*gen_crash_diag=*/false);
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

void MOSImagRegAlloc::displace(MachineInstr &MI, MCPhysReg Phys,
                               ValueNumber Value, bool BeforeUses) {
  SmallVector<Register, 2> Occupants;
  // Preserve incoming values, not results produced by MI itself. In
  // particular, a regmask does not invalidate a new early-clobber result.
  for (Register R : LiveRegs.imagRanges())
    if (MRI->getVRegDef(R) != &MI && !LiveRegs.preservesValue(R, Phys, Value))
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

Register MOSImagRegAlloc::split(Register R, MCPhysReg Phys) {
  Register New = MRI->cloneVirtualRegister(R);
  VRM->grow();
  ValueNumbers->recordCopy(New, R);
  VRM->assignVirt2Phys(New, Phys);
  ChangedRegs.set(R);
  ChangedRegs.set(New);
  Changed = true;
  return New;
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
  if (LiveRegs.imagRanges().test(R)) {
    LiveRegs.erase(R);
    LiveRegs.insert(New);
  }
}

void MOSImagRegAlloc::restoreRegisters(MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator InsertPt) {
  SmallVector<Copy> Copies;
  for (Register R : LiveRegs.imagRanges()) {
    MCPhysReg Phys = globalImagReg(R);
    if (VRM->getPhys(R) == Phys)
      continue;
    if (!LiveRegs.preservesPhysicalValues(Phys,
                                          ValueNumbers->getValueNumber(R)))
      report_fatal_error("MOS imaginary repairing cannot restore an imaginary "
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
    if (!LiveRegs.imagRanges().test(Current))
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
      ValueNumbers->recordCopy(Reg, R);
      ChangedRegs.set(Reg);
    }
  }
}

void MOSImagRegAlloc::updateLiveness(MachineBasicBlock &MBB) {
  for (Register R : ChangedRegs)
    LV->recomputeForSingleDefVirtReg(R);
  for (Register R : ChangedRegs) {
    LiveRegs.erase(R);
    if (VRM->hasPhys(R) && isLiveOut(R, MBB))
      LiveRegs.insert(R);
    for (auto &[Node, Regs] : DomLiveRegs) {
      Regs.reset(R);
      if (VRM->hasPhys(R) && isLiveOut(R, *Node->getBlock()))
        Regs.set(R);
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

template <typename LiveSet>
void MOSImagRegAlloc::walkDominatorTree(
    LiveSet &Regs, void (MOSImagRegAlloc::*VisitBlock)(MachineBasicBlock &)) {
  DomLiveRegs.clear();
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock &MBB = *Node->getBlock();
    while (!DomLiveRegs.empty() && DomLiveRegs.back().first != Node->getIDom())
      DomLiveRegs.pop_back();
    if (!DomLiveRegs.empty()) {
      Regs.inherit(DomLiveRegs.back().second);
      // Trim the inherited virtual live ranges to this block's live-ins.
      for (auto I = Regs.imagRanges().begin(), E = Regs.imagRanges().end();
           I != E;) {
        Register R = *I;
        ++I;
        if (!LV->isLiveIn(R, MBB))
          Regs.erase(R);
      }
    }

    (this->*VisitBlock)(MBB);
    assert(llvm::all_of(Regs.imagRanges(),
                        [](Register R) { return R.isVirtual(); }) &&
           "physical register live out of basic block");
    DomLiveRegs.emplace_back(Node, Regs.imagRanges());
  }
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

bool MOSImagRegAlloc::canReuseTiedInput(const MachineOperand &Use) const {
  Register Input = Use.getReg();
  // Rematerializable inputs may have no imaginary assignment to reuse. Repair
  // gives the tied occurrence an assigned temporary; hardware allocation can
  // rematerialize the input there.
  if (!Use.isKill() || !VRM->hasPhys(Input))
    return false;
  const MachineInstr &MI = *Use.getParent();
  ValueNumber Value = ValueNumbers->getValueNumber(
      MI.getOperand(MI.findTiedOperandIdx(Use.getOperandNo())));
  MCPhysReg Phys = VRM->getPhys(Input);
  return !isPhysicalOutput(MI, Phys) && LiveRegs.canReuse(Input, Value);
}

bool MOSImagRegAlloc::isRepairRegisterAvailable(const MachineInstr &MI,
                                                MCPhysReg Phys) const {
  if (!IncomingPhysRegs.available(Phys) || isPhysicalInput(MI, Phys) ||
      isPhysicalOutput(MI, Phys))
    return false;
  if (!LiveRegs.canPlace(Phys, {}))
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
                                                ValueNumber Value) const {
  // Account for fixed outputs not yet visited, including ordinary physical
  // definitions when choosing an early-clobber result's location.
  return !isPhysicalOutput(MI, Phys) && LiveRegs.canPlace(Phys, Value);
}

bool MOSImagRegAlloc::needsReg(Register R) const {
  const MachineRegisterInfo &MRI = MF->getRegInfo();
  if (R.isPhysical())
    return R &&
           (MOS::Imag8RegClass.contains(R) ||
            MOS::Imag16RegClass.contains(R)) &&
           !MRI.isReserved(R);
  // A reservation's IMPLICIT_DEF is rematerializable, but reserves an imaginary
  // register for its PHI. Its incoming copies must also inherit that register
  // even when their sources are rematerializable.
  if (Interference.getReservationRoot(R))
    return true;
  if (MRI.use_nodbg_empty(R))
    return false;
  unsigned Bits = TRI->getRegSizeInBits(*MRI.getRegClass(R)).getFixedValue();
  if (Bits != 1 && Bits != 8 && Bits != 16)
    report_fatal_error(
        "MOSImagRegAlloc only supports Imag8 and Imag16 registers",
        /*GenCrashDiag=*/false);
  auto V = ValueNumbers->getValueNumber(R);
  if (V.isUndef()) {
    // An imaginary-only operand still needs an encodable location, even when
    // nothing needs to be stored there.
    const TargetRegisterClass *RC = MRI.getRegClass(R);
    return MOS::Imag8RegClass.hasSubClassEq(RC) ||
           MOS::Imag16RegClass.hasSubClassEq(RC);
  }
  const MachineInstr *Def = MRI.getVRegDef(ValueNumbers->source(V).Reg);
  return !Def ||
         !MF->getSubtarget().getInstrInfo()->isTriviallyReMaterializable(*Def);
}

bool ImagInterference::conflict(Register R, Register LiveReg) const {
  if (R == LiveReg)
    return false;
  // Physical demand is repaired later; conservatively allow it to displace any
  // imaginary register, including reservations.
  if (R.isPhysical() || LiveReg.isPhysical())
    return true;
  if (R == getReservationRoot(R))
    return conflictsWithReservation(R, LiveReg);
  if (LiveReg == getReservationRoot(LiveReg))
    return conflictsWithReservation(LiveReg, R);
  // Simultaneously live copies of this value can share its new assignment,
  // regardless of which locations were chosen for those copies.
  return ValueNumbers->getValueNumber(R) !=
         ValueNumbers->getValueNumber(LiveReg);
}

Register ImagInterference::getReservationRoot(Register R) const {
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
    return Register(); // Entry PCOPY destinations have independent imaginary
                       // assignments.
  assert(MI->getNumOperands() == 3 * NumDefs &&
         "expected exit PCOPY reservation operands");
  Register Root = MI->getOperand(2 * NumDefs + Def->getOperandNo()).getReg();
  assert(MRI->getVRegDef(Root)->isImplicitDef() &&
         "expected a reservation root");
  return Root;
}

bool ImagInterference::conflictsWithReservation(Register Root,
                                                Register R) const {
  bool IsReservation = R == getReservationRoot(R);
  for (const MachineInstr &Copy : MRI->use_nodbg_instructions(Root)) {
    assert(Copy.getOpcode() == MOS::PCOPY && "unexpected reservation use");
    // Roots conflict at shared exit PCOPYs. Other registers conflict if they
    // need their assigned register anywhere from the exit PCOPY through the
    // terminators.
    if (IsReservation ? Copy.readsRegister(R, /*TRI=*/nullptr)
                      : overlapsExit(R, Copy))
      return true;
  }
  return false;
}

bool ImagInterference::overlapsExit(Register R,
                                    const MachineInstr &Copy) const {
  const MachineBasicBlock &MBB = *Copy.getParent();
  // CSSA edge operands inherit assignments rather than initiating
  // allocation. Other values surviving the exit region are live in to a
  // successor.
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

bool ImagPressure::insert(Register R) {
  if (ImagRanges.test(R))
    return true;
  // PHI inputs and results inherit the assignment guaranteed at their root's
  // definition; they introduce no new imaginary assignment to check.
  Register Root = Interference->getReservationRoot(R);
  if ((!Root || R == Root) && getRequiredCapacity(R) > getCapacity(R))
    return false;
  if (R.isPhysical())
    erase(R);
  ImagRanges.set(R);
  return true;
}

void ImagPressure::erase(Register R) {
  if (ImagRanges.test(R)) {
    ImagRanges.reset(R);
    return;
  }
  if (R.isVirtual() || !R)
    return;
  if (MOS::Imag16RegClass.contains(R)) {
    erase(TRI->getSubReg(R, MOS::sublo));
    erase(TRI->getSubReg(R, MOS::subhi));
  } else if (MOS::Imag8RegClass.contains(R)) {
    for (MCPhysReg Super : TRI->superregs(R)) {
      if (!MOS::Imag16RegClass.contains(Super) || !ImagRanges.test(Super))
        continue;
      // Only this byte died or was overwritten. Preserve the other byte's
      // lifetime, now independently of its former pair.
      erase(Super);
      Register Lo = TRI->getSubReg(Super, MOS::sublo);
      Register Hi = TRI->getSubReg(Super, MOS::subhi);
      ImagRanges.set(R == Lo ? Hi : Lo);
      break;
    }
  }
}

void ImagPressure::releaseKilledUses(const MachineInstr &MI) {
  // LiveVariables accounts for PHI edge uses in its kill flags.
  if (MI.isPHI())
    return;
  for (const MachineOperand &MO : MI.all_uses())
    if (MO.isKill())
      erase(MO.getReg());
}

void ImagPressure::clobber(const MachineOperand &RegMask) {
  for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
    if (RegMask.clobbersPhysReg(R))
      erase(R);
}

void ImagPressure::releaseDeadDefs(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isDead())
      erase(MO.getReg());
}

unsigned ImagPressure::getRequiredCapacity(Register R) const {
  bool IsImag16 = TRI->getRegSizeInBits(R, *MRI) == 16;
  unsigned Required = 1;
  for (Register LiveReg : ImagRanges) {
    if (!Interference->conflict(R, LiveReg))
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
  if (R.isPhysical()) {
    PhysRegs.removeReg(R);
    PhysContents->clobber(R);
  } else {
    ImagRanges.reset(R);
  }
}

void LiveRegisters::beginBlock(const MachineBasicBlock &MBB) {
  PhysRegs.clear();
  PhysContents->clear();
  if (MBB.isEntryBlock())
    PhysRegs.addLiveInsNoPristines(MBB);
}

void LiveRegisters::definePhysical(const MachineOperand &Def) {
  MCPhysReg R = Def.getReg();
  PhysRegs.removeReg(R);
  PhysRegs.addReg(R);
  PhysContents->define(R, ValueNumbers->getValueNumber(Def));
}

void LiveRegisters::recordPhysicalValue(MCPhysReg R, ValueNumber Value) {
  if (!PhysContents->read(R))
    PhysContents->define(R, Value);
}

void LiveRegisters::releaseKilledUses(const MachineInstr &MI) {
  // LiveVariables accounts for PHI edge uses in its kill flags.
  if (MI.isPHI())
    return;
  for (const MachineOperand &MO : MI.all_uses())
    if (MO.isKill())
      erase(MO.getReg());
}

void LiveRegisters::clobber(const MachineOperand &RegMask) {
  PhysRegs.removeRegsInMask(RegMask);
  PhysContents->clobber(RegMask.getRegMask());
}

void LiveRegisters::releaseDeadDefs(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs()) {
    if (!MO.isDead())
      continue;
    if (MO.getReg().isPhysical())
      PhysRegs.removeReg(MO.getReg());
    else
      erase(MO.getReg());
  }
  // LiveVariables can mark a whole result dead and describe its surviving
  // bytes with implicit defs. Keep both their lifetimes and known contents.
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.getReg().isPhysical() && !MO.isDead())
      PhysRegs.addReg(MO.getReg());
  PhysContents->forgetIf(
      [&](MCPhysReg R, ValueNumber) { return PhysRegs.available(*MRI, R); });
}

bool LiveRegisters::canPlace(MCPhysReg Phys, ValueNumber Value) const {
  return preservesPhysicalValues(Phys, Value) &&
         llvm::all_of(ImagRanges, [&](Register R) {
           return preservesValue(R, Phys, Value);
         });
}

bool LiveRegisters::canReuse(Register Input, ValueNumber Value) const {
  MCPhysReg Phys = VRM->getPhys(Input);
  return preservesPhysicalValues(Phys, Value) &&
         llvm::all_of(ImagRanges, [&](Register R) {
           return R == Input || preservesValue(R, Phys, Value);
         });
}

bool LiveRegisters::preservesValue(Register R, MCPhysReg Phys,
                                   ValueNumber Value) const {
  MCPhysReg Assigned =
      R.isPhysical() ? MCPhysReg(R) : MCPhysReg(VRM->getPhys(R));
  for (unsigned SubReg : ValueNumbers->subRegIndices(R)) {
    MCPhysReg Part = Assigned;
    if (SubReg)
      Part = TRI->getSubReg(Assigned, SubReg);
    if (!TRI->regsOverlap(Phys, Part))
      continue;
    ValueNumber Existing = R.isPhysical()
                               ? PhysContents->read(Part)
                               : ValueNumbers->getValueNumber(R, SubReg);
    // Reservation roots and ordinary undef values have no contents to retain.
    // Future reservation conflicts are checked during global assignment.
    if (Existing.isUndef())
      continue;
    ValueNumber Written =
        ValueNumbers->getSubValue(Value, TRI->getSubRegIndex(Phys, Part));
    if (!Written.isUndef() && (!Written || Written != Existing))
      return false;
  }
  return true;
}

bool LiveRegisters::preservesPhysicalValues(MCPhysReg Phys,
                                            ValueNumber Value) const {
  return llvm::all_of(PhysRegs, [&](MCPhysReg R) {
    // LivePhysRegs includes every subregister, including the Imag8 LSB aliases.
    // The live super-register already accounts for their contents; their bit
    // projections need not have separate value numbers.
    if (llvm::any_of(TRI->superregs(R),
                     [&](MCPhysReg Super) { return PhysRegs.contains(Super); }))
      return true;
    return preservesValue(R, Phys, Value);
  });
}

LiveRegUnits LiveRegisters::physicalUnits() const {
  LiveRegUnits Units(*TRI);
  for (MCPhysReg R : PhysRegs)
    Units.addReg(R);
  return Units;
}

} // namespace

char MOSImagRegAlloc::ID = 0;
INITIALIZE_PASS_BEGIN(MOSImagRegAlloc, DEBUG_TYPE,
                      "MOS imaginary register allocation", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MOSValueNumberingWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_END(MOSImagRegAlloc, DEBUG_TYPE,
                    "MOS imaginary register allocation", false, false)
MachineFunctionPass *llvm::createMOSImagRegAllocPass() {
  return new MOSImagRegAlloc;
}
