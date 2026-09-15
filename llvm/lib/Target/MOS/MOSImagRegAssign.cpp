//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Assign imaginary registers before MOS hardware register allocation.
///
/// MOSImagRegAssign operates on conventional SSA machine IR. Its contract is to
/// assign imaginary registers and repair local placement constraints while
/// preserving SSA form. MOSSpill has already established the conservative
/// capacity bound for global assignment. Hardware register allocation may
/// subsequently eliminate imaginary storage and transfers.
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
/// Global assignment uses the same reservation and value interference rules as
/// MOSSpill. Ordinary SSA ranges with equal values may share imaginary
/// registers. Local repair separately preserves the contents needed at each
/// instruction.
///
/// This implementation handles Imag8 and Imag16 registers, with Imag8
/// registers for flags. Local repairing handles fixed physical locations and
/// whole-register ties. It omits optimized restore placement, does not move
/// repairs across terminators, and diagnoses repairs requiring extra spills.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegAssign.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSImagRegAllocUtils.h"
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

#define DEBUG_TYPE "mos-imag-regassign"

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

  // Compare two placements independently of liveness and reservations.
  // Overlapping bytes must hold equal values; undef imposes no requirement,
  // and unknown contents cannot establish compatibility.
  bool haveCompatibleContents(MCPhysReg Reg, ValueNumber Value,
                              MCPhysReg OtherReg, ValueNumber OtherValue) const;

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

class MOSImagRegAssign : public MachineFunctionPass {
public:
  using ValueNumber = MOSValueNumbering::ValueNumber;

  static char ID;
  MOSImagRegAssign();

  bool runOnMachineFunction(MachineFunction &F) override;
  MachineFunctionProperties getRequiredProperties() const override;
  MachineFunctionProperties getClearedProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  void assign();
  void assignMBB(MachineBasicBlock &MBB);
  void assignMI(MachineInstr &MI);

  void assignUses(MachineInstr &MI);
  void assignDefs(MachineInstr &MI, bool Early);
  void releaseInputsAndClobbers(MachineInstr &MI);

  MCPhysReg chooseGlobalRegister(const MachineInstr &MI, Register R) const;
  // Check Candidate against existing global assignments, including their
  // reservation regions. Current local placements are checked separately.
  bool isGlobalAssignmentAvailable(Register Def, MCPhysReg Candidate) const;
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

  bool isPhysicalInput(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isPhysicalOutput(const MachineInstr &MI, MCPhysReg Phys) const;
  // Reuse the dying input's assignment if writing the tied result preserves
  // the contents required by other virtual and physical lifetimes.
  bool canReuseTiedInput(const MachineOperand &Use) const;
  bool isRepairRegisterAvailable(const MachineInstr &MI, MCPhysReg Phys) const;
  bool isResultRegisterAvailable(const MachineInstr &MI, MCPhysReg Phys,
                                 ValueNumber Value) const;

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

MOSImagRegAssign::MOSImagRegAssign() : MachineFunctionPass(ID) {
  initializeMOSImagRegAssignPass(*PassRegistry::getPassRegistry());
}

bool MOSImagRegAssign::runOnMachineFunction(MachineFunction &F) {
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
  LiveRegs.init(F, *ValueNumbers, *VRM);
  Changed = recomputeLiveIns(F.front());

  assign();
  return Changed;
}

MachineFunctionProperties MOSImagRegAssign::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

MachineFunctionProperties MOSImagRegAssign::getClearedProperties() const {
  return MachineFunctionProperties().setNoPHIs();
}

void MOSImagRegAssign::getAnalysisUsage(AnalysisUsage &AU) const {
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

void MOSImagRegAssign::assign() {
  LiveRegs.clear();
  DomLiveRegs.clear();
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock &MBB = *Node->getBlock();
    while (!DomLiveRegs.empty() && DomLiveRegs.back().first != Node->getIDom())
      DomLiveRegs.pop_back();
    if (!DomLiveRegs.empty()) {
      LiveRegs.inherit(DomLiveRegs.back().second);
      // Trim the inherited virtual live ranges to this block's live-ins.
      for (auto I = LiveRegs.imagRanges().begin(),
                E = LiveRegs.imagRanges().end();
           I != E;) {
        Register R = *I;
        ++I;
        if (!LV->isLiveIn(R, MBB))
          LiveRegs.erase(R);
      }
    }

    assignMBB(MBB);
    assert(llvm::all_of(LiveRegs.imagRanges(),
                        [](Register R) { return R.isVirtual(); }) &&
           "physical register live out of basic block");
    DomLiveRegs.emplace_back(Node, LiveRegs.imagRanges());
  }
}

void MOSImagRegAssign::assignMBB(MachineBasicBlock &MBB) {
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

void MOSImagRegAssign::assignMI(MachineInstr &MI) {
  IncomingPhysRegs = LiveRegs.physicalUnits();

  assignUses(MI);
  assignDefs(MI, true);
  releaseInputsAndClobbers(MI);
  assignDefs(MI, false);
  LiveRegs.releaseDeadDefs(MI);
}

void MOSImagRegAssign::assignUses(MachineInstr &MI) {
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

void MOSImagRegAssign::assignDefs(MachineInstr &MI, bool Early) {
  for (MachineOperand &MO : MI.all_defs()) {
    if (MO.isEarlyClobber() != Early || !MO.getReg().isPhysical())
      continue;
    // Implicit alias defs describe the same COPY write, so they preserve the
    // source value just as the explicit whole-register definition does.
    if (mos::needsImagReg(MO.getReg(), *MF, *ValueNumbers))
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
      bool NeedsImagReg = mos::needsImagReg(R, *MF, *ValueNumbers);
      if (!NeedsImagReg && !InheritImagReg)
        continue;
      ValueNumber Value = ValueNumbers->getValueNumber(MO);
      MCPhysReg Global = !NeedsImagReg     ? MCPhysReg()
                         : VRM->hasPhys(R) ? MCPhysReg(VRM->getPhys(R))
                                           : chooseGlobalRegister(MI, R);
      MCPhysReg Local = Global;
      if (InheritImagReg)
        Local = VRM->getPhys(TiedUse->getReg());
      else if (CopiesPhysicalReg &&
               mos::needsImagReg(CopySource->getReg(), *MF, *ValueNumbers) &&
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

void MOSImagRegAssign::releaseInputsAndClobbers(MachineInstr &MI) {
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

MCPhysReg MOSImagRegAssign::chooseGlobalRegister(const MachineInstr &MI,
                                                 Register R) const {
  Register Root = mos::getImagReservationRoot(R, *MRI);
  if (Root && Root != R) {
    assert(globalImagReg(Root) && "reservation must be assigned first");
    return globalImagReg(Root);
  }
  auto Order = RCI->getOrder(imagRegClass(R));
  ValueNumber Value = ValueNumbers->getValueNumber(R);
  auto Chosen = llvm::find_if(Order, [&](MCPhysReg Candidate) {
    return isGlobalAssignmentAvailable(R, Candidate) &&
           (Root == R || isResultRegisterAvailable(MI, Candidate, Value));
  });
  if (Chosen == Order.end())
    Chosen = llvm::find_if(Order, [&](MCPhysReg Candidate) {
      return isGlobalAssignmentAvailable(R, Candidate);
    });
  assert(Chosen != Order.end() &&
         "pressure check guaranteed an imaginary register");
  return *Chosen;
}

bool MOSImagRegAssign::isGlobalAssignmentAvailable(Register Def,
                                                   MCPhysReg Candidate) const {
  Register Root = mos::getImagReservationRoot(Def, *MRI);
  if (Root && Root != Def)
    return Candidate == globalImagReg(Root);
  ValueNumber Value = ValueNumbers->getValueNumber(Def);
  for (Register LiveReg : LiveRegs.imagRanges()) {
    if (LiveReg == Def)
      continue;
    MCPhysReg Assigned = globalImagReg(LiveReg);
    if (!Assigned || !TRI->regsOverlap(Candidate, Assigned))
      continue;
    // Local splits retain the original range's lifetime for future reservation
    // queries until block-end SSA repair updates LiveVariables.
    Register Original = originalRange(LiveReg);
    if (Root == Def) {
      if (mos::overlapsImagReservation(Root, Original, *MRI, *LV))
        return false;
    } else if (Original == mos::getImagReservationRoot(Original, *MRI)) {
      if (mos::overlapsImagReservation(Original, Def, *MRI, *LV))
        return false;
    } else if (!LiveRegs.haveCompatibleContents(
                   Candidate, Value, Assigned,
                   ValueNumbers->getValueNumber(LiveReg))) {
      return false;
    }
  }
  return true;
}

MCPhysReg MOSImagRegAssign::chooseRepairRegister(const MachineInstr &MI,
                                                 Register R) const {
  for (MCPhysReg Phys : RCI->getOrder(imagRegClass(R)))
    if (isRepairRegisterAvailable(MI, Phys))
      return Phys;
  report_fatal_error("MOS imaginary repairing requires spill insertion",
                     /*gen_crash_diag=*/false);
}

MCPhysReg MOSImagRegAssign::chooseResultRegister(const MachineInstr &MI,
                                                 Register R) const {
  ValueNumber Value = ValueNumbers->getValueNumber(R);
  for (MCPhysReg Phys : RCI->getOrder(imagRegClass(R)))
    if (isResultRegisterAvailable(MI, Phys, Value))
      return Phys;
  report_fatal_error("MOS imaginary repairing requires spill insertion",
                     /*gen_crash_diag=*/false);
}

void MOSImagRegAssign::assignDefinition(Register R, MCPhysReg Local,
                                        MCPhysReg Global) {
  if (!VRM->hasPhys(R))
    VRM->assignVirt2Phys(R, Local);
  else
    assert(VRM->getPhys(R) == Local && "changed an existing assignment");
  if (Global && Local != Global)
    Restorations[R] = {R, Global};
}

void MOSImagRegAssign::displace(MachineInstr &MI, MCPhysReg Phys,
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

Register MOSImagRegAssign::moveBeforeInstruction(MachineInstr &MI, Register R,
                                                 MCPhysReg Phys,
                                                 bool ReplaceUses) {
  Register New = split(R, Phys);
  Copy C{New, R};
  insertCopies(*MI.getParent(), MI.getIterator(), C);
  if (ReplaceUses)
    replaceLocalUses(R, New, MI.getIterator());
  return New;
}

Register MOSImagRegAssign::split(Register R, MCPhysReg Phys) {
  Register New = MRI->cloneVirtualRegister(R);
  VRM->grow();
  ValueNumbers->recordCopy(New, R);
  VRM->assignVirt2Phys(New, Phys);
  ChangedRegs.set(R);
  ChangedRegs.set(New);
  Changed = true;
  return New;
}

void MOSImagRegAssign::insertCopies(MachineBasicBlock &MBB,
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

void MOSImagRegAssign::replaceLocalUses(Register R, Register New,
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

void MOSImagRegAssign::restoreRegisters(MachineBasicBlock &MBB,
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

void MOSImagRegAssign::repairOutgoingUses(MachineBasicBlock &MBB) {
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

void MOSImagRegAssign::updateLiveness(MachineBasicBlock &MBB) {
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

bool MOSImagRegAssign::isLiveOut(Register R,
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

bool MOSImagRegAssign::isPhysicalInput(const MachineInstr &MI,
                                       MCPhysReg Phys) const {
  return llvm::any_of(MI.all_uses(), [&](const MachineOperand &MO) {
    return MO.getReg().isPhysical() && MO.readsReg() &&
           TRI->regsOverlap(Phys, MO.getReg());
  });
}

bool MOSImagRegAssign::isPhysicalOutput(const MachineInstr &MI,
                                        MCPhysReg Phys) const {
  return llvm::any_of(MI.all_defs(), [&](const MachineOperand &MO) {
    return MO.getReg().isPhysical() && TRI->regsOverlap(Phys, MO.getReg());
  });
}

bool MOSImagRegAssign::canReuseTiedInput(const MachineOperand &Use) const {
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

bool MOSImagRegAssign::isRepairRegisterAvailable(const MachineInstr &MI,
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

bool MOSImagRegAssign::isResultRegisterAvailable(const MachineInstr &MI,
                                                 MCPhysReg Phys,
                                                 ValueNumber Value) const {
  // Account for fixed outputs not yet visited, including ordinary physical
  // definitions when choosing an early-clobber result's location.
  return !isPhysicalOutput(MI, Phys) && LiveRegs.canPlace(Phys, Value);
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

bool LiveRegisters::haveCompatibleContents(MCPhysReg Reg, ValueNumber Value,
                                           MCPhysReg OtherReg,
                                           ValueNumber OtherValue) const {
  if (!TRI->regsOverlap(Reg, OtherReg))
    return true;
  for (unsigned SubReg : ValueNumbers->subRegIndices(Reg)) {
    MCPhysReg Part = Reg;
    if (SubReg)
      Part = TRI->getSubReg(Reg, SubReg);
    auto PartValue = ValueNumbers->getSubValue(Value, SubReg);
    for (unsigned OtherSubReg : ValueNumbers->subRegIndices(OtherReg)) {
      MCPhysReg OtherPart = OtherReg;
      if (OtherSubReg)
        OtherPart = TRI->getSubReg(OtherReg, OtherSubReg);
      if (!TRI->regsOverlap(Part, OtherPart))
        continue;
      auto OtherPartValue = ValueNumbers->getSubValue(OtherValue, OtherSubReg);
      if (PartValue.isUndef() || OtherPartValue.isUndef())
        continue;
      // Distinct overlapping subregister views may require a projection not
      // represented by value numbering (for example, a byte and its LSB).
      if (Part != OtherPart || !PartValue || PartValue != OtherPartValue)
        return false;
    }
  }
  return true;
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
  if (R.isVirtual())
    return haveCompatibleContents(Phys, Value, VRM->getPhys(R),
                                  ValueNumbers->getValueNumber(R));
  // A physical pair may have known byte contents without a whole-value name.
  for (unsigned SubReg : ValueNumbers->subRegIndices(R)) {
    MCPhysReg Part = R;
    if (SubReg)
      Part = TRI->getSubReg(R, SubReg);
    if (!haveCompatibleContents(Phys, Value, Part, PhysContents->read(Part)))
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

char MOSImagRegAssign::ID = 0;
INITIALIZE_PASS_BEGIN(MOSImagRegAssign, DEBUG_TYPE,
                      "MOS imaginary register assignment", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MOSValueNumberingWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_END(MOSImagRegAssign, DEBUG_TYPE,
                    "MOS imaginary register assignment", false, false)
MachineFunctionPass *llvm::createMOSImagRegAssignPass() {
  return new MOSImagRegAssign;
}
