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
/// Physical definitions establish register constraints lasting through
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
/// registers for flags. Local repairing handles physical register constraints
/// and whole-register ties. It omits optimized restore placement. Global
/// assignments avoid physical imaginary register constraints at restoration
/// points, so repairs stay inside blocks.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegAssign.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSImagRegAllocUtils.h"
#include "MOSInstructionInterferenceGraph.h"
#include "MOSLiveRegisters.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"
#include "MOSValueNumbering.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
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

  using Assignments = MapVector<Register, MCPhysReg>;
  MCPhysReg
  chooseGlobalRegister(const MachineInstr &MI, Register Reg,
                       const Assignments &Globals,
                       const MOSInstructionInterferenceGraph &Graph) const;
  bool isGlobalAssignmentAvailable(const MachineInstr &MI, Register Def,
                                   MCPhysReg Candidate,
                                   const Assignments &Globals) const;
  bool globalAssignmentsConflict(Register Def, MCPhysReg Candidate,
                                 Register Other, MCPhysReg Assigned) const;
  bool conflictsAtRestorePoints(Register Reg, MCPhysReg Phys) const;
  Assignments assignInstruction(const MachineInstr &MI,
                                const MOSInstructionInterferenceGraph &Graph,
                                const Assignments &Globals) const;
  bool assignmentConflicts(const MOSInstructionInterferenceGraph &Graph,
                           Register Reg, MCPhysReg Phys,
                           const Assignments &Assigned) const;
  void repairInstruction(MachineInstr &MI, const Assignments &Locals,
                         const Assignments &Globals);
  void assignDefinition(Register Reg, MCPhysReg Local, MCPhysReg Global);
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

  MCPhysReg globalImagReg(Register R) const {
    auto I = Restorations.find(R);
    return I != Restorations.end() ? I->second.Phys
           : VRM->hasPhys(R)       ? MCPhysReg(VRM->getPhys(R))
                                   : MCPhysReg();
  }
  // LiveVariables still describes this unsplit range until block-end repair.
  Register originalRange(Register R) const {
    auto I = Restorations.find(R);
    return I == Restorations.end() ? R : I->second.Original;
  }

  MachineFunction *MF = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  const TargetInstrInfo *TII = nullptr;
  const MOSRegisterInfo *TRI = nullptr;
  VirtRegMap *VRM = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  const MachineDominatorTree *MDT = nullptr;

  MOSValueNumbering *ValueNumbers = nullptr;
  MOSLiveRegisters LiveRegs;
  DenseMap<Register, BitVector> RestoreExclusions;

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
};

MOSImagRegAssign::MOSImagRegAssign() : MachineFunctionPass(ID) {
  initializeMOSImagRegAssignPass(*PassRegistry::getPassRegistry());
}

bool MOSImagRegAssign::runOnMachineFunction(MachineFunction &F) {
  MF = &F;
  MRI = &F.getRegInfo();
  TII = F.getSubtarget().getInstrInfo();
  TRI = F.getSubtarget<MOSSubtarget>().getRegisterInfo();
  F.getRegInfo().freezeReservedRegs();
  RCI = &getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI();
  VRM = &getAnalysis<VirtRegMapWrapperLegacy>().getVRM();
  LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  ValueNumbers = &getAnalysis<MOSValueNumberingWrapperPass>().valueNumbers();
  LiveRegs.init(F, *ValueNumbers);
  Changed = recomputeLiveIns(F.front());
  RestoreExclusions =
      mos::computeRestoreExclusions(F, *LV, *MDT, *ValueNumbers);

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
      for (auto I = LiveRegs.liveVirtRegs().begin(),
                E = LiveRegs.liveVirtRegs().end();
           I != E;) {
        Register R = *I;
        ++I;
        if (!LV->isLiveIn(R, MBB))
          LiveRegs.erase(R);
      }
    }

    assignMBB(MBB);
    assert(llvm::none_of(LiveRegs.livePhysRegs(),
                         [&](MCPhysReg Phys) {
                           return mos::needsImagReg(Phys, *MF, *ValueNumbers);
                         }) &&
           "physical imaginary register live out of basic block");
    DomLiveRegs.emplace_back(Node, LiveRegs.liveVirtRegs());
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
  MOSInstructionInterferenceGraph Graph(MI, LiveRegs, *ValueNumbers, *RCI);
  Assignments Globals;
  // This is the same definition order used by MOSSpill's global bound.
  for (bool Early : {true, false})
    for (const MachineOperand &Def : MI.all_defs()) {
      Register Reg = Def.getReg();
      if (!Reg.isVirtual() || Def.isEarlyClobber() != Early ||
          MRI->use_nodbg_empty(Reg) ||
          !mos::needsImagReg(Reg, *MF, *ValueNumbers))
        continue;
      Globals[Reg] = VRM->hasPhys(Reg)
                         ? globalImagReg(Reg)
                         : chooseGlobalRegister(MI, Reg, Globals, Graph);
    }
  // CSSA transfers already describe their parallel reads and writes. Their
  // reserved destinations use global assignments, without another local repair.
  if (MI.isPHI() || MI.getOpcode() == MOS::PCOPY || MI.isImplicitDef()) {
    for (auto [Reg, Phys] : Globals)
      assignDefinition(Reg, Phys, Phys);
    LiveRegs.stepForward(MI);
    return;
  }
  Assignments Locals = assignInstruction(MI, Graph, Globals);
  repairInstruction(MI, Locals, Globals);
}

MCPhysReg MOSImagRegAssign::chooseGlobalRegister(
    const MachineInstr &MI, Register Reg, const Assignments &Globals,
    const MOSInstructionInterferenceGraph &Graph) const {
  Register Root = mos::getImagReservationRoot(Reg, *MRI);
  if (Root && Root != Reg) {
    assert(globalImagReg(Root) && "reservation must be assigned first");
    return globalImagReg(Root);
  }
  auto Order = RCI->getOrder(TRI->getImagRegClass(Reg, *MRI));
  // Prefer a global assignment that is also usable at the definition. The
  // global colorability guarantee itself is independent of local repairing.
  if (llvm::is_contained(Graph.regs(), Reg)) {
    Assignments Existing;
    for (Register Other : Graph.regs())
      if (Other != Reg && VRM->hasPhys(Other))
        Existing[Other] = VRM->getPhys(Other);
    for (MCPhysReg Phys : Order)
      if (isGlobalAssignmentAvailable(MI, Reg, Phys, Globals) &&
          llvm::is_contained(Graph.candidatePhysRegs(Reg), Phys) &&
          !assignmentConflicts(Graph, Reg, Phys, Existing))
        return Phys;
  }
  for (MCPhysReg Phys : Order)
    if (isGlobalAssignmentAvailable(MI, Reg, Phys, Globals))
      return Phys;
  llvm_unreachable("MOSSpill guaranteed a global imaginary assignment");
}

bool MOSImagRegAssign::isGlobalAssignmentAvailable(
    const MachineInstr &MI, Register Def, MCPhysReg Candidate,
    const Assignments &Globals) const {
  if (conflictsAtRestorePoints(Def, Candidate))
    return false;
  for (Register Other : LiveRegs.liveVirtRegs()) {
    if (Other == Def || !VRM->hasPhys(Other))
      continue;
    if (!MRI->def_begin(Def)->isEarlyClobber() && !MI.isPHI() &&
        MI.killsRegister(Other, nullptr))
      continue;
    if (globalAssignmentsConflict(Def, Candidate, originalRange(Other),
                                  globalImagReg(Other)))
      return false;
  }
  for (auto [Other, Phys] : Globals)
    if (globalAssignmentsConflict(Def, Candidate, Other, Phys))
      return false;
  return true;
}

bool MOSImagRegAssign::globalAssignmentsConflict(Register Def,
                                                 MCPhysReg Candidate,
                                                 Register Other,
                                                 MCPhysReg Assigned) const {
  if (!Assigned || !TRI->regsOverlap(Candidate, Assigned))
    return false;
  Register Root = mos::getImagReservationRoot(Def, *MRI);
  if (Root == Def)
    return mos::overlapsImagReservation(Root, Other, *MRI, *LV);
  if (Other == mos::getImagReservationRoot(Other, *MRI))
    return mos::overlapsImagReservation(Other, Def, *MRI, *LV);
  return !mos::haveCompatibleContents(
      Candidate, ValueNumbers->getValueNumber(Def), Assigned,
      ValueNumbers->getValueNumber(Other), *TRI, *ValueNumbers);
}

bool MOSImagRegAssign::conflictsAtRestorePoints(Register Reg,
                                                MCPhysReg Phys) const {
  auto I = RestoreExclusions.find(Reg);
  return I != RestoreExclusions.end() &&
         llvm::any_of(I->second.set_bits(), [&](unsigned OtherPhys) {
           return TRI->regsOverlap(Phys, OtherPhys);
         });
}

MOSImagRegAssign::Assignments MOSImagRegAssign::assignInstruction(
    const MachineInstr &MI, const MOSInstructionInterferenceGraph &Graph,
    const Assignments &Globals) const {
  SmallVector<Register> SelectStack;
  if (Graph.simplify(SelectStack))
    report_fatal_error(
        "MOS instruction interference graph requires spill insertion", false);
  Assignments Assigned;
  while (!SelectStack.empty()) {
    Register Reg = SelectStack.pop_back_val();
    auto Candidates = Graph.candidatePhysRegs(Reg);
    const MachineInstr *Def = MRI->getVRegDef(Reg);
    Register Source = Reg;
    if (Def == &MI) {
      Source = Register();
      unsigned UseIdx;
      if (mos::canUseImagReg(Reg, *MRI) &&
          MI.isRegTiedToUseOperand(MRI->def_begin(Reg)->getOperandNo(),
                                   &UseIdx)) {
        const MachineOperand &Use = MI.getOperand(UseIdx);
        if (!Use.isUndef() && mos::canUseImagReg(Use.getReg(), *MRI))
          Source = Use.getReg();
      }
    }
    MCPhysReg Existing = Source && VRM->hasPhys(Source)
                             ? MCPhysReg(VRM->getPhys(Source))
                             : MCPhysReg();
    MCPhysReg Captured =
        Def->isCopy() && Def->getOperand(1).getReg().isPhysical()
            ? MCPhysReg(Def->getOperand(1).getReg())
            : MCPhysReg();
    MCPhysReg Chosen = 0;
    // Prefer an existing location, then take the first legal color. Successful
    // simplification guarantees a color regardless of these preferences.
    SmallVector<MCPhysReg, 32> Order = {Existing, Captured,
                                        Globals.lookup(Reg)};
    llvm::append_range(Order, Candidates);
    for (MCPhysReg Phys : Order) {
      if (!Phys || !llvm::is_contained(Candidates, Phys) ||
          assignmentConflicts(Graph, Reg, Phys, Assigned))
        continue;
      Chosen = Phys;
      break;
    }
    assert(Chosen && "graph simplification guaranteed a color");
    Assigned[Reg] = Chosen;
  }
  return Assigned;
}

bool MOSImagRegAssign::assignmentConflicts(
    const MOSInstructionInterferenceGraph &Graph, Register Reg, MCPhysReg Phys,
    const Assignments &Assigned) const {
  return llvm::any_of(Assigned, [&](const auto &Other) {
    return Graph.assignmentsConflict(Reg, Phys, Other.first, Other.second);
  });
}

void MOSImagRegAssign::repairInstruction(MachineInstr &MI,
                                         const Assignments &Locals,
                                         const Assignments &Globals) {
  SmallVector<Copy> Copies;
  SmallVector<Copy> MovedRegs;
  SmallVector<std::pair<unsigned, Register>> TiedUses;
  SmallVector<Register> KilledRegs;
  SparseBitVector<> Survivors;
  // PHIs are handled separately. For an ordinary SSA instruction, an incoming
  // register survives precisely when the instruction does not kill it.
  for (Register Reg : LiveRegs.liveVirtRegs()) {
    if (!MI.killsRegister(Reg, /*TRI=*/nullptr))
      Survivors.set(Reg);
    else
      KilledRegs.push_back(Reg);
  }

  // Determine the complete parallel shuffle while the instruction and incoming
  // state are unchanged. Sources continue to name their original locations,
  // including when two occupied registers exchange values.
  for (auto [Reg, Phys] : Locals) {
    if (MRI->getVRegDef(Reg) == &MI || VRM->getPhys(Reg) == Phys)
      continue;
    Register New = split(Reg, Phys);
    Copies.push_back({New, Reg});
    MovedRegs.push_back({New, Reg});
    if (Survivors.test(Reg)) {
      Survivors.reset(Reg);
      Survivors.set(New);
    }
  }
  for (auto [Reg, Phys] : Locals) {
    if (MRI->getVRegDef(Reg) != &MI)
      continue;
    const MachineOperand &Def = *MRI->def_begin(Reg);
    unsigned UseIdx;
    if (!mos::canUseImagReg(Reg, *MRI) ||
        !MI.isRegTiedToUseOperand(Def.getOperandNo(), &UseIdx))
      continue;
    const MachineOperand &Use = MI.getOperand(UseIdx);
    if (Use.isUndef() || !mos::canUseImagReg(Use.getReg(), *MRI))
      continue;
    assert(!Def.getSubReg() && !Use.getSubReg() &&
           "CSSA must extract tied subregister uses");
    Register Input = Use.getReg();
    Register Chosen;
    for (const Copy &Copy : MovedRegs)
      if (Copy.Use == Input && VRM->getPhys(Copy.Def) == Phys)
        Chosen = Copy.Def;
    if (!Chosen && VRM->hasPhys(Input) && VRM->getPhys(Input) == Phys)
      Chosen = Input;
    if (!Chosen) {
      Chosen = split(Input, Phys);
      Copies.push_back({Chosen, Input});
    }
    TiedUses.emplace_back(UseIdx, Chosen);
  }
  // All constraint queries are complete before any MIR or live-set mutation.
  for (const MachineOperand &Def : MI.all_defs()) {
    Register Reg = Def.getReg();
    if (!Reg.isVirtual())
      continue;
    MCPhysReg Global = Globals.lookup(Reg);
    MCPhysReg Local = Locals.lookup(Reg);
    if (Global || Local)
      assignDefinition(Reg, Local ? Local : Global, Global);
  }
  insertCopies(*MI.getParent(), MI.getIterator(), Copies);
  for (const Copy &Copy : MovedRegs)
    replaceLocalUses(Copy.Use, Copy.Def, MI.getIterator());
  for (auto [Operand, Input] : TiedUses) {
    MachineOperand &Use = MI.getOperand(Operand);
    Use.setReg(Input);
    Use.setIsKill(!Survivors.test(Input));
  }
  LiveRegs.stepForward(MI);
  // Splitting a dying tied input can move its last use into the preceding
  // PCOPY. Its original kill flag is consequently no longer on MI.
  for (Register Reg : KilledRegs) {
    LiveRegs.erase(Reg);
    for (const Copy &Copy : MovedRegs)
      if (Copy.Use == Reg)
        LiveRegs.erase(Copy.Def);
  }
}

void MOSImagRegAssign::assignDefinition(Register Reg, MCPhysReg Local,
                                        MCPhysReg Global) {
  if (!VRM->hasPhys(Reg))
    VRM->assignVirt2Phys(Reg, Local);
  else
    assert(VRM->getPhys(Reg) == Local && "changed an existing assignment");
  if (Global && Local != Global)
    Restorations[Reg] = {Reg, Global};
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
  if (LiveRegs.liveVirtRegs().test(R)) {
    LiveRegs.erase(R);
    LiveRegs.insert(New);
  }
}

void MOSImagRegAssign::restoreRegisters(MachineBasicBlock &MBB,
                                        MachineBasicBlock::iterator InsertPt) {
  SmallVector<Copy> Copies;
  for (Register R : LiveRegs.liveVirtRegs()) {
    MCPhysReg Phys = globalImagReg(R);
    if (!Phys || VRM->getPhys(R) == Phys)
      continue;
    for (MCPhysReg OtherPhys : MOS::Imag8RegClass)
      assert((LiveRegs.livePhysRegs().available(*MRI, OtherPhys) ||
              mos::haveCompatibleContents(
                  Phys, ValueNumbers->getValueNumber(R), OtherPhys,
                  LiveRegs.contents(OtherPhys), *TRI, *ValueNumbers)) &&
             "global assignment conflicts at its restoration point");
    Copies.push_back({split(R, Phys), R});
  }
  insertCopies(MBB, InsertPt, Copies);
  for (const Copy &C : Copies)
    replaceLocalUses(C.Use, C.Def, InsertPt);
}

void MOSImagRegAssign::repairOutgoingUses(MachineBasicBlock &MBB) {
  for (auto [Current, Restore] : Restorations) {
    if (!LiveRegs.liveVirtRegs().test(Current))
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
    if (isLiveOut(R, MBB))
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
