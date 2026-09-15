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
/// imaginary demand permits treescan assignment. Demand counts known physical
/// occupancy exactly and conservatively bounds earlier virtual assignments.
/// MOSImagRegAssign repairs fixed-location constraints locally. Both passes
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
#include "MOSSubtarget.h"
#include "MOSValueNumbering.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveRegUnits.h"
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
  using ValueNumber = MOSValueNumbering::ValueNumber;

  static char ID;
  MOSSpill();
  bool runOnMachineFunction(MachineFunction &F) override;
  MachineFunctionProperties getRequiredProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  void computeMaxBlockedLocations();
  void checkPressure();
  void checkMIPressure(MachineInstr &MI);
  void checkDefPressure(const MachineOperand &Def);
  void checkClobbers(MachineInstr &MI);
  [[noreturn]] void reportSpillRequired(const MachineInstr &MI,
                                        Register R) const;

  // Test demand before inserting R. Failure leaves the live set unchanged.
  // Reservation members inherit the capacity guaranteed at their root.
  [[nodiscard]] bool tryAddLiveReg(Register R, ValueNumber Value);
  void removeLiveReg(Register R);
  void releaseKilledUses(const MachineInstr &MI);
  void removeClobbers(const MachineOperand &RegMask);
  void releaseDeadDefs(const MachineInstr &MI);
  // Whether R can coexist with the live ranges. Physical occupancy is known;
  // earlier virtual assignments conservatively reduce the available locations.
  bool canFit(Register R, ValueNumber Value) const;

  MachineFunction *MF = nullptr;
  const MachineRegisterInfo *MRI = nullptr;
  const MOSRegisterInfo *TRI = nullptr;
  const RegisterClassInfo *RCI = nullptr;
  LiveVariables *LV = nullptr;
  const MachineDominatorTree *MDT = nullptr;
  const MOSValueNumbering *ValueNumbers = nullptr;

  // (queried class, earlier assignment's class) -> maximum number of queried
  // locations that one earlier assignment can block. Uses allocatable orders.
  // This is worst_1(N, C), the single-neighbor worst-case displacement in
  // Smith, Ramsey, and Holloway, "A Generalized Algorithm for Graph-Coloring
  // Register Allocation", sections 3.2-3.4 and 4:
  // https://www.cs.cmu.edu/afs/cs/academic/class/15745-s05/www/papers/gcra.pdf
  //
  // Normally Imag8 and Imag16 are alias-equivalent: they cover the same
  // storage, and reserved registers remove whole pairs. Their class tree has
  // one node, whose saturation cap is the queried class's size. Capping squeeze
  // there cannot make a failing virtual-definition fit test succeed: it still
  // leaves no location for the new value. No class-tree refinement is needed.
  // Independently truncated orders (-stress-regalloc) can break that
  // equivalence. The additive bound remains safe, but may then miss useful
  // saturation caps.
  SmallDenseMap<
      std::pair<const TargetRegisterClass *, const TargetRegisterClass *>,
      unsigned, 4>
      MaxBlockedLocations;

  // Live SSA ranges needing imaginary storage; locations are not yet assigned.
  SparseBitVector<> LiveVirtRegs;
  // Occupied physical imaginary storage. Partial kills remove only their units.
  LiveRegUnits LivePhysUnits;
};

MOSSpill::MOSSpill() : MachineFunctionPass(ID) {
  initializeMOSSpillPass(*PassRegistry::getPassRegistry());
}

bool MOSSpill::runOnMachineFunction(MachineFunction &F) {
  MF = &F;
  MRI = &F.getRegInfo();
  TRI = F.getSubtarget<MOSSubtarget>().getRegisterInfo();
  F.getRegInfo().freezeReservedRegs();
  RCI = &getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI();
  LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  ValueNumbers = &getAnalysis<MOSValueNumberingWrapperPass>().valueNumbers();
  bool Changed = recomputeLiveIns(F.front());
  computeMaxBlockedLocations();
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

void MOSSpill::computeMaxBlockedLocations() {
  MaxBlockedLocations.clear();
  const TargetRegisterClass *ImagClasses[] = {&MOS::Imag8RegClass,
                                              &MOS::Imag16RegClass};
  for (const TargetRegisterClass *RC : ImagClasses) {
    for (const TargetRegisterClass *OtherRC : ImagClasses) {
      unsigned &Bound = MaxBlockedLocations[{RC, OtherRC}];
      for (MCPhysReg Other : RCI->getOrder(OtherRC)) {
        unsigned Blocked =
            llvm::count_if(RCI->getOrder(RC), [&](MCPhysReg Reg) {
              return TRI->regsOverlap(Reg, Other);
            });
        Bound = std::max(Bound, Blocked);
      }
    }
  }
}

void MOSSpill::checkPressure() {
  LiveVirtRegs.clear();
  LivePhysUnits.init(*TRI);
  // Physical live-ins are confined to the entry block. Their locations already
  // exist; initializing occupancy requires no assignment or pressure check.
  for (const auto &LiveIn : MF->front().liveins())
    if (mos::needsImagReg(LiveIn.PhysReg, *MF, *ValueNumbers))
      LivePhysUnits.addRegMasked(LiveIn.PhysReg, LiveIn.LaneMask);
  // Completed ancestors' live-outs for the dominance walk.
  SmallVector<std::pair<const MachineDomTreeNode *, SparseBitVector<>>>
      DomLiveRegs;
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock &MBB = *Node->getBlock();
    while (!DomLiveRegs.empty() && DomLiveRegs.back().first != Node->getIDom())
      DomLiveRegs.pop_back();
    if (!DomLiveRegs.empty()) {
      LiveVirtRegs = DomLiveRegs.back().second;
      // Trim the inherited virtual live ranges to this block's live-ins.
      for (auto I = LiveVirtRegs.begin(), E = LiveVirtRegs.end(); I != E;) {
        Register R = *I++;
        if (!LV->isLiveIn(R, MBB))
          LiveVirtRegs.reset(R);
      }
    }
    for (MachineInstr &MI : MBB)
      checkMIPressure(MI);
    assert(LivePhysUnits.empty() &&
           "physical register live out of basic block");
    DomLiveRegs.emplace_back(Node, LiveVirtRegs);
  }
}

void MOSSpill::checkMIPressure(MachineInstr &MI) {
  if (MI.isDebugInstr())
    return;
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isEarlyClobber())
      checkDefPressure(MO);
  releaseKilledUses(MI);

  checkClobbers(MI);
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      removeClobbers(MO);

  for (const MachineOperand &MO : MI.all_defs())
    if (!MO.isEarlyClobber())
      checkDefPressure(MO);

  releaseDeadDefs(MI);
}

void MOSSpill::checkDefPressure(const MachineOperand &Def) {
  Register R = Def.getReg();
  if (!mos::needsImagReg(R, *MF, *ValueNumbers))
    return;
  if (!tryAddLiveReg(R, ValueNumbers->getValueNumber(Def)))
    reportSpillRequired(*Def.getParent(), R);
}

void MOSSpill::checkClobbers(MachineInstr &MI) {
  for (const MachineOperand &MO : MI.operands()) {
    if (!MO.isRegMask())
      continue;
    // Clobbers occupy storage without establishing a known value.
    for (MCPhysReg R : RCI->getOrder(&MOS::Imag8RegClass))
      if (MO.clobbersPhysReg(R) && !tryAddLiveReg(R, {}))
        reportSpillRequired(MI, R);
  }
}

void MOSSpill::reportSpillRequired(const MachineInstr &MI, Register R) const {
  errs() << "MOSSpill: cannot prove imaginary register assignability in "
         << MF->getName() << ", bb." << MI.getParent()->getNumber() << " for "
         << TRI->getRegClassName(TRI->getImagRegClass(R, *MRI)) << '\n'
         << MI;
  report_fatal_error("MOSSpill spill insertion is not implemented",
                     /*GenCrashDiag=*/false);
}

bool MOSSpill::tryAddLiveReg(Register R, ValueNumber Value) {
  if (R.isVirtual() && LiveVirtRegs.test(R))
    return true;
  if (!canFit(R, Value))
    return false;
  if (R.isPhysical())
    LivePhysUnits.addReg(R);
  else
    LiveVirtRegs.set(R);
  return true;
}

void MOSSpill::removeLiveReg(Register R) {
  if (R.isVirtual())
    LiveVirtRegs.reset(R);
  else if (R)
    LivePhysUnits.removeReg(R);
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
  LivePhysUnits.removeRegsNotPreserved(RegMask.getRegMask());
}

void MOSSpill::releaseDeadDefs(const MachineInstr &MI) {
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isDead())
      removeLiveReg(MO.getReg());
}

bool MOSSpill::canFit(Register R, ValueNumber Value) const {
  // Overwriting entirely occupied storage does not increase pressure.
  if (R.isPhysical() && llvm::all_of(TRI->regunits(R), [&](MCRegUnit Unit) {
        return LivePhysUnits.getBitVector().test(static_cast<unsigned>(Unit));
      }))
    return true;
  // PHI inputs and results inherit the assignment guaranteed at their root's
  // definition; they introduce no new imaginary assignment to check.
  Register Root = mos::getImagReservationRoot(R, *MRI);
  if (Root && Root != R)
    return true;

  const TargetRegisterClass *RC = TRI->getImagRegClass(R, *MRI);
  auto Order = RCI->getOrder(RC);
  int Available = Order.size();
  // Summing single-neighbor displacements gives an upper bound on squeeze:
  // the number of locations earlier assignments could deny to R. We use the
  // paper's additive approximation without its class-tree saturation bounds.
  for (Register LiveReg : LiveVirtRegs) {
    if (R == LiveReg)
      continue;
    // Reservation roots promise future storage, irrespective of their current
    // undef value. Only virtual ranges have the SSA lifetime used by this
    // query.
    if (R == Root) {
      if (!mos::overlapsImagReservation(R, LiveReg, *MRI, *LV))
        continue;
    } else if (LiveReg == mos::getImagReservationRoot(LiveReg, *MRI)) {
      if (R.isVirtual() && !mos::overlapsImagReservation(LiveReg, R, *MRI, *LV))
        continue;
    } else if (Value && (R.isVirtual() || !Value.isUndef()) &&
               Value == ValueNumbers->getValueNumber(LiveReg)) {
      // A known physical COPY result can share storage with its still-live
      // virtual source, just as equal virtual definitions can share.
      continue;
    }
    Available -=
        MaxBlockedLocations.lookup({RC, TRI->getImagRegClass(LiveReg, *MRI)});
  }
  // Count locations in the queried domain, not physical live ranges: two
  // separately defined bytes may occupy the same pair. Including a physical
  // definition in the union also accounts for overwrites without extra demand.
  Available -= llvm::count_if(Order, [&](MCPhysReg Phys) {
    return !LivePhysUnits.available(Phys) ||
           (R.isPhysical() && TRI->regsOverlap(R, Phys));
  });
  // A virtual definition needs one free byte or pair. A physical definition
  // is already included in the occupied locations above.
  return Available >= (R.isVirtual() ? 1 : 0);
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
