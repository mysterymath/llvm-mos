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
/// imaginary demand permits treescan assignment. Demand accounts for physical
/// register constraints and conservatively bounds interference from earlier
/// virtual assignments. MOSImagRegAssign repairs physical-register constraints
/// locally. Both passes use the same value and reservation interference rules.
/// The boundary between them is SSA MIR; this pass does not assign locations in
/// VirtRegMap.
///
/// Currently this pass validates demand and diagnoses cases requiring spills.
/// It does not insert spills. The local bound guarantees a simultaneous
/// imaginary assignment, not scratch registers for expanding parallel copies
/// into hardware instructions.
///
//===----------------------------------------------------------------------===//

#include "MOSSpill.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSImagRegAllocUtils.h"
#include "MOSInstructionInterferenceGraph.h"
#include "MOSLiveRegisters.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"
#include "MOSValueNumbering.h"
#include "llvm/ADT/DenseMap.h"
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

  [[noreturn]] void reportSpillRequired(const MachineInstr &MI,
                                        Register R) const;

  bool canFit(Register Reg, const SparseBitVector<> &LiveGlobals) const;
  bool conflictsAtRestorePoints(Register Reg, MCPhysReg Phys) const;

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

  MOSLiveRegisters LiveRegs;
  DenseMap<Register, BitVector> RestoreExclusions;
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
  LiveRegs.init(F, *ValueNumbers);
  RestoreExclusions =
      mos::computeRestoreExclusions(F, *LV, *MDT, *ValueNumbers);
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
  LiveRegs.clear();
  // Completed ancestors' live-outs for the dominance walk.
  SmallVector<std::pair<const MachineDomTreeNode *, SparseBitVector<>>>
      DomLiveRegs;
  for (const MachineDomTreeNode *Node : depth_first(MDT->getRootNode())) {
    MachineBasicBlock &MBB = *Node->getBlock();
    while (!DomLiveRegs.empty() && DomLiveRegs.back().first != Node->getIDom())
      DomLiveRegs.pop_back();
    if (!DomLiveRegs.empty()) {
      LiveRegs.inherit(DomLiveRegs.back().second);
      for (auto I = LiveRegs.liveVirtRegs().begin(),
                E = LiveRegs.liveVirtRegs().end();
           I != E;) {
        Register Reg = *I++;
        if (!LV->isLiveIn(Reg, MBB))
          LiveRegs.erase(Reg);
      }
    }
    LiveRegs.beginBlock(MBB);
    for (MachineInstr &MI : MBB)
      checkMIPressure(MI);
    assert(llvm::none_of(LiveRegs.livePhysRegs(),
                         [&](MCPhysReg Phys) {
                           return mos::needsImagReg(Phys, *MF, *ValueNumbers);
                         }) &&
           "physical imaginary register live out of basic block");
    DomLiveRegs.emplace_back(Node, LiveRegs.liveVirtRegs());
  }
}

void MOSSpill::checkMIPressure(MachineInstr &MI) {
  if (MI.isDebugInstr())
    return;
  // Global assignments survive local repairs. Only virtual ranges and future
  // restoration constraints restrict this treescan colorability check.
  auto Globals = LiveRegs.liveVirtRegs();
  for (bool Early : {true, false}) {
    if (!Early && !MI.isPHI())
      for (const MachineOperand &Use : MI.all_uses())
        if (Use.getReg().isVirtual() && Use.isKill())
          Globals.reset(Use.getReg());
    for (const MachineOperand &Def : MI.all_defs()) {
      Register Reg = Def.getReg();
      if (Def.isEarlyClobber() != Early || !Reg.isVirtual() ||
          MRI->use_nodbg_empty(Reg) ||
          !mos::needsImagReg(Reg, *MF, *ValueNumbers))
        continue;
      if (!canFit(Reg, Globals))
        reportSpillRequired(MI, Reg);
      Globals.set(Reg);
    }
  }

  // PHIs and CSSA parallel copies use the global assignments guaranteed by
  // their reservations. Ordinary instructions additionally need a legal local
  // assignment, including somewhere to preserve their live-through values.
  if (!MI.isPHI() && MI.getOpcode() != MOS::PCOPY) {
    MOSInstructionInterferenceGraph Graph(MI, LiveRegs, *ValueNumbers, *RCI);
    SmallVector<Register> SelectStack;
    if (Register Reg = Graph.simplify(SelectStack))
      reportSpillRequired(MI, Reg);
  }
  LiveRegs.stepForward(MI);
}

void MOSSpill::reportSpillRequired(const MachineInstr &MI, Register R) const {
  errs() << "MOSSpill: cannot prove imaginary register assignability in "
         << MF->getName() << ", bb." << MI.getParent()->getNumber() << " for "
         << TRI->getRegClassName(TRI->getImagRegClass(R, *MRI)) << '\n'
         << MI;
  report_fatal_error("MOSSpill spill insertion is not implemented",
                     /*GenCrashDiag=*/false);
}

bool MOSSpill::canFit(Register Reg,
                      const SparseBitVector<> &LiveGlobals) const {
  Register Root = mos::getImagReservationRoot(Reg, *MRI);
  if (Root && Root != Reg)
    return true;
  const TargetRegisterClass *RC = TRI->getImagRegClass(Reg, *MRI);
  auto Value = ValueNumbers->getValueNumber(Reg);
  int Available = llvm::count_if(RCI->getOrder(RC), [&](MCPhysReg Phys) {
    return !conflictsAtRestorePoints(Reg, Phys);
  });
  for (Register Other : LiveGlobals) {
    if (Other == Reg || !mos::needsImagReg(Other, *MF, *ValueNumbers))
      continue;
    if (Root == Reg) {
      if (!mos::overlapsImagReservation(Root, Other, *MRI, *LV))
        continue;
    } else if (Other == mos::getImagReservationRoot(Other, *MRI)) {
      if (!mos::overlapsImagReservation(Other, Reg, *MRI, *LV))
        continue;
    } else if (Value && Value == ValueNumbers->getValueNumber(Other)) {
      continue;
    }
    Available -=
        MaxBlockedLocations.lookup({RC, TRI->getImagRegClass(Other, *MRI)});
  }
  return Available > 0;
}

bool MOSSpill::conflictsAtRestorePoints(Register Reg, MCPhysReg Phys) const {
  auto I = RestoreExclusions.find(Reg);
  return I != RestoreExclusions.end() &&
         llvm::any_of(I->second.set_bits(), [&](unsigned OtherPhys) {
           return TRI->regsOverlap(Phys, OtherPhys);
         });
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
