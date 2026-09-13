//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Repair imaginary register assignments using parallel copies, following
/// Colombet et al., "Graph-Coloring and Treescan Register Allocation Using
/// Repairing", sections 2.3 and 3.2.
///
/// The treescan supplies global assignments. Within a block, repairing may
/// move values to satisfy constraints. Each move starts a new SSA live range
/// with its own VirtRegMap assignment. Restoring parallel copies reestablish
/// the global assignments before leaving the block. SSA reconstruction may
/// introduce PHIs; their inputs and results all have the same global assignment
/// and require no edge copies.
///
/// This implementation omits biased coloring and optimized restore placement.
/// It handles whole-register imaginary ties and physical imaginary constraints.
/// It does not insert spills or move repairs across terminators.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegRepair.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
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

#define DEBUG_TYPE "mos-imag-reg-repair"

using namespace llvm;

namespace {

class MOSImagRegRepair : public MachineFunctionPass {
public:
  static char ID;
  MOSImagRegRepair();

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override;
  MachineFunctionProperties getClearedProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

// Instruction-local lifetimes. Clobbers occur after inputs die and before
// ordinary results are defined; early-clobber results span all three phases.
enum Phase { Uses = 1, Clobbers = 2, Defs = 4, Through = 7 };

struct LocalAssignment {
  Register Use;
  Register Def;
  unsigned Phases;
  MCPhysReg Phys = 0;
};

// Keep operand indices rather than pointers: adding operands to a parallel
// copy can reallocate its operand storage.
struct UsePosition {
  MachineInstr *MI;
  unsigned Operand;
};

struct SplitValue {
  SmallVector<std::pair<MachineBasicBlock *, Register>> LiveOuts;
  SmallVector<UsePosition> IncomingUses;
};

struct Copy {
  Register Def;
  Register Use;
  // Nonzero when the source is an incoming value needing SSA reconstruction.
  Register Incoming;
};

class ImagRegRepair {
public:
  ImagRegRepair(MachineFunction &MF, VirtRegMap &VRM,
                const RegisterClassInfo &RCI, LiveVariables &LV,
                const MachineDominatorTree &MDT)
      : MF(MF), MRI(MF.getRegInfo()), VRM(VRM), RCI(RCI), LV(LV), MDT(MDT),
        TRI(*MF.getSubtarget().getRegisterInfo()),
        TII(*MF.getSubtarget().getInstrInfo()), PhysRegs(TRI) {
    for (unsigned I = 0; I != MRI.getNumVirtRegs(); ++I) {
      Register R = Register::index2VirtReg(I);
      if (VRM.hasPhys(R))
        GlobalRegs[R] = VRM.getPhys(R);
    }
  }

  bool run();

private:
  void repairBlock(MachineBasicBlock &MBB);
  void repairInstruction(MachineInstr &MI);
  void restoreRegisters(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator InsertPt);
  void reconstructSSA();

  const TargetRegisterClass *backingClass(Register R) const {
    return TRI.getRegSizeInBits(R, MRI) == 16 ? &MOS::Imag16RegClass
                                              : &MOS::Imag8RegClass;
  }
  bool hasImaginaryOption(Register R) const {
    return TRI.getCommonSubClass(MRI.getRegClass(R), backingClass(R));
  }
  bool isReservation(Register R) const {
    const MachineInstr *Def = MRI.getVRegDef(R);
    return Def && Def->isImplicitDef();
  }
  Register current(Register R) const {
    auto I = CurrentRegs.find(R);
    return I == CurrentRegs.end() ? R : I->second;
  }
  Register split(Register R, MCPhysReg Phys);
  Copy move(Register R, MCPhysReg Phys);
  void insertCopies(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator InsertPt,
                    ArrayRef<Copy> Copies);
  void recordUse(MachineInstr &MI, unsigned Operand, Register R);

  // Allocate the small, instruction-local problem. Fixed physical lifetimes
  // occupy their actual locations, rather than anonymous pressure slots.
  bool assign(SmallVectorImpl<LocalAssignment> &Assignments,
              ArrayRef<LiveRegUnits> FixedRegs, bool PreferCurrent) const;
  std::array<LiveRegUnits, 3> fixedRegisters(const MachineInstr &MI) const;

  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  VirtRegMap &VRM;
  const RegisterClassInfo &RCI;
  LiveVariables &LV;
  const MachineDominatorTree &MDT;
  const TargetRegisterInfo &TRI;
  const TargetInstrInfo &TII;

  // Global assignments remain unchanged as the local SSA ranges are split.
  MapVector<Register, MCPhysReg> GlobalRegs;
  MapVector<Register, SplitValue> SplitValues;
  SparseBitVector<> ChangedRegs;

  // State for one block. Keys retain the input MIR's names so that its kill
  // flags and LiveVariables remain usable throughout the repair traversal.
  SparseBitVector<> LiveRegs;
  DenseMap<Register, Register> CurrentRegs;
  LivePhysRegs PhysRegs;
};

bool ImagRegRepair::run() {
  // The original definition remains in its block when splitting. Thus the
  // original LiveVariables remains suitable for trimming this traversal.
  SmallVector<std::pair<const MachineDomTreeNode *, SparseBitVector<>>> Parents;
  for (const MachineDomTreeNode *Node : depth_first(MDT.getRootNode())) {
    while (!Parents.empty() && Parents.back().first != Node->getIDom())
      Parents.pop_back();
    LiveRegs = Parents.empty() ? SparseBitVector<>() : Parents.back().second;
    for (auto I = LiveRegs.begin(), E = LiveRegs.end(); I != E;) {
      Register R = *I++;
      if (!LV.isLiveIn(R, *Node->getBlock()))
        LiveRegs.reset(R);
    }
    repairBlock(*Node->getBlock());
    Parents.emplace_back(Node, LiveRegs);
  }
  reconstructSSA();
  return !ChangedRegs.empty();
}

Register ImagRegRepair::split(Register R, MCPhysReg Phys) {
  Register New = MRI.cloneVirtualRegister(R);
  VRM.grow();
  VRM.setIsSplitFromReg(New, VRM.getOriginal(R));
  VRM.assignVirt2Phys(New, Phys);
  ChangedRegs.set(R);
  return New;
}

Copy ImagRegRepair::move(Register R, MCPhysReg Phys) {
  return {split(R, Phys), current(R), CurrentRegs.count(R) ? Register() : R};
}

void ImagRegRepair::insertCopies(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator InsertPt,
                                 ArrayRef<Copy> Copies) {
  if (Copies.empty())
    return;
  MachineInstrBuilder MIB =
      BuildMI(MBB, InsertPt, DebugLoc(), TII.get(MOS::PCOPY));
  for (const Copy &C : Copies)
    MIB.addReg(C.Def, RegState::Define);
  for (const Copy &C : Copies)
    MIB.addReg(C.Use);
  for (unsigned I = 0; I != Copies.size(); ++I)
    if (Copies[I].Incoming)
      SplitValues[Copies[I].Incoming].IncomingUses.push_back(
          {MIB.getInstr(), unsigned(Copies.size()) + I});
}

void ImagRegRepair::recordUse(MachineInstr &MI, unsigned Operand, Register R) {
  if (!CurrentRegs.count(R))
    SplitValues[R].IncomingUses.push_back({&MI, Operand});
  MI.getOperand(Operand).setReg(current(R));
}

std::array<LiveRegUnits, 3>
ImagRegRepair::fixedRegisters(const MachineInstr &MI) const {
  std::array<LiveRegUnits, 3> Fixed = {LiveRegUnits(TRI), LiveRegUnits(TRI),
                                       LiveRegUnits(TRI)};
  for (MCPhysReg R : PhysRegs)
    for (LiveRegUnits &Regs : Fixed)
      Regs.addReg(R);
  for (const MachineOperand &MO : MI.all_uses()) {
    Register R = MO.getReg();
    if (!R.isPhysical() || !R || MO.isUndef())
      continue;
    Fixed[0].addReg(R);
    if (MO.isKill()) {
      Fixed[1].removeReg(R);
      Fixed[2].removeReg(R);
    }
  }
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask()) {
      Fixed[2].removeRegsNotPreserved(MO.getRegMask());
      for (MCPhysReg R : RCI.getOrder(&MOS::Imag8RegClass))
        if (MO.clobbersPhysReg(R))
          Fixed[1].addReg(R);
    }
  for (const MachineOperand &MO : MI.all_defs()) {
    Register R = MO.getReg();
    if (!R.isPhysical() || !R)
      continue;
    if (MO.isEarlyClobber()) {
      Fixed[0].addReg(R);
      Fixed[1].addReg(R);
    }
    Fixed[2].addReg(R);
  }
  return Fixed;
}

bool ImagRegRepair::assign(SmallVectorImpl<LocalAssignment> &Assignments,
                           ArrayRef<LiveRegUnits> FixedRegs,
                           bool PreferCurrent) const {
  for (LocalAssignment &A : Assignments)
    A.Phys = 0;
  SmallVector<unsigned> Order;
  for (unsigned I = 0; I != Assignments.size(); ++I)
    if (Assignments[I].Phases)
      Order.push_back(I);
  // Place pairs before bytes, and live-through ranges before shorter ranges.
  // Otherwise a byte can unnecessarily fragment an available pair.
  llvm::stable_sort(Order, [&](unsigned A, unsigned B) {
    const LocalAssignment &X = Assignments[A], &Y = Assignments[B];
    return std::pair(backingClass(X.Def ? X.Def : X.Use) ==
                         &MOS::Imag16RegClass,
                     llvm::popcount(X.Phases)) >
           std::pair(backingClass(Y.Def ? Y.Def : Y.Use) ==
                         &MOS::Imag16RegClass,
                     llvm::popcount(Y.Phases));
  });
  for (unsigned I : Order) {
    LocalAssignment &A = Assignments[I];
    auto Available = [&](MCPhysReg Phys) {
      for (unsigned P = 0; P != 3; ++P)
        if ((A.Phases & (1u << P)) && !FixedRegs[P].available(Phys))
          return false;
      return llvm::none_of(Assignments, [&](const LocalAssignment &B) {
        return B.Phys && (A.Phases & B.Phases) && TRI.regsOverlap(Phys, B.Phys);
      });
    };
    Register R = A.Def ? A.Def : A.Use;
    MCPhysReg Preferred =
        A.Use ? MCPhysReg(VRM.getPhys(current(A.Use))) : GlobalRegs.lookup(R);
    if (PreferCurrent && Preferred && Available(Preferred)) {
      A.Phys = Preferred;
      continue;
    }
    for (MCPhysReg Phys : RCI.getOrder(backingClass(R)))
      if (Available(Phys)) {
        A.Phys = Phys;
        break;
      }
    if (!A.Phys)
      return false;
  }
  return true;
}

void ImagRegRepair::repairInstruction(MachineInstr &MI) {
  SmallVector<LocalAssignment> Assignments;
  DenseMap<Register, unsigned> Inputs;
  SmallVector<int> Operands(MI.getNumOperands(), -1);
  for (Register R : LiveRegs) {
    Inputs[R] = Assignments.size();
    Assignments.push_back(
        {R, Register(), MI.killsRegister(R, nullptr) ? Uses : Through});
  }
  for (unsigned I = 0; I != MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.getReg().isVirtual())
      continue;
    Register R = MO.getReg();
    if (MO.isDef() && GlobalRegs.count(R) && !isReservation(R)) {
      Operands[I] = Assignments.size();
      Assignments.push_back(
          {Register(), R, MO.isEarlyClobber() ? Through : Defs});
    } else if (MO.isUse() && !MO.isUndef()) {
      auto Input = Inputs.find(R);
      if (Input != Inputs.end())
        Operands[I] = Input->second;
    }
  }
  // A tied live-through input needs a separate, killed copy for the operation.
  // A killed input can share its local assignment directly with the result.
  for (unsigned I = 0; I != MI.getNumOperands(); ++I) {
    unsigned Def;
    if (!MI.isRegTiedToDefOperand(I, &Def))
      continue;
    const MachineOperand &Use = MI.getOperand(I);
    const MachineOperand &Output = MI.getOperand(Def);
    if (!Use.getReg().isVirtual() || !Output.getReg().isVirtual() ||
        !hasImaginaryOption(Use.getReg()) ||
        !hasImaginaryOption(Output.getReg()))
      continue;
    if (Use.getSubReg() || Output.getSubReg())
      report_fatal_error("MOS imaginary repairing requires whole-register ties",
                         /*gen_crash_diag=*/false);
    if (Operands[Def] < 0) {
      Operands[Def] = Assignments.size();
      Assignments.push_back({Register(), Output.getReg(),
                             Output.isEarlyClobber() ? Through : Defs});
    }
    LocalAssignment &Tied = Assignments[Operands[Def]];
    Tied.Use = Use.getReg();
    Tied.Phases |= Uses;
    if (Operands[I] >= 0 && Assignments[Operands[I]].Phases == Uses)
      Assignments[Operands[I]].Phases = 0;
    Operands[I] = Operands[Def];
  }
  auto FixedRegs = fixedRegisters(MI);
  if (!assign(Assignments, FixedRegs, true) &&
      !assign(Assignments, FixedRegs, false))
    report_fatal_error("MOS imaginary repairing requires spill insertion",
                       /*gen_crash_diag=*/false);

  SmallVector<Copy> Copies;
  SmallVector<Register> Sources(Assignments.size());
  for (unsigned I = 0; I != Assignments.size(); ++I) {
    const LocalAssignment &A = Assignments[I];
    if (!A.Phases || !A.Use)
      continue;
    Register Source = current(A.Use);
    if (!VRM.hasPhys(Source) || VRM.getPhys(Source) != A.Phys) {
      if (MI.isTerminator() && A.Phases == Through)
        report_fatal_error(
            "MOS imaginary repairing cannot split a live-through "
            "value at a terminator",
            /*gen_crash_diag=*/false);
      Copies.push_back(move(A.Use, A.Phys));
      Source = Copies.back().Def;
    }
    Sources[I] = Source;
  }
  insertCopies(*MI.getParent(), MI.getIterator(), Copies);
  for (unsigned I = 0; I != Operands.size(); ++I) {
    MachineOperand &MO = MI.getOperand(I);
    int Index = Operands[I];
    if (!MO.isReg() || !MO.getReg().isVirtual() || !MO.isUse())
      continue;
    Register R = MO.getReg();
    if (Index >= 0 && Sources[Index] != current(R))
      MO.setReg(Sources[Index]);
    else
      recordUse(MI, I, R);
  }
  for (unsigned I = 0; I != Assignments.size(); ++I) {
    const LocalAssignment &A = Assignments[I];
    if (!A.Phases)
      continue;
    if (A.Use && !A.Def && Sources[I] != current(A.Use))
      CurrentRegs[A.Use] = Sources[I];
    if (A.Def) {
      if (!VRM.hasPhys(A.Def) || VRM.getPhys(A.Def) != A.Phys) {
        VRM.clearVirt(A.Def);
        VRM.assignVirt2Phys(A.Def, A.Phys);
        ChangedRegs.set(A.Def);
      }
      CurrentRegs[A.Def] = A.Def;
    }
  }
}

void ImagRegRepair::restoreRegisters(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator InsertPt) {
  SmallVector<Copy> Copies;
  for (Register R : LiveRegs) {
    MCPhysReg Phys = GlobalRegs.lookup(R);
    if (VRM.getPhys(current(R)) == Phys)
      continue;
    if (!PhysRegs.available(MRI, Phys))
      report_fatal_error("MOS imaginary repairing cannot restore a backing "
                         "register across a fixed physical lifetime",
                         /*gen_crash_diag=*/false);
    Copies.push_back(move(R, Phys));
    CurrentRegs[R] = Copies.back().Def;
  }
  insertCopies(MBB, InsertPt, Copies);
}

void ImagRegRepair::repairBlock(MachineBasicBlock &MBB) {
  CurrentRegs.clear();
  computeLiveIns(PhysRegs, MBB);
  bool Restored = false;
  for (MachineInstr &MI : make_early_inc_range(MBB)) {
    if (MI.isDebugInstr()) {
      for (unsigned I = 0; I != MI.getNumOperands(); ++I)
        if (MI.getOperand(I).isReg() && MI.getOperand(I).getReg().isVirtual())
          recordUse(MI, I, MI.getOperand(I).getReg());
      continue;
    }
    if ((MI.isTerminator() && !Restored) || MI.getOpcode() == MOS::PCOPY) {
      restoreRegisters(MBB, MI.getIterator());
      Restored |= MI.isTerminator();
    }
    // Retain the input MIR's lifetime events before rewriting its operands.
    SmallVector<Register> Kills, Definitions;
    for (const MachineOperand &MO : MI.all_uses())
      if (MO.isKill() && MO.getReg().isVirtual() && !MI.isPHI())
        Kills.push_back(MO.getReg());
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.getReg().isVirtual() && GlobalRegs.count(MO.getReg()) &&
          !MO.isDead() && !isReservation(MO.getReg()))
        Definitions.push_back(MO.getReg());
    if (MI.isPHI()) {
      for (unsigned I = 1; I < MI.getNumOperands(); I += 2)
        SplitValues[MI.getOperand(I).getReg()].IncomingUses.push_back({&MI, I});
    } else if (MI.getOpcode() == MOS::PCOPY || MI.isImplicitDef()) {
      for (unsigned I = 0; I != MI.getNumOperands(); ++I)
        if (MI.getOperand(I).isReg() && MI.getOperand(I).isUse())
          recordUse(MI, I, MI.getOperand(I).getReg());
    } else {
      repairInstruction(MI);
    }
    SmallVector<std::pair<MCPhysReg, const MachineOperand *>> ClobberedRegs;
    PhysRegs.stepForward(MI, ClobberedRegs);
    for (Register R : Kills)
      LiveRegs.reset(R);
    for (Register R : Definitions) {
      LiveRegs.set(R);
      CurrentRegs[R] = R;
    }
  }
  if (!Restored)
    restoreRegisters(MBB, MBB.end());
  for (auto [R, Current] : CurrentRegs)
    SplitValues[R].LiveOuts.emplace_back(&MBB, Current);
}

void ImagRegRepair::reconstructSSA() {
  for (Register R : ChangedRegs) {
    SmallVector<MachineInstr *> PHIs;
    MachineSSAUpdater Updater(MF, &PHIs);
    Updater.Initialize(R);
    SplitValue &Value = SplitValues[R];
    MachineBasicBlock *DefBlock = MRI.getVRegDef(R)->getParent();
    Updater.AddAvailableValue(DefBlock, R);
    for (auto [MBB, Reg] : Value.LiveOuts)
      Updater.AddAvailableValue(MBB, Reg);
    for (UsePosition Use : Value.IncomingUses)
      Updater.RewriteUse(Use.MI->getOperand(Use.Operand));
    for (MachineInstr *PHI : PHIs) {
      Register Reg = PHI->getOperand(0).getReg();
      VRM.grow();
      VRM.assignVirt2Phys(Reg, GlobalRegs.lookup(R));
      VRM.setIsSplitFromReg(Reg, VRM.getOriginal(R));
    }
  }
}

MOSImagRegRepair::MOSImagRegRepair() : MachineFunctionPass(ID) {
  initializeMOSImagRegRepairPass(*PassRegistry::getPassRegistry());
}

bool MOSImagRegRepair::runOnMachineFunction(MachineFunction &MF) {
  return ImagRegRepair(
             MF, getAnalysis<VirtRegMapWrapperLegacy>().getVRM(),
             getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI(),
             getAnalysis<LiveVariablesWrapperPass>().getLV(),
             getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree())
      .run();
}

MachineFunctionProperties MOSImagRegRepair::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

MachineFunctionProperties MOSImagRegRepair::getClearedProperties() const {
  return MachineFunctionProperties().setNoPHIs();
}

void MOSImagRegRepair::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addRequired<MachineDominatorTreeWrapperPass>();
  AU.addRequired<MachineRegisterClassInfoWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addPreserved<VirtRegMapWrapperLegacy>();
  AU.addPreserved<MachineRegisterClassInfoWrapperPass>();
  AU.addPreserved<MachineDominatorTreeWrapperPass>();
  // Recomputing LiveVariables must not rerun unreachable-block elimination:
  // that transformation would invalidate the assignments in VirtRegMap.
  AU.addPreservedID(UnreachableMachineBlockElimID);
  AU.setPreservesCFG();
}

} // namespace

char MOSImagRegRepair::ID = 0;
INITIALIZE_PASS_BEGIN(MOSImagRegRepair, DEBUG_TYPE,
                      "MOS imaginary register repairing", false, false)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTreeWrapperPass)
INITIALIZE_PASS_END(MOSImagRegRepair, DEBUG_TYPE,
                    "MOS imaginary register repairing", false, false)

MachineFunctionPass *llvm::createMOSImagRegRepairPass() {
  return new MOSImagRegRepair;
}
