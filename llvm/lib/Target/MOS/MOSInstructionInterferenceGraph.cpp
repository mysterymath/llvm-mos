//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Generalized graph simplification for local imaginary register coloring.
///
//===----------------------------------------------------------------------===//

#include "MOSInstructionInterferenceGraph.h"
#include "MOSImagRegAllocUtils.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"

using namespace llvm;

MOSInstructionInterferenceGraph::MOSInstructionInterferenceGraph(
    const MachineInstr &MI, const MOSLiveRegisters &Live,
    const MOSValueNumbering &ValueNumbers, const RegisterClassInfo &RCI)
    : MI(MI), Before(Live), After(Live), MRI(MI.getMF()->getRegInfo()),
      TRI(*MI.getMF()->getSubtarget<MOSSubtarget>().getRegisterInfo()),
      ValueNumbers(ValueNumbers) {
  After.stepForward(MI);
  collectRegisters();
  for (Register Reg : Registers) {
    auto &Regs = Candidates[Reg];
    for (MCPhysReg Phys : RCI.getOrder(TRI.getImagRegClass(Reg, MRI)))
      if (!conflictsWithPhysRegs(Reg, Phys))
        Regs.push_back(Phys);
  }
}

Register MOSInstructionInterferenceGraph::simplify(
    SmallVectorImpl<Register> &SelectStack) const {
  SelectStack.clear();
  // Only the remaining nodes and their squeeze change during simplification.
  // Recompute the removed node's contributions instead of retaining a matrix.
  SmallVector<std::pair<Register, unsigned>> Remaining;
  for (Register Reg : Registers) {
    unsigned Squeeze = 0;
    for (Register Other : Registers)
      if (Other != Reg)
        Squeeze += maxBlockedLocations(Reg, Other);
    Remaining.emplace_back(Reg, Squeeze);
  }
  while (!Remaining.empty()) {
    auto I = llvm::find_if(llvm::reverse(Remaining), [&](const auto &Node) {
      return Node.second < candidatePhysRegs(Node.first).size();
    });
    if (I == Remaining.rend())
      return Remaining.back().first;
    Register Reg = I->first;
    SelectStack.push_back(Reg);
    Remaining.erase(std::next(I).base());
    for (auto &[Other, Squeeze] : Remaining)
      Squeeze -= maxBlockedLocations(Other, Reg);
  }
  // Any eligible node can be removed: doing so only reduces other nodes'
  // squeeze. Whether simplification succeeds is independent of SSA numbering.
  return Register();
}

bool MOSInstructionInterferenceGraph::assignmentsConflict(
    Register A, MCPhysReg APhys, Register B, MCPhysReg BPhys) const {
  if (!TRI.regsOverlap(APhys, BPhys))
    return false;
  // Early-clobber restrictions are operand constraints, even if value
  // numbering happens to establish equal input and output contents.
  if ((isEarlyDef(A) && contentsAt(B, Inputs)) ||
      (isEarlyDef(B) && contentsAt(A, Inputs)))
    return true;
  for (Point At : {Inputs, EarlyDefs, Clobbers, Defs}) {
    auto AValue = contentsAt(A, At), BValue = contentsAt(B, At);
    if (AValue && BValue &&
        !mos::haveCompatibleContents(APhys, *AValue, BPhys, *BValue, TRI,
                                     ValueNumbers))
      return true;
  }
  return false;
}

unsigned
MOSInstructionInterferenceGraph::maxBlockedLocations(Register Reg,
                                                     Register Other) const {
  unsigned Max = 0;
  for (MCPhysReg OtherPhys : candidatePhysRegs(Other)) {
    unsigned Blocked = 0;
    // Only aliases can be blocked. Avoid a Cartesian product of the two
    // allocation orders for every pair of live values.
    for (MCRegAliasIterator Alias(OtherPhys, &TRI, /*IncludeSelf=*/true);
         Alias.isValid(); ++Alias)
      if (llvm::is_contained(candidatePhysRegs(Reg), MCPhysReg(*Alias)) &&
          assignmentsConflict(Reg, *Alias, Other, OtherPhys))
        ++Blocked;
    Max = std::max(Max, Blocked);
  }
  return Max;
}

bool MOSInstructionInterferenceGraph::conflictsWithPhysRegs(
    Register Reg, MCPhysReg Phys) const {
  // Masks destroy preserved incoming values, but explicit results are new
  // definitions. A tied ordinary result consumes its input before the mask.
  if (!definedHere(Reg) && After.liveVirtRegs().test(Reg))
    for (const MachineOperand &MO : MI.operands())
      if (MO.isRegMask() && MO.clobbersPhysReg(Phys))
        return true;
  for (Point At : {Inputs, EarlyDefs, Clobbers, Defs}) {
    auto Value = contentsAt(Reg, At);
    if (!Value)
      continue;
    // Comparing byte views preserves both pair alignment and partial physical
    // lifetimes. Unknown upper/lower components remain occupied, not free.
    for (MCPhysReg Byte : MOS::Imag8RegClass) {
      if (!TRI.regsOverlap(Phys, Byte))
        continue;
      auto PhysValue = physicalContentsAt(Byte, At);
      if (PhysValue && !mos::haveCompatibleContents(
                           Phys, *Value, Byte, *PhysValue, TRI, ValueNumbers))
        return true;
    }
  }
  if (isEarlyDef(Reg))
    for (const MachineOperand &Use : MI.all_uses())
      if (Use.getReg().isPhysical() && Use.readsReg() &&
          TRI.regsOverlap(Phys, Use.getReg()))
        return true;
  return false;
}

void MOSInstructionInterferenceGraph::collectRegisters() {
  for (Register Reg : Before.liveVirtRegs()) {
    if (!needsAssignment(Reg))
      continue;
    // A dying value used only through ties is supplied by those operands'
    // assignments. A separate copy is needed only for other uses or survival.
    if (After.liveVirtRegs().test(Reg) || hasUntiedUse(Reg))
      Registers.push_back(Reg);
  }
  for (const MachineOperand &Def : MI.all_defs())
    if (Def.getReg().isVirtual() && needsAssignment(Def.getReg()))
      Registers.push_back(Def.getReg());
}

bool MOSInstructionInterferenceGraph::needsAssignment(Register Reg) const {
  if (Reg == mos::getImagReservationRoot(Reg, MRI))
    return false; // An undefined reservation holds no current value.
  if ((!definedHere(Reg) || !MRI.use_nodbg_empty(Reg)) &&
      mos::needsImagReg(Reg, *MI.getMF(), ValueNumbers))
    return true;
  if (!definedHere(Reg))
    return false;
  // Dead imaginary-only results still write storage, and tied occurrences of
  // rematerializable values still need an encodable operand assignment.
  unsigned UseIdx;
  if (mos::canUseImagReg(Reg, MRI) &&
      MI.isRegTiedToUseOperand(MRI.def_begin(Reg)->getOperandNo(), &UseIdx)) {
    const MachineOperand &Use = MI.getOperand(UseIdx);
    if (!Use.isUndef() && mos::canUseImagReg(Use.getReg(), MRI))
      return true;
  }
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return MOS::Imag8RegClass.hasSubClassEq(RC) ||
         MOS::Imag16RegClass.hasSubClassEq(RC);
}

bool MOSInstructionInterferenceGraph::hasUntiedUse(Register Reg) const {
  for (const MachineOperand &Use : MI.all_uses()) {
    if (Use.getReg() != Reg || !Use.readsReg())
      continue;
    unsigned DefIdx;
    if (!MI.isRegTiedToDefOperand(Use.getOperandNo(), &DefIdx))
      return true;
    const MachineOperand &Def = MI.getOperand(DefIdx);
    if (!mos::canUseImagReg(Reg, MRI) || !mos::canUseImagReg(Def.getReg(), MRI))
      return true;
  }
  return false;
}

bool MOSInstructionInterferenceGraph::definedHere(Register Reg) const {
  return MRI.getVRegDef(Reg) == &MI;
}

bool MOSInstructionInterferenceGraph::isEarlyDef(Register Reg) const {
  return definedHere(Reg) && MRI.def_begin(Reg)->isEarlyClobber();
}

std::optional<MOSInstructionInterferenceGraph::ValueNumber>
MOSInstructionInterferenceGraph::contentsAt(Register Reg, Point At) const {
  if (definedHere(Reg)) {
    if (At == Defs || (At != Inputs && isEarlyDef(Reg)))
      return ValueNumbers.getValueNumber(Reg);
    const MachineOperand &Def = *MRI.def_begin(Reg);
    unsigned UseIdx;
    if (At != Clobbers && mos::canUseImagReg(Reg, MRI) &&
        MI.isRegTiedToUseOperand(Def.getOperandNo(), &UseIdx)) {
      const MachineOperand &Use = MI.getOperand(UseIdx);
      if (!Use.isUndef() && mos::canUseImagReg(Use.getReg(), MRI)) {
        assert(!Def.getSubReg() && !Use.getSubReg() &&
               "CSSA must extract tied subregister uses");
        return ValueNumbers.getValueNumber(Use.getReg());
      }
    }
    return std::nullopt;
  }
  if (At == Inputs || At == EarlyDefs || After.liveVirtRegs().test(Reg))
    return ValueNumbers.getValueNumber(Reg);
  return std::nullopt;
}

std::optional<MOSInstructionInterferenceGraph::ValueNumber>
MOSInstructionInterferenceGraph::physicalContentsAt(MCPhysReg Reg,
                                                    Point At) const {
  // Definitions matter even when dead: the write must not destroy a value
  // preserved across the instruction. Implicit alias defs describe that same
  // write rather than additional independent values.
  if (At == Defs)
    for (const MachineOperand &Def : MI.all_defs())
      if (Def.getReg().isPhysical() && !Def.isEarlyClobber() &&
          TRI.regsOverlap(Reg, Def.getReg()))
        return physicalDefValue(Def, Reg);
  if (At != Inputs)
    for (const MachineOperand &Def : MI.all_defs())
      if (Def.getReg().isPhysical() && Def.isEarlyClobber() &&
          TRI.regsOverlap(Reg, Def.getReg()))
        return physicalDefValue(Def, Reg);
  const auto &Live = At == Inputs || At == EarlyDefs ? Before : After;
  if (Live.livePhysRegs().available(MRI, Reg))
    return std::nullopt;
  return At == Inputs || At == EarlyDefs ? incomingPhysicalValue(Reg)
                                         : Live.contents(Reg);
}

MOSInstructionInterferenceGraph::ValueNumber
MOSInstructionInterferenceGraph::incomingPhysicalValue(MCPhysReg Reg) const {
  ValueNumber Value = Before.contents(Reg);
  if (Value || !MI.isCopy() || !MI.getOperand(0).getReg().isVirtual() ||
      !MI.getOperand(1).getReg().isPhysical() || MI.getOperand(1).isUndef())
    return Value;
  MCPhysReg Source = MI.getOperand(1).getReg();
  ValueNumber Result = ValueNumbers.getValueNumber(MI.getOperand(0));
  if (Source == Reg)
    return Result;
  if (unsigned SubReg = TRI.getSubRegIndex(Source, Reg))
    return ValueNumbers.getSubValue(Result, SubReg);
  return {};
}

MOSInstructionInterferenceGraph::ValueNumber
MOSInstructionInterferenceGraph::physicalDefValue(const MachineOperand &Def,
                                                  MCPhysReg Reg) const {
  ValueNumber Value = ValueNumbers.getValueNumber(Def);
  if (Def.getReg() == Reg)
    return Value;
  unsigned SubReg = TRI.getSubRegIndex(Def.getReg(), Reg);
  return SubReg ? ValueNumbers.getSubValue(Value, SubReg) : ValueNumber();
}
