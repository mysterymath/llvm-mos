//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Identify equal values through whole copies, allocation splits, and the
/// individual bytes of REG_SEQUENCE, independently of their backing registers.
///
//===----------------------------------------------------------------------===//

#include "MOSValueNumbering.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOSRegisterInfo.h"
#include "llvm/ADT/EquivalenceClasses.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"

using namespace llvm;

MOSValueNumbering::MOSValueNumbering(VirtRegMap &VRM)
    : MRI(VRM.getMachineFunction().getRegInfo()),
      TRI(*VRM.getMachineFunction().getSubtarget().getRegisterInfo()),
      VRM(VRM) {}

void MOSValueNumbering::recordCopyOrigins() {
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
          TRI.getRegSizeInBits(*MRI.getRegClass(R)) ==
              TRI.getRegSizeInBits(*MRI.getRegClass(CopySource)))
        Source = CopySource;
    }
    if (Source == R)
      Roots.push_back(R);
    ECs.unionSets(Source, R);
  }
  for (Register Root : Roots)
    for (Register R : ECs.members(Root))
      if (R != Root)
        VRM.setIsSplitFromReg(R, Root);
}

ArrayRef<unsigned> MOSValueNumbering::subRegIndices(Register R) const {
  static constexpr unsigned Whole[] = {0};
  static constexpr unsigned Bytes[] = {MOS::sublo, MOS::subhi};
  bool IsImag16 = R.isPhysical()
                      ? MOS::Imag16RegClass.contains(R)
                      : TRI.getRegSizeInBits(*MRI.getRegClass(R)) == 16;
  return IsImag16 ? ArrayRef<unsigned>(Bytes) : ArrayRef<unsigned>(Whole);
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::getValueNumber(Register R, unsigned SubReg) const {
  assert(!SubReg || llvm::is_contained(subRegIndices(R), SubReg));
  // Copy ancestry preserves size and byte order. REG_SEQUENCE additionally
  // identifies a pair's selected byte with an entire Imag8 source value.
  R = VRM.getOriginal(R);
  const MachineInstr *Def = MRI.getVRegDef(R);
  if (Def && Def->getOpcode() == TargetOpcode::REG_SEQUENCE)
    for (unsigned I = 1; I < Def->getNumOperands(); I += 2) {
      const MachineOperand &Source = Def->getOperand(I);
      if (Def->getOperand(I + 1).getImm() == SubReg &&
          Source.getReg().isVirtual() && !Source.isUndef() &&
          !Source.getSubReg() &&
          TRI.getRegSizeInBits(*MRI.getRegClass(Source.getReg())) == 8)
        return ValueNumber(VRM.getOriginal(Source.getReg()));
    }
  // PHIs and physical-register captures introduce new values. In particular,
  // loop-carried PHI operands are not unconditionally equal to their result.
  return ValueNumber(R, SubReg);
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::getDefValueNumber(const MachineOperand &MO,
                                     unsigned SubReg) const {
  assert(MO.isDef());
  if (MO.getReg().isVirtual())
    return getValueNumber(MO.getReg(), SubReg);
  const MachineInstr &MI = *MO.getParent();
  if (!MI.isFullCopy() || !MI.getOperand(0).getReg().isPhysical() ||
      !MI.getOperand(1).getReg().isVirtual() || MI.getOperand(1).isUndef())
    return {};
  MCPhysReg Dst = MI.getOperand(0).getReg();
  Register Source = MI.getOperand(1).getReg();
  MCPhysReg Part = MO.getReg();
  if (SubReg)
    Part = TRI.getSubReg(Part, SubReg);
  unsigned SourceSubReg = TRI.getSubRegIndex(Dst, Part);
  if ((Part != Dst && !SourceSubReg) ||
      (SourceSubReg &&
       !llvm::is_contained(subRegIndices(Source), SourceSubReg)))
    return {};
  return getValueNumber(Source, SourceSubReg);
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::getSubValue(ValueNumber V, unsigned SubReg) const {
  if (!V.Reg || !SubReg)
    return V;
  if (V.SubReg || !llvm::is_contained(subRegIndices(V.Reg), SubReg))
    return {};
  return getValueNumber(V.Reg, SubReg);
}

bool MOSValueNumbering::sameValue(ValueNumber A, ValueNumber B) const {
  if (A == B)
    return true;
  if (!A.Reg || !B.Reg || A.SubReg || B.SubReg)
    return false;
  return sameValue(A.Reg, B.Reg);
}

bool MOSValueNumbering::sameValue(Register A, Register B) const {
  if (A == B)
    return true;
  if (!A.isVirtual() || !B.isVirtual() ||
      subRegIndices(A).size() != subRegIndices(B).size())
    return false;
  for (unsigned SubReg : subRegIndices(A))
    if (getValueNumber(A, SubReg) != getValueNumber(B, SubReg))
      return false;
  return true;
}
