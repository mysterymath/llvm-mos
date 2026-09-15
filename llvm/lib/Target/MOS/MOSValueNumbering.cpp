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
/// individual bytes of REG_SEQUENCE, independently of their imaginary
/// registers.
///
//===----------------------------------------------------------------------===//

#include "MOSValueNumbering.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

MOSValueNumbering::MOSValueNumbering(MachineFunction &MF)
    : MRI(MF.getRegInfo()), TRI(*MF.getSubtarget().getRegisterInfo()) {
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register R = Register::index2VirtReg(I);
    // Instruction selection can leave unused generic registers without a
    // register class. They do not name any value in the selected MIR.
    if (!MRI.reg_nodbg_empty(R))
      numberRegister(R);
  }
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::getValueNumber(Register R, unsigned SubReg) const {
  if (!R.isVirtual())
    return {};
  ValueNumber V = RegValueNumbers.lookup({R});
  assert(V && "unrecorded SSA register");
  return getSubValue(V, SubReg);
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::getValueNumber(const MachineOperand &MO) const {
  assert(MO.isReg());
  if (MO.isUse() && MO.isUndef())
    return ValueNumber(ValueNumber::UndefID);
  if (MO.getReg().isVirtual())
    return getValueNumber(MO.getReg(), MO.getSubReg());
  if (MO.isUse())
    return {};
  const MachineInstr &MI = *MO.getParent();
  if (MI.isImplicitDef())
    return ValueNumber(ValueNumber::UndefID);
  if (!MI.isCopy() || !MI.getOperand(0).getReg().isPhysical() ||
      !MI.getOperand(1).getReg().isVirtual())
    return {};
  assert(!MI.getOperand(1).isUndef() && "undef COPY source must be simplified");
  MCPhysReg Dst = MI.getOperand(0).getReg();
  const MachineOperand &Source = MI.getOperand(1);
  ValueNumber V = getValueNumber(Source);
  // LiveVariables can represent a partly used physical definition as
  // "dead $rs1 = COPY %pair, implicit-def $rc3": the whole pair is dead,
  // but its high byte remains live. MO may be that implicit definition,
  // describing part of the same write rather than a separate value.
  MCPhysReg Part = MO.getReg();
  unsigned SubReg = TRI.getSubRegIndex(Dst, Part);
  if (Part != Dst && !SubReg)
    return {};
  // First select the copied source value, then project it for alias defs.
  return getSubValue(V, SubReg);
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::getSubValue(ValueNumber V, unsigned SubReg) const {
  if (!V || V.isUndef() || !SubReg)
    return V;
  auto Source = source(V);
  if (Source.SubReg)
    return {};
  return RegValueNumbers.lookup({Source.Reg, SubReg});
}

void MOSValueNumbering::recordCopy(Register R, Register Source) {
  assert(TRI.getRegSizeInBits(*MRI.getRegClass(R)) ==
         TRI.getRegSizeInBits(*MRI.getRegClass(Source)));
  ValueNumber V = getValueNumber(Source);
  assert((!RegValueNumbers.lookup({R}) || RegValueNumbers.lookup({R}) == V) &&
         "changed a value's identity");
  RegValueNumbers[{R}] = V;
}

TargetInstrInfo::RegSubRegPair MOSValueNumbering::source(ValueNumber V) const {
  return ValueNumbers[V.ID];
}

ArrayRef<unsigned> MOSValueNumbering::subRegIndices(Register R) const {
  static constexpr unsigned Whole[] = {0};
  static constexpr unsigned Bytes[] = {MOS::sublo, MOS::subhi};
  bool HasBytes = llvm::all_of(Bytes, [&](unsigned SubReg) {
    return R.isPhysical()
               ? bool(TRI.getSubReg(R, SubReg))
               : TRI.isSubRegValidForRegClass(MRI.getRegClass(R), SubReg);
  });
  return HasBytes ? ArrayRef<unsigned>(Bytes) : ArrayRef<unsigned>(Whole);
}

MOSValueNumbering::ValueNumber MOSValueNumbering::numberRegister(Register R) {
  if (ValueNumber V = RegValueNumbers.lookup({R}))
    return V;
  if (const MachineOperand *Source = copySource(R)) {
    ValueNumber V = numberOperand(*Source);
    if (V.isUndef())
      return RegValueNumbers[{R}] = V;
    if (V) {
      unsigned SourceBits =
          Source->getSubReg()
              ? TRI.getSubRegIdxSize(Source->getSubReg())
              : TRI.getRegSizeInBits(*MRI.getRegClass(Source->getReg()));
      // Only identify the whole destination with an equally sized source.
      if (SourceBits == TRI.getRegSizeInBits(*MRI.getRegClass(R)))
        return RegValueNumbers[{R}] = V;
    }
  }
  const MachineInstr *Def = MRI.getVRegDef(R);
  if (Def && Def->isImplicitDef())
    return RegValueNumbers[{R}] = ValueNumber(ValueNumber::UndefID);
  if (Def && Def->isRegSequence()) {
    ValueNumber Lo(ValueNumber::UndefID), Hi(ValueNumber::UndefID);
    for (unsigned I = 1; I < Def->getNumOperands(); I += 2) {
      unsigned SubReg = Def->getOperand(I + 1).getImm();
      ValueNumber V = numberOperand(Def->getOperand(I));
      // An unmodeled source still defines bits; it is not undef.
      if (!V)
        V = newValue({R, SubReg});
      if (SubReg == MOS::sublo)
        Lo = V;
      else if (SubReg == MOS::subhi)
        Hi = V;
    }
    return RegValueNumbers[{R}] = numberRegSequence(R, Lo, Hi);
  }

  // PHIs stop traversal, including loop-carried cycles. A physical capture is
  // likewise a new identity; we do not infer physical-register round trips.
  if (subRegIndices(R).size() == 2) {
    ValueNumber Lo = newValue({R, MOS::sublo});
    ValueNumber Hi = newValue({R, MOS::subhi});
    return RegValueNumbers[{R}] = numberRegSequence(R, Lo, Hi);
  }
  return RegValueNumbers[{R}] = newValue({R});
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::numberOperand(const MachineOperand &MO) {
  assert(MO.isReg() && MO.isUse());
  if (MO.isUndef())
    return ValueNumber(ValueNumber::UndefID);
  if (!MO.getReg().isVirtual())
    return {};
  return getSubValue(numberRegister(MO.getReg()), MO.getSubReg());
}

const MachineOperand *MOSValueNumbering::copySource(Register R) const {
  const MachineInstr *Def = MRI.getVRegDef(R);
  if (!Def)
    return nullptr;
  if (Def->isCopy() && !Def->getOperand(0).getSubReg()) {
    assert(!Def->getOperand(1).isUndef() &&
           "undef COPY source must be simplified");
    return &Def->getOperand(1);
  }
  if (Def->getOpcode() == MOS::PCOPY)
    for (unsigned I = 0, E = Def->getNumExplicitDefs(); I != E; ++I)
      if (Def->getOperand(I).getReg() == R && !Def->getOperand(I).getSubReg())
        return &Def->getOperand(E + I);
  return nullptr;
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::numberRegSequence(Register R, ValueNumber Lo,
                                     ValueNumber Hi) {
  if (Lo.isUndef() && Hi.isUndef())
    return ValueNumber(ValueNumber::UndefID);
  auto [I, Inserted] = RegSequenceValueNumbers.try_emplace({Lo, Hi});
  if (Inserted) {
    I->second = newValue({R});
    RegValueNumbers[{R, MOS::sublo}] = Lo;
    RegValueNumbers[{R, MOS::subhi}] = Hi;
  }
  return I->second;
}

MOSValueNumbering::ValueNumber
MOSValueNumbering::newValue(TargetInstrInfo::RegSubRegPair Source) {
  ValueNumber V(ValueNumbers.size());
  ValueNumbers.push_back(Source);
  return V;
}

MOSValueNumberingWrapperPass::MOSValueNumberingWrapperPass()
    : MachineFunctionPass(ID) {
  initializeMOSValueNumberingWrapperPassPass(*PassRegistry::getPassRegistry());
}

bool MOSValueNumberingWrapperPass::runOnMachineFunction(MachineFunction &MF) {
  ValueNumbers.emplace(MF);
  return false;
}

MachineFunctionProperties
MOSValueNumberingWrapperPass::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

void MOSValueNumberingWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.setPreservesAll();
}

char MOSValueNumberingWrapperPass::ID = 0;
INITIALIZE_PASS(MOSValueNumberingWrapperPass, "mos-value-numbering",
                "MOS value numbering", false, true)
