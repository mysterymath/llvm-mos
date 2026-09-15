//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Imaginary-storage requirements shared by spilling and assignment.
///
//===----------------------------------------------------------------------===//

#include "MOSImagRegAllocUtils.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOSRegisterInfo.h"
#include "MOSValueNumbering.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

static bool overlapsExit(Register R, const MachineInstr &Copy,
                         const MachineRegisterInfo &MRI, LiveVariables &LV);

bool mos::needsImagReg(Register R, const MachineFunction &MF,
                       const MOSValueNumbering &ValueNumbers) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  const TargetRegisterInfo &TRI = *MF.getSubtarget().getRegisterInfo();
  if (R.isPhysical())
    return R &&
           (MOS::Imag8RegClass.contains(R) ||
            MOS::Imag16RegClass.contains(R)) &&
           !MRI.isReserved(R);
  if (MRI.use_nodbg_empty(R))
    return false;
  // A reservation's IMPLICIT_DEF is rematerializable, but reserves an imaginary
  // register for its PHI. Its incoming copies must also inherit that register
  // even when their sources are rematerializable.
  if (mos::getImagReservationRoot(R, MRI))
    return true;
  auto V = ValueNumbers.getValueNumber(R);
  if (V.isUndef()) {
    // An imaginary-only operand still needs an encodable location, even when
    // nothing needs to be stored there.
    const TargetRegisterClass *RC = MRI.getRegClass(R);
    return MOS::Imag8RegClass.hasSubClassEq(RC) ||
           MOS::Imag16RegClass.hasSubClassEq(RC);
  }
  const MachineInstr *Def = MRI.getVRegDef(ValueNumbers.source(V).Reg);
  return !Def ||
         !MF.getSubtarget().getInstrInfo()->isTriviallyReMaterializable(*Def);
}

Register mos::getImagReservationRoot(Register R,
                                     const MachineRegisterInfo &MRI) {
  if (!R.isVirtual())
    return Register();
  const MachineOperand *Def = &*MRI.def_begin(R);
  const MachineInstr *MI = Def->getParent();
  if (MI->isImplicitDef()) {
    auto Uses = MRI.use_nodbg_operands(R);
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
    Def = &*MRI.def_begin(MI->getOperand(1).getReg());
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
  assert(MRI.getVRegDef(Root)->isImplicitDef() &&
         "expected a reservation root");
  return Root;
}

bool mos::overlapsImagReservation(Register Root, Register Range,
                                  const MachineRegisterInfo &MRI,
                                  LiveVariables &LV) {
  assert(Root && Root == getImagReservationRoot(Root, MRI) &&
         "expected a reservation root");
  bool IsReservation = Range == mos::getImagReservationRoot(Range, MRI);
  for (const MachineInstr &Copy : MRI.use_nodbg_instructions(Root)) {
    assert(Copy.getOpcode() == MOS::PCOPY && "unexpected reservation use");
    // Roots conflict at shared exit PCOPYs. Other registers conflict if they
    // need their assigned register anywhere from the exit PCOPY through the
    // terminators.
    if (IsReservation ? Copy.readsRegister(Range, /*TRI=*/nullptr)
                      : overlapsExit(Range, Copy, MRI, LV))
      return true;
  }
  return false;
}

static bool overlapsExit(Register R, const MachineInstr &Copy,
                         const MachineRegisterInfo &MRI, LiveVariables &LV) {
  const MachineBasicBlock &MBB = *Copy.getParent();
  if (Copy.definesRegister(R, /*TRI=*/nullptr))
    return true;
  // Ordinary live-outs and PHI edge uses both extend through the exit region.
  if (llvm::any_of(MBB.successors(), [&](const MachineBasicBlock *Succ) {
        return LV.isLiveIn(R, *Succ);
      }))
    return true;
  for (const MachineOperand &Use : MRI.use_nodbg_operands(R)) {
    const MachineInstr &MI = *Use.getParent();
    if (MI.isPHI() && MI.getOperand(Use.getOperandNo() + 1).getMBB() == &MBB)
      return true;
  }

  // The exit PCOPY precedes the terminators. A value killed there, or even a
  // dead definition there, still needs storage alongside its destinations.
  // Sources killed by the PCOPY itself can reuse their locations.
  // During block-local repair, these are the original range's kill points.
  // Its liveness is recomputed once outgoing uses have been repaired.
  const auto &Kills = LV.getVarInfo(R).Kills;
  const MachineInstr *Def = MRI.getVRegDef(R);
  for (const MachineInstr &MI :
       make_range(std::next(Copy.getIterator()), MBB.instr_end()))
    if (&MI == Def || llvm::is_contained(Kills, &MI))
      return true;
  return false;
}
