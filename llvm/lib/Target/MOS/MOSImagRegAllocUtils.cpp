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
#include "MOSLiveRegisters.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"
#include "MOSValueNumbering.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;

using ValueNumber = MOSValueNumbering::ValueNumber;

bool mos::haveCompatibleContents(MCPhysReg Reg, ValueNumber Value,
                                 MCPhysReg OtherReg, ValueNumber OtherValue,
                                 const TargetRegisterInfo &TRI,
                                 const MOSValueNumbering &ValueNumbers) {
  if (!TRI.regsOverlap(Reg, OtherReg))
    return true;
  for (unsigned SubReg : ValueNumbers.subRegIndices(Reg)) {
    MCPhysReg Part = Reg;
    if (SubReg)
      Part = TRI.getSubReg(Reg, SubReg);
    auto PartValue = ValueNumbers.getSubValue(Value, SubReg);
    for (unsigned OtherSubReg : ValueNumbers.subRegIndices(OtherReg)) {
      MCPhysReg OtherPart = OtherReg;
      if (OtherSubReg)
        OtherPart = TRI.getSubReg(OtherReg, OtherSubReg);
      if (!TRI.regsOverlap(Part, OtherPart))
        continue;
      auto OtherPartValue = ValueNumbers.getSubValue(OtherValue, OtherSubReg);
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

bool mos::canUseImagReg(Register Reg, const MachineRegisterInfo &MRI) {
  if (!Reg.isVirtual())
    return false;
  const auto &TRI = *MRI.getMF().getSubtarget<MOSSubtarget>().getRegisterInfo();
  return TRI.getCommonSubClass(MRI.getRegClass(Reg),
                               TRI.getImagRegClass(Reg, MRI));
}

static bool overlapsExit(Register R, const MachineInstr &Copy,
                         const MachineRegisterInfo &MRI, LiveVariables &LV);

// Locations that cannot be overwritten when global assignments are restored.
// A terminator suffix is indivisible: copies cannot be inserted after a branch.
static BitVector physRegsAtRestore(const MachineInstr &MI,
                                   const MOSLiveRegisters &Live,
                                   const MOSValueNumbering &ValueNumbers) {
  const MachineFunction &MF = *MI.getMF();
  const auto &MRI = MF.getRegInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  BitVector PhysRegs(TRI.getNumRegs());
  for (MCPhysReg Phys : Live.livePhysRegs())
    if (mos::needsImagReg(Phys, MF, ValueNumbers))
      PhysRegs.set(Phys);
  if (!MI.isTerminator())
    return PhysRegs;
  for (const MachineInstr &Term : MI.getParent()->terminators()) {
    for (const MachineOperand &MO : Term.operands()) {
      if (MO.isReg() && MO.getReg().isVirtual() && MO.isDef())
        report_fatal_error("MOS imaginary allocation does not support "
                           "virtual definitions in terminators",
                           false);
      if (MO.isReg() && MO.getReg().isPhysical() &&
          mos::needsImagReg(MO.getReg(), MF, ValueNumbers))
        PhysRegs.set(MO.getReg());
      if (MO.isRegMask())
        for (MCPhysReg Phys : MOS::Imag8RegClass)
          if (!MRI.isReserved(Phys) && MO.clobbersPhysReg(Phys))
            PhysRegs.set(Phys);
    }
  }
  return PhysRegs;
}

DenseMap<Register, BitVector>
mos::computeRestoreExclusions(MachineFunction &MF, LiveVariables &LV,
                              const MachineDominatorTree &MDT,
                              const MOSValueNumbering &ValueNumbers) {
  const auto &MRI = MF.getRegInfo();
  const auto &TRI = *MF.getSubtarget().getRegisterInfo();
  DenseMap<Register, BitVector> Exclusions;
  MOSLiveRegisters Live;
  Live.init(MF, ValueNumbers);
  SmallVector<std::pair<const MachineDomTreeNode *, SparseBitVector<>>>
      DomLiveRegs;
  for (const MachineDomTreeNode *Node : depth_first(MDT.getRootNode())) {
    auto &MBB = *Node->getBlock();
    while (!DomLiveRegs.empty() && DomLiveRegs.back().first != Node->getIDom())
      DomLiveRegs.pop_back();
    if (!DomLiveRegs.empty()) {
      Live.inherit(DomLiveRegs.back().second);
      for (auto I = Live.liveVirtRegs().begin(), E = Live.liveVirtRegs().end();
           I != E;) {
        Register Reg = *I++;
        if (!LV.isLiveIn(Reg, MBB))
          Live.erase(Reg);
      }
    }
    Live.beginBlock(MBB);
    for (MachineInstr &MI : MBB) {
      bool AtTerminators = MI.getIterator() == MBB.getFirstTerminator();
      if (AtTerminators || MI.getOpcode() == MOS::PCOPY) {
        BitVector PhysRegs = physRegsAtRestore(MI, Live, ValueNumbers);
        if (PhysRegs.any()) {
          SmallVector<Register> Regs;
          for (Register Reg : Live.liveVirtRegs())
            Regs.push_back(Reg);
          if (MI.getOpcode() == MOS::PCOPY)
            for (const MachineOperand &Def : MI.all_defs())
              if (Def.getReg().isVirtual())
                Regs.push_back(Def.getReg());
          for (Register Reg : Regs) {
            for (Register Name : {Reg, mos::getImagReservationRoot(Reg, MRI)}) {
              if (!Name)
                continue;
              auto &Excluded = Exclusions[Name];
              Excluded.resize(TRI.getNumRegs());
              Excluded |= PhysRegs;
            }
          }
        }
      }
      Live.stepForward(MI);
    }
    DomLiveRegs.emplace_back(Node, Live.liveVirtRegs());
  }
  return Exclusions;
}

bool mos::needsImagReg(Register R, const MachineFunction &MF,
                       const MOSValueNumbering &ValueNumbers) {
  const MachineRegisterInfo &MRI = MF.getRegInfo();
  if (R.isPhysical())
    return R &&
           (MOS::Imag8RegClass.contains(R) ||
            MOS::Imag16RegClass.contains(R)) &&
           !MRI.isReserved(R);
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
