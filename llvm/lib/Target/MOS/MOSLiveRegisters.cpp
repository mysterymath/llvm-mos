//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Track liveness and physical values while walking SSA instructions.
///
//===----------------------------------------------------------------------===//

#include "MOSLiveRegisters.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"

using namespace llvm;

MOSLiveRegisters::MOSLiveRegisters(const MOSLiveRegisters &Other) {
  *this = Other;
}

MOSLiveRegisters &MOSLiveRegisters::operator=(const MOSLiveRegisters &Other) {
  if (this == &Other)
    return *this;
  MRI = Other.MRI;
  TRI = Other.TRI;
  ValueNumbers = Other.ValueNumbers;
  VirtRegs = Other.VirtRegs;
  PhysRegs.init(*TRI);
  for (MCPhysReg Reg : Other.PhysRegs)
    PhysRegs.addReg(Reg);
  Contents = Other.Contents;
  return *this;
}

void MOSLiveRegisters::init(const MachineFunction &MF,
                            const MOSValueNumbering &Numbers) {
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  ValueNumbers = &Numbers;
  PhysRegs.init(*TRI);
  Contents.emplace(*TRI, Numbers);
  clear();
}

void MOSLiveRegisters::clear() {
  VirtRegs.clear();
  PhysRegs.clear();
  Contents->clear();
}

void MOSLiveRegisters::beginBlock(const MachineBasicBlock &MBB) {
  PhysRegs.clear();
  Contents->clear();
  if (MBB.isEntryBlock())
    PhysRegs.addLiveInsNoPristines(MBB);
}

void MOSLiveRegisters::stepForward(const MachineInstr &MI) {
  if (MI.isDebugInstr())
    return;
  // The virtual COPY result names the incoming physical value, even when the
  // physical register continues to be live after the copy.
  if (MI.isCopy() && MI.getOperand(0).getReg().isVirtual() &&
      MI.getOperand(1).getReg().isPhysical() && !MI.getOperand(1).isUndef()) {
    MCPhysReg Reg = MI.getOperand(1).getReg();
    if (!Contents->read(Reg))
      Contents->define(Reg, ValueNumbers->getValueNumber(MI.getOperand(0)));
  }
  if (!MI.isPHI())
    for (const MachineOperand &Use : MI.all_uses())
      if (Use.getReg().isVirtual() && Use.isKill())
        VirtRegs.reset(Use.getReg());
  for (const MachineOperand &Def : MI.all_defs())
    if (Def.getReg().isVirtual() && !Def.isDead())
      VirtRegs.set(Def.getReg());

  // LLVM consumes old inputs before establishing the surviving results. In
  // particular, an input kill cannot erase its tied early-clobber result.
  SmallVector<std::pair<MCPhysReg, const MachineOperand *>, 8> Clobbers;
  PhysRegs.stepForward(MI, Clobbers);
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      Contents->clobber(MO.getRegMask());
  for (const MachineOperand &Def : MI.all_defs())
    if (Def.getReg().isPhysical())
      Contents->define(Def.getReg(), ValueNumbers->getValueNumber(Def));

  SmallVector<MCPhysReg, 8> Reserved;
  for (MCPhysReg Reg : PhysRegs)
    if (MRI->isReserved(Reg))
      Reserved.push_back(Reg);
  for (MCPhysReg Reg : Reserved)
    PhysRegs.removeReg(Reg);
  Contents->forgetIf([&](MCPhysReg Reg, ValueNumber) {
    return PhysRegs.available(*MRI, Reg);
  });
}
