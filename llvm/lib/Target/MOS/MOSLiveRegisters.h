//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Live SSA registers and the known contents of live physical registers.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSLIVEREGISTERS_H
#define LLVM_LIB_TARGET_MOS_MOSLIVEREGISTERS_H

#include "MOSRegisterContents.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include <optional>

namespace llvm {
class MachineFunction;
class MachineRegisterInfo;

// Liveness at an instruction boundary. Virtual registers retain their SSA
// names; this class knows neither their assignments nor the spilling policy.
// Reserved physical registers do not require liveness metadata.
class MOSLiveRegisters {
public:
  using ValueNumber = MOSValueNumbering::ValueNumber;

  MOSLiveRegisters() = default;
  MOSLiveRegisters(const MOSLiveRegisters &Other);
  MOSLiveRegisters &operator=(const MOSLiveRegisters &Other);
  void init(const MachineFunction &MF, const MOSValueNumbering &ValueNumbers);
  void clear();
  void beginBlock(const MachineBasicBlock &MBB);
  void inherit(const SparseBitVector<> &LiveOuts) { VirtRegs = LiveOuts; }
  void stepForward(const MachineInstr &MI);

  const SparseBitVector<> &liveVirtRegs() const { return VirtRegs; }
  const LivePhysRegs &livePhysRegs() const { return PhysRegs; }
  ValueNumber contents(MCPhysReg Reg) const { return Contents->read(Reg); }
  void insert(Register Reg) {
    assert(Reg.isVirtual());
    VirtRegs.set(Reg);
  }
  void erase(Register Reg) {
    assert(Reg.isVirtual());
    VirtRegs.reset(Reg);
  }

private:
  const MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  const MOSValueNumbering *ValueNumbers = nullptr;
  SparseBitVector<> VirtRegs;
  LivePhysRegs PhysRegs;
  std::optional<MOSRegisterContents> Contents;
};
} // namespace llvm
#endif
