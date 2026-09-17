//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Instruction interference graphs shared by imaginary spilling and coloring.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSINSTRUCTIONINTERFERENCEGRAPH_H
#define LLVM_LIB_TARGET_MOS_MOSINSTRUCTIONINTERFERENCEGRAPH_H

#include "MOSLiveRegisters.h"
#include "llvm/ADT/DenseMap.h"

namespace llvm {
class MOSRegisterInfo;
class RegisterClassInfo;

// Generalized interference graph for coloring one instruction's imaginary
// register requirements. Nodes are SSA names; colors are physical imaginary
// registers. Incoming names represent values to read or preserve. A result
// node also represents its tied input occurrence: they must have the same
// color. An input that survives a destructive tie has a separate node.
//
// Edges constrain pairs of colors, accounting for aliasing and equal contents;
// they cannot be represented just as "different register numbers". Physical
// register lifetimes restrict the allowed colors of each node. The graph is
// implicit: pairwise queries derive interference from the instruction rather
// than storing an adjacency matrix.
//
// No colors are assigned here. The instruction and incoming live registers
// must remain unchanged while the graph is queried. The outgoing snapshot
// distinguishes input-only values from values that must survive the
// instruction.
class MOSInstructionInterferenceGraph {
public:
  using ValueNumber = MOSValueNumbering::ValueNumber;

  MOSInstructionInterferenceGraph(const MachineInstr &MI,
                                  const MOSLiveRegisters &Live,
                                  const MOSValueNumbering &ValueNumbers,
                                  const RegisterClassInfo &RCI);

  // Simplify by removing nodes whose squeeze is less than their color count.
  // On success, return zero; popping SelectStack gives a guaranteed coloring
  // order. Otherwise return an unsimplified node. Failure is not proof that
  // coloring is impossible: this is a sufficient test, without optimistic
  // coloring.
  Register simplify(SmallVectorImpl<Register> &SelectStack) const;

  ArrayRef<Register> regs() const { return Registers; }
  ArrayRef<MCPhysReg> candidatePhysRegs(Register Reg) const {
    return Candidates.find(Reg)->second;
  }
  // Pairwise compatibility of the supplied assignments throughout MI. This
  // includes the tied input when an assignment names a tied result.
  bool assignmentsConflict(Register A, MCPhysReg APhys, Register B,
                           MCPhysReg BPhys) const;

private:
  enum Point { Inputs, EarlyDefs, Clobbers, Defs };
  // The maximum number of Reg's colors one color of Other can exclude.
  // Summing these directed bounds gives Reg's squeeze during simplification.
  unsigned maxBlockedLocations(Register Reg, Register Other) const;
  bool conflictsWithPhysRegs(Register Reg, MCPhysReg Phys) const;
  void collectRegisters();
  bool needsAssignment(Register Reg) const;
  bool hasUntiedUse(Register Reg) const;
  bool definedHere(Register Reg) const;
  bool isEarlyDef(Register Reg) const;
  std::optional<ValueNumber> contentsAt(Register Reg, Point At) const;
  std::optional<ValueNumber> physicalContentsAt(MCPhysReg Reg, Point At) const;
  ValueNumber incomingPhysicalValue(MCPhysReg Reg) const;
  ValueNumber physicalDefValue(const MachineOperand &Def, MCPhysReg Reg) const;

  const MachineInstr &MI;
  const MOSLiveRegisters &Before;
  MOSLiveRegisters After;
  const MachineRegisterInfo &MRI;
  const MOSRegisterInfo &TRI;
  const MOSValueNumbering &ValueNumbers;
  SmallVector<Register> Registers;
  DenseMap<Register, SmallVector<MCPhysReg, 16>> Candidates;
};
} // namespace llvm
#endif
