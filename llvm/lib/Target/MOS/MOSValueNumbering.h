//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared static value identities for MOS imaginary and hardware allocation.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSVALUENUMBERING_H
#define LLVM_LIB_TARGET_MOS_MOSVALUENUMBERING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/CodeGen/TargetInstrInfo.h"

namespace llvm {

class MachineRegisterInfo;
class TargetRegisterInfo;
class VirtRegMap;

class MOSValueNumbering {
public:
  explicit MOSValueNumbering(VirtRegMap &VRM);

  // Names a static value by its source SSA register and subregister index.
  // Index zero denotes the whole value, including an Imag16. Components are
  // resolved through copies and REG_SEQUENCE; whole aggregates retain a name.
  // Different aggregate names can have equal components (see sameValue).
  // A null register denotes unknown contents.
  using ValueNumber = TargetInstrInfo::RegSubRegPair;

  // Establish whole-copy equivalence before allocation. This updates only
  // VirtRegMap's split ancestry, not register assignments. Later splits must
  // inherit getOriginal(Source), preserving size and byte order.
  void recordCopyOrigins();

  // Both bytes of an Imag16, or the whole register (index 0) for an Imag8 or
  // flag. Physical registers are accepted here only to describe their width.
  ArrayRef<unsigned> subRegIndices(Register R) const;
  ValueNumber getValueNumber(Register R, unsigned SubReg = 0) const;
  // Project a value onto a subregister; zero selects the whole value. Return
  // unknown for unsupported projections, such as individual bits of a byte.
  ValueNumber getSubValue(ValueNumber V, unsigned SubReg) const;
  // Value produced by a definition, including the bytes of an explicit
  // physical COPY and its implicit alias defs. Opaque physical results have
  // unknown identities until named by SSA captures.
  ValueNumber getDefValueNumber(const MachineOperand &MO,
                                unsigned SubReg = 0) const;
  // Whole-register equality; unequal widths cannot share a whole assignment.
  bool sameValue(Register A, Register B) const;
  bool sameValue(ValueNumber A, ValueNumber B) const;

private:
  // Queries consult current MIR and ancestry directly: splitting or rewriting
  // instructions does not leave a separate value-number cache to invalidate.
  const MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  VirtRegMap &VRM;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSVALUENUMBERING_H
