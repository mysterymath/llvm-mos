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

  // Uniquely identifies a static value by its canonical source SSA register
  // and subregister index. A null register denotes unknown contents.
  using ValueNumber = TargetInstrInfo::RegSubRegPair;

  // Establish whole-copy equivalence before allocation. This updates only
  // VirtRegMap's split ancestry, not register assignments. Later splits must
  // inherit getOriginal(Source), preserving size and byte order.
  void recordCopyOrigins();

  // Both bytes of an Imag16, or the whole register (index 0) for an Imag8 or
  // flag. Physical registers are accepted here only to describe their width.
  ArrayRef<unsigned> subRegIndices(Register R) const;
  ValueNumber getValueNumber(Register R, unsigned SubReg = 0) const;
  // Whole-register equality; unequal widths cannot share a whole assignment.
  bool sameValue(Register A, Register B) const;

private:
  // Queries consult current MIR and ancestry directly: splitting or rewriting
  // instructions does not leave a separate value-number cache to invalidate.
  const MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  VirtRegMap &VRM;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSVALUENUMBERING_H
