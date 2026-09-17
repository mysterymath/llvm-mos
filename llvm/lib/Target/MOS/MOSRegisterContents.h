//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Known contents of physical storage, independently of register liveness.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSREGISTERCONTENTS_H
#define LLVM_LIB_TARGET_MOS_MOSREGISTERCONTENTS_H

#include "MOSValueNumbering.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

class TargetRegisterInfo;

class MOSRegisterContents {
public:
  using ValueNumber = MOSValueNumbering::ValueNumber;

  MOSRegisterContents(const TargetRegisterInfo &TRI,
                      const MOSValueNumbering &ValueNumbers)
      : TRI(&TRI), ValueNumbers(&ValueNumbers) {}

  // Whole-register definitions record both the whole value and its known
  // subregister values. Undef components impose no write or contents
  // requirement. A partial overwrite invalidates incompatible aliases.
  void define(MCPhysReg R, ValueNumber V);
  void copy(MCPhysReg Dst, MCPhysReg Src) { define(Dst, read(Src)); }

  // Missing contents are unknown, not free. Independent subregister writes do
  // not invent a number for super-registers.
  ValueNumber read(MCPhysReg R) const { return Contents.lookup(R); }
  // Test contents against a whole value or component. This can succeed from
  // component identities even when read(R) has no whole-value name to return.
  bool contains(MCPhysReg R, ValueNumber V) const;
  bool hasCopy(ValueNumber V) const;
  bool hasCopyIn(ValueNumber V, ArrayRef<MCPhysReg> Regs) const;
  SmallVector<MCPhysReg> copies(ValueNumber V) const;

  // Invalidate all contents affected by an overwrite or register mask,
  // including partial aliases. Neither operation establishes a lifetime.
  void clobber(MCPhysReg R);
  void clobber(const uint32_t *RegMask);
  // Callers decide when knowledge ceases to be useful. A dead copy may remain
  // useful to hardware allocation even after its physical live range ends.
  void forgetIf(function_ref<bool(MCPhysReg, ValueNumber)> Predicate);
  void clear() { Contents.clear(); }

private:
  const TargetRegisterInfo *TRI;
  const MOSValueNumbering *ValueNumbers;
  SmallDenseMap<MCPhysReg, ValueNumber, 8> Contents;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSREGISTERCONTENTS_H
