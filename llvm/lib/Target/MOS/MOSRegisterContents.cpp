//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared register-content operations for imaginary and hardware allocation.
///
//===----------------------------------------------------------------------===//

#include "MOSRegisterContents.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

void MOSRegisterContents::define(MCPhysReg R, ValueNumber V) {
  assert(R);
  // Materializing undef requires no write and preserves existing contents.
  if (V.isUndef())
    return;
  if (!V) {
    clobber(R);
    return;
  }
  Contents.remove_if([&](const auto &Entry) {
    MCPhysReg Other = Entry.first;
    if (Other == R || !TRI->regsOverlap(R, Other))
      return false;
    if (unsigned SubReg = TRI->getSubRegIndex(R, Other)) {
      ValueNumber Part = ValueNumbers->getSubValue(V, SubReg);
      return !Part.isUndef() && Part != Entry.second;
    }
    if (unsigned SubReg = TRI->getSubRegIndex(Other, R)) {
      ValueNumber Part = ValueNumbers->getSubValue(Entry.second, SubReg);
      return !Part.isUndef() && Part != V;
    }
    return true;
  });
  Contents[R] = V;
  for (MCPhysReg Sub : TRI->subregs(R)) {
    ValueNumber Part =
        ValueNumbers->getSubValue(V, TRI->getSubRegIndex(R, Sub));
    if (Part && !Part.isUndef())
      Contents[Sub] = Part;
  }
}

bool MOSRegisterContents::contains(MCPhysReg R, ValueNumber V) const {
  if (!V)
    return false;
  if (V.isUndef() || read(R) == V)
    return true;
  for (unsigned SubReg : ValueNumbers->subRegIndices(R)) {
    MCPhysReg Part = R;
    if (SubReg)
      Part = TRI->getSubReg(R, SubReg);
    ValueNumber Expected = ValueNumbers->getSubValue(V, SubReg);
    if (!Expected || (!Expected.isUndef() && read(Part) != Expected))
      return false;
  }
  return true;
}

bool MOSRegisterContents::hasCopy(ValueNumber V) const {
  return V && llvm::any_of(Contents, [=](const auto &Entry) {
           return Entry.second == V;
         });
}

bool MOSRegisterContents::hasCopyIn(ValueNumber V,
                                    ArrayRef<MCPhysReg> Regs) const {
  return V && llvm::any_of(Regs, [&](MCPhysReg R) { return read(R) == V; });
}

SmallVector<MCPhysReg> MOSRegisterContents::copies(ValueNumber V) const {
  SmallVector<MCPhysReg> Regs;
  for (auto [R, Value] : Contents)
    if (Value == V)
      Regs.push_back(R);
  // Transfer selection must not depend on hash table iteration order.
  llvm::sort(Regs);
  return Regs;
}

void MOSRegisterContents::clobber(MCPhysReg R) {
  Contents.remove_if(
      [&](const auto &Entry) { return TRI->regsOverlap(R, Entry.first); });
}

void MOSRegisterContents::clobber(const uint32_t *RegMask) {
  Contents.remove_if([&](const auto &Entry) {
    return llvm::any_of(TRI->subregs_inclusive(Entry.first), [&](MCPhysReg R) {
      return MachineOperand::clobbersPhysReg(RegMask, R);
    });
  });
}

void MOSRegisterContents::forgetIf(
    function_ref<bool(MCPhysReg, ValueNumber)> Predicate) {
  Contents.remove_if(
      [&](const auto &Entry) { return Predicate(Entry.first, Entry.second); });
}
