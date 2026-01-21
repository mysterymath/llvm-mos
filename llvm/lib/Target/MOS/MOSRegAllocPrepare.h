//===-- MOSRegAllocPrepare.h - Prepare for MOS RegAlloc ---------*- C++ -*-===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MOS register allocation preparation pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSREGALLOCPREPARE_H
#define LLVM_LIB_TARGET_MOS_MOSREGALLOCPREPARE_H

#include "llvm/CodeGen/MachineFunctionPass.h"

namespace llvm {

MachineFunctionPass *createMOSRegAllocPreparePass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSREGALLOCPREPARE_H
