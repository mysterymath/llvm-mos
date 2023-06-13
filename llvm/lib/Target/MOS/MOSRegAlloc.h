//===-- MOSRegAlloc.h - MOS Register Allocator ------------------*- C++ -*-===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MOS register allocator pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSREGALLOC_H
#define LLVM_LIB_TARGET_MOS_MOSREGALLOC_H

namespace llvm {
class MachineFunctionPass;
MachineFunctionPass *createMOSRegAllocPass();
} // namespace llvm

#endif // not LLVM_LIB_TARGET_MOS_MOSREGALLOC_H
