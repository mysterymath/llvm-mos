//===-- MOSMachineSSA.h - MOS Machine SSA Construction ----------*- C++ -*-===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the MOS Machine SSA construction pass.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSMACHINESSA_H
#define LLVM_LIB_TARGET_MOS_MOSMACHINESSA_H

namespace llvm {
class MachineFunctionPass;

MachineFunctionPass *createMOSMachineSSAPass();

} // namespace llvm

#endif // not LLVM_LIB_TARGET_MOS_MOSMACHINESSA_H
