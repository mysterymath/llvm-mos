//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the MOS imaginary register spilling pass.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSSPILL_H
#define LLVM_LIB_TARGET_MOS_MOSSPILL_H

namespace llvm {

class MachineFunctionPass;

MachineFunctionPass *createMOSSpillPass();

} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSSPILL_H
