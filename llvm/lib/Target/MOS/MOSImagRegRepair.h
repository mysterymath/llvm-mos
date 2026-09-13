//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSIMAGREGREPAIR_H
#define LLVM_LIB_TARGET_MOS_MOSIMAGREGREPAIR_H

namespace llvm {
class MachineFunctionPass;

// Split assigned imaginary live ranges to satisfy physical imaginary operands,
// clobbers, and ties. Keep the global assignments at basic block boundaries.
MachineFunctionPass *createMOSImagRegRepairPass();
} // namespace llvm

#endif
