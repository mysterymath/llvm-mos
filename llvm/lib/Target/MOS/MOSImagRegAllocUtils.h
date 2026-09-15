//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Shared imaginary-storage requirements for spilling and register assignment.
/// These queries use MIR and analyses; they own no liveness or allocation
/// state.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSIMAGREGALLOCUTILS_H
#define LLVM_LIB_TARGET_MOS_MOSIMAGREGALLOCUTILS_H

#include "llvm/CodeGen/Register.h"

namespace llvm {
class LiveVariables;
class MachineFunction;
class MachineRegisterInfo;
class MOSValueNumbering;

namespace mos {

// Whether R needs imaginary storage, accounting for rematerialization, undef,
// and CSSA reservations. Physical imaginary registers contribute demand unless
// reserved; their fixed locations are handled during assignment and repair.
bool needsImagReg(Register R, const MachineFunction &MF,
                  const MOSValueNumbering &ValueNumbers);

// Return the CSSA reservation's IMPLICIT_DEF, or zero if R has no reservation.
Register getImagReservationRoot(Register R, const MachineRegisterInfo &MRI);

// Whether Range needs storage in one of Root's reserved predecessor exits:
// after an exit PCOPY through the terminators and outgoing PHI edge uses.
// Root must be a reservation IMPLICIT_DEF. If Range is another reservation
// root, compare their reserved exits rather than their IMPLICIT_DEF lifetimes.
// This query does not inspect locations or value identities.
bool overlapsImagReservation(Register Root, Register Range,
                             const MachineRegisterInfo &MRI, LiveVariables &LV);

} // namespace mos
} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSIMAGREGALLOCUTILS_H
