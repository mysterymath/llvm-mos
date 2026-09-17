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

#include "MOSValueNumbering.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/Register.h"

namespace llvm {
class LiveVariables;
class MachineFunction;
class MachineRegisterInfo;
class MachineDominatorTree;
class MOSValueNumbering;

namespace mos {

// Whether overlapping storage can hold both values. Undef imposes no contents
// requirement; unknown values do not establish equality.
bool haveCompatibleContents(MCPhysReg Reg, MOSValueNumbering::ValueNumber Value,
                            MCPhysReg OtherReg,
                            MOSValueNumbering::ValueNumber OtherValue,
                            const TargetRegisterInfo &TRI,
                            const MOSValueNumbering &ValueNumbers);

// Whether Reg is virtual and its class admits imaginary register operands.
// Unlike needsImagReg, this tests operand eligibility, not storage needs.
bool canUseImagReg(Register Reg, const MachineRegisterInfo &MRI);

// Physical registers that a global assignment must avoid at restoration
// points. Reservation roots inherit the exclusions of their isolated members.
DenseMap<Register, BitVector>
computeRestoreExclusions(MachineFunction &MF, LiveVariables &LV,
                         const MachineDominatorTree &MDT,
                         const MOSValueNumbering &ValueNumbers);

// Whether a live R needs imaginary storage, accounting for rematerialization,
// undef, and CSSA reservations. Physical imaginary registers contribute demand
// unless reserved; their location constraints are handled during assignment and
// repair. Callers determine liveness: a block-local split can temporarily have
// no uses until its restoration and outgoing SSA repair have been emitted.
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
