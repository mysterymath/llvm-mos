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
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include <optional>

namespace llvm {

class MachineRegisterInfo;

// Canonical identities for the values named by SSA registers. Copies and
// projections preserve identity; constructing the same ordered pair of bytes
// produces the same identity. PHIs and opaque definitions introduce new values.
// This is structural numbering, not general instruction equivalence analysis.
class MOSValueNumbering {
public:
  // An index into ValueNumbers, privately minted by this analysis and stable
  // for its lifetime. The default value means unknown contents, rather than
  // a value known equal to other unknown contents.
  class ValueNumber {
  public:
    ValueNumber() = default;
    explicit operator bool() const { return ID != 0; }
    // Undef imposes no contents requirement; it is distinct from unknown.
    // Equality of sentinels does not prove equal register contents. Operand
    // register identities and ties still constrain instruction placement.
    bool isUndef() const { return ID == UndefID; }
    bool operator==(ValueNumber Other) const { return ID == Other.ID; }
    bool operator!=(ValueNumber Other) const { return !(*this == Other); }

  private:
    friend class MOSValueNumbering;
    friend struct DenseMapInfo<ValueNumber>;
    static constexpr unsigned UndefID = 1;
    explicit ValueNumber(unsigned ID) : ID(ID) {}
    unsigned ID = 0;
  };

  explicit MOSValueNumbering(MachineFunction &MF);

  // Physical registers have no static identity and return unknown. Their
  // contents must be tracked separately over their lifetimes.
  ValueNumber getValueNumber(Register R, unsigned SubReg = 0) const;
  // Include the operand's subregister and return undef for undef uses. Other
  // physical uses return unknown; physical definitions identify COPY results
  // and their aliases when possible. Undef flags on definitions do not affect
  // their value.
  ValueNumber getValueNumber(const MachineOperand &MO) const;
  // Zero selects the whole value. Unsupported projections return unknown.
  ValueNumber getSubValue(ValueNumber V, unsigned SubReg) const;

  // Record an allocation split or repair PHI known to preserve Source's value.
  // The new register need not have a definition yet. Its width must match.
  void recordCopy(Register R, Register Source);

  // An SSA definition (or component of it) representing V. This supplies the
  // recipe for rematerialization, not a register guaranteed to dominate a use.
  TargetInstrInfo::RegSubRegPair source(ValueNumber V) const;
  // Both bytes of an Imag16, or index zero for an Imag8 or flag. Physical
  // registers are accepted here only to describe their width.
  ArrayRef<unsigned> subRegIndices(Register R) const;

private:
  ValueNumber numberRegister(Register R);
  // Number a use's source on demand, including undef and subregister semantics.
  ValueNumber numberOperand(const MachineOperand &MO);
  const MachineOperand *copySource(Register R) const;

  ValueNumber numberRegSequence(Register R, ValueNumber Lo, ValueNumber Hi);
  ValueNumber newValue(TargetInstrInfo::RegSubRegPair Source);

  const MachineRegisterInfo &MRI;
  const TargetRegisterInfo &TRI;
  // Each number identifies a representative SSA definition or component.
  // Zero is unknown; one is undef. Neither sentinel names a definition.
  SmallVector<TargetInstrInfo::RegSubRegPair> ValueNumbers = {{}, {}};
  // Map SSA names, including subregister projections, to their canonical
  // numbers. Copies and splits can give one number several names.
  DenseMap<TargetInstrInfo::RegSubRegPair, ValueNumber> RegValueNumbers;
  // Find the existing whole value for an pair of REG_SEQUENCE arguments,
  // ordered sublo to subhi.
  DenseMap<std::pair<ValueNumber, ValueNumber>, ValueNumber>
      RegSequenceValueNumbers;
};

template <> struct DenseMapInfo<MOSValueNumbering::ValueNumber> {
  using ValueNumber = MOSValueNumbering::ValueNumber;
  static unsigned getHashValue(ValueNumber V) {
    return DenseMapInfo<unsigned>::getHashValue(V.ID);
  }
  static bool isEqual(ValueNumber A, ValueNumber B) { return A == B; }
};

// Transformations preserving this analysis must record the identities of new
// SSA registers. The result can otherwise be rebuilt from the current MIR;
// rebuilding may conservatively lose equivalences established during repair.
class MOSValueNumberingWrapperPass : public MachineFunctionPass {
public:
  static char ID;
  MOSValueNumberingWrapperPass();
  bool runOnMachineFunction(MachineFunction &MF) override;
  MOSValueNumbering &valueNumbers() { return *ValueNumbers; }
  void releaseMemory() override { ValueNumbers.reset(); }

  MachineFunctionProperties getRequiredProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;

private:
  std::optional<MOSValueNumbering> ValueNumbers;
};

} // namespace llvm

#endif // LLVM_LIB_TARGET_MOS_MOSVALUENUMBERING_H
