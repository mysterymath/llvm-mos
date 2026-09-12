//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Choose registers for SSA instructions using dynamic programming at
/// instruction boundaries. MOSImagRegAlloc supplies backing registers through
/// VirtRegMap. The DP state consists of A/X/Y/C/V contents and backing-validity
/// bits. A live, non-rematerializable value absent from hardware must have a
/// valid backing copy; imaginary contents can therefore be reconstructed from
/// the IR and VirtRegMap.
///
/// Each transition searches locally for implementations of one instruction:
/// prepare uses, choose defs, preserve live values, apply instruction effects,
/// and restore the backing invariant. Temporary imaginary copies and operand
/// choices stay within that search. Clobbers are forced effects between early
/// and ordinary definitions, with no extra instructions inserted between them.
///
/// Each boundary table maps a compact allocation state to the cheapest known
/// implementation, accumulated cost, and predecessor index. Code is emitted
/// only after selecting an entire block's allocations. Block exits establish
/// live values in backing registers before the terminators. Value numbers
/// identify equal contents independently of SSA names and backing assignments.
///
/// This initial implementation retains the input schedule and supports the
/// Imag8/Imag16 operations used by sieve. Equal states retain only their
/// cheapest predecessor; there is no beam pruning. Copy construction chooses
/// scratch locally, so this is not an exhaustive search of instruction
/// sequences. Actual spills, nonconstant flag materialization, and repairs of
/// pinned imaginary live ranges across block boundaries are not implemented.
/// Unsupported preservation is diagnosed.
///
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"
#include "MOSSubtarget.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/VirtRegMap.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <utility>
#include <variant>

using namespace llvm;

#define DEBUG_TYPE "mos-regalloc"
STATISTIC(NumStates, "Number of MOS register placement states retained");
STATISTIC(NumTransfers, "Number of MOS register repair instructions emitted");

namespace {

// Uniquely identifies a static value. getValueNumber resolves known copies to
// a canonical source SSA register and subregister index. Imag16 values are
// tracked as sublo and subhi; Imag8 and flags use index 0 (the whole register).
// A null register denotes unknown contents.
using ValueNumber = TargetInstrInfo::RegSubRegPair;

constexpr std::array<MCPhysReg, 5> HardwareRegs = {MOS::A, MOS::X, MOS::Y,
                                                   MOS::C, MOS::V};

// The DP key at an instruction boundary. Everything else needed to recover
// register contents comes from the live SSA registers and VirtRegMap at that
// boundary. ValidBackings is indexed by physical register, not by value number:
// equivalent SSA names can have different backing assignments.
struct AllocationState {
  std::array<ValueNumber, HardwareRegs.size()> Hardware = {};
  BitVector ValidBackings = BitVector(MOS::NUM_TARGET_REGS);

  bool operator==(const AllocationState &Other) const {
    return Hardware == Other.Hardware && ValidBackings == Other.ValidBackings;
  }
};

struct AllocationStateInfo {
  static unsigned getHashValue(const AllocationState &State) {
    unsigned Hash = DenseMapInfo<BitVector>::getHashValue(State.ValidBackings);
    for (ValueNumber V : State.Hardware)
      Hash = detail::combineHashValue(
          Hash, DenseMapInfo<ValueNumber>::getHashValue(V));
    return Hash;
  }
  static bool isEqual(const AllocationState &A, const AllocationState &B) {
    return A == B;
  }
};

// An ordered implementation of the original instruction. Copies and
// rematerializations insert MIR. OperandAssignment realizes the original MI
// in place and separates code emitted before it from code emitted after it.
struct RegisterCopy {
  unsigned Opcode;
  MCPhysReg Dst, Src;
};
struct Rematerialization {
  const MachineInstr *Definition;
  MCPhysReg Dst;
};
struct OperandAssignment {
  SmallVector<MCPhysReg> Registers;
};
using EmittedInstruction =
    std::variant<RegisterCopy, Rematerialization, OperandAssignment>;

// The path reaching an allocation: total cost so far, predecessor, and code
// for the last transition. Previous indexes the preceding point's table.
struct DPEntry {
  unsigned Cost;
  unsigned Previous;
  SmallVector<EmittedInstruction> Instructions;
};

// Each allocation has one cheapest known path reaching it. Previous indexes
// the preceding table in iteration order. Completed tables are immutable,
// so their iteration order stays fixed through successor search and emission.
using AllocationTable = DenseMap<AllocationState, DPEntry, AllocationStateInfo>;

// An instruction boundary: fixed facts from MIR and the alternatives found by
// the DP. MI is the instruction leading to this point. A null MI denotes the
// block entry or the restoration of backing registers before terminators.
struct ProgramPoint {
  explicit ProgramPoint(MachineInstr *MI) : MI(MI) {}

  MachineInstr *MI;
  // Virtual register indices, including PHI edge uses.
  SparseBitVector<> LiveRegs;
  // Physical registers; Imag16s use their Imag8s.
  BitVector FixedRegs;
  AllocationTable Allocations;
};

// Points stay in schedule order. Their original instructions remain intact
// until planning finishes, for rematerialization and deferred code emission.
struct BlockPlan {
  SmallVector<ProgramPoint> Points;
};

// Known contents of physical registers during instruction search. Imag16
// contents are stored under their Imag8 subregisters, so a byte overwrite
// affects only that byte. Missing entries have unknown contents.
class RegisterContents {
public:
  ValueNumber read(MCPhysReg R) const { return Contents.lookup(R); }
  void define(MCPhysReg R, ValueNumber V) {
    assert(R && V.Reg);
    Contents[R] = V;
  }
  void clobber(MCPhysReg R) { Contents.erase(R); }
  void copy(MCPhysReg Dst, MCPhysReg Src) { define(Dst, read(Src)); }
  bool hasCopy(ValueNumber V) const {
    return V.Reg && llvm::any_of(Contents, [=](const auto &Entry) {
             return Entry.second == V;
           });
  }
  bool hasHardwareCopy(ValueNumber V) const {
    return V.Reg && llvm::any_of(HardwareRegs, [this, V](MCPhysReg R) {
             return read(R) == V;
           });
  }
  SmallVector<MCPhysReg> copies(ValueNumber V) const {
    SmallVector<MCPhysReg> Regs;
    for (auto [R, Value] : Contents)
      if (Value == V)
        Regs.push_back(R);
    // Copy construction must not depend on hash table iteration order.
    llvm::sort(Regs);
    return Regs;
  }
  void forget(function_ref<bool(ValueNumber)> IsDead) {
    Contents.remove_if([&](const auto &Entry) { return IsDead(Entry.second); });
  }

private:
  SmallDenseMap<MCPhysReg, ValueNumber, 8> Contents;
};

// Working contents and generated code while constructing one instruction's
// implementation. Registers may contain temporary imaginary copies until the
// instruction search restores the backing invariant. Cost is local to this
// implementation, in estimated bytes; it excludes preceding instructions.
struct Implementation {
  Implementation() = default;
  explicit Implementation(RegisterContents Input)
      : Registers(std::move(Input)) {}

  RegisterContents Registers;
  SmallVector<EmittedInstruction> Instructions;
  unsigned Cost = 0;
};

class FunctionAllocator {
public:
  FunctionAllocator(MachineFunction &MF, const VirtRegMap &VRM,
                    LiveVariables &LV, const RegisterClassInfo &RCI);
  void run();

private:
  class BlockSearch;
  class InstructionSearch;
  class OperandChoices;
  class InstructionPlacement;

  struct PHIEdge {
    MachineBasicBlock *From, *To;
    SmallVector<std::pair<Register, Register>> Copies;
  };

  void collectConstraints();
  void analyzeLiveness();
  void planBlock(MachineBasicBlock &MBB);
  void emitEdges();
  SmallVector<PHIEdge> collectPHIEdges();
  void emitPHIEdge(const PHIEdge &Edge);
  void emitSolution(MachineBasicBlock &MBB, const BlockPlan &Block);
  void eraseVirtualInstructions();
  void recomputePhysicalLiveness();

  BitVector fixedRegisters(const LivePhysRegs &LiveRegs) const;

  bool isRematerialized(const MachineInstr &MI) const;
  // Visit each live byte's required backing register and static value. Use
  // the live range's assignment, even when its value number names an ancestor.
  void forEachBacking(const SparseBitVector<> &LiveRegs,
                      function_ref<void(MCPhysReg, ValueNumber)> Visit) const;
  AllocationState getAllocationState(const RegisterContents &Registers,
                                     const SparseBitVector<> &LiveRegs) const;
  RegisterContents getRegisterContents(const AllocationState &State,
                                       const SparseBitVector<> &LiveRegs) const;
  bool restoreBackingInvariant(Implementation &Plan,
                               const SparseBitVector<> &LiveRegs,
                               BitVector Locked, bool CanInsert);
  bool restoreBackingRegisters(Implementation &Plan,
                               const SparseBitVector<> &LiveRegs,
                               const SparseBitVector<> &Preserve,
                               BitVector Locked);

  // Locked registers may be read but not changed by inserted instructions.
  // Forbidden additionally protects registers about to be clobbered: evacuation
  // must not preserve a value in another member of that same clobber set.
  bool placeValue(ValueNumber V, MCPhysReg Dst, Implementation &Plan,
                  const SparseBitVector<> &LiveRegs, BitVector Locked,
                  const BitVector *Forbidden = nullptr);
  bool rematerialize(ValueNumber V, MCPhysReg Reg, Implementation &Plan,
                     const SparseBitVector<> &LiveRegs, BitVector Locked,
                     const BitVector &Forbidden);
  bool evacuate(MCPhysReg Reg, Implementation &Plan,
                const SparseBitVector<> &LiveRegs, BitVector Locked,
                const BitVector &Forbidden);
  bool copyRegister(MCPhysReg Dst, MCPhysReg Src, Implementation &Plan);
  void emitInstructions(MachineBasicBlock &MBB, MachineBasicBlock::iterator At,
                        const DebugLoc &DL,
                        ArrayRef<EmittedInstruction> Instructions,
                        MachineInstr *MI = nullptr);
  unsigned instructionCost(const MachineInstr &MI,
                           ArrayRef<MCPhysReg> Operands) const;

  SmallVector<MCPhysReg> destinations(Register R) const;
  SmallVector<MCPhysReg> copies(Register R, const RegisterContents &S) const;
  bool hasLiveValues(const RegisterContents &S,
                     const SparseBitVector<> &LiveRegs) const;
  bool isLiveValue(ValueNumber V, const SparseBitVector<> &LiveRegs) const;
  bool sameValue(Register A, Register B) const;
  SparseBitVector<> liveOuts(MachineBasicBlock &MBB) const;

  // Select each independently tracked value: both bytes of an Imag16, or
  // the whole register (index 0) for an Imag8 or flag.
  ArrayRef<unsigned> subRegIndices(Register R) const;
  ValueNumber getValueNumber(Register R, unsigned SubReg = 0) const;
  // Storage modeled by the search: A/X/Y/C/V and Imag8. An Imag16 denotes
  // its two Imag8s; implicit hardware aliases are not separate storage.
  SmallVector<MCPhysReg, 2> registerParts(MCPhysReg R) const;
  // Backing storage belongs to a live range, not to its value number.
  // Return zero if the live range has no backing assignment.
  MCPhysReg getBackingRegister(Register R, unsigned SubReg = 0) const;
  const MachineInstr *rematerialization(ValueNumber V) const;
  [[noreturn]] void fail(const Twine &Reason,
                         const MachineInstr *MI = nullptr) const;

  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  // VirtRegMap records backing storage here, independently of operand classes.
  // This pass realizes the operands itself and consumes the map's assignments.
  const VirtRegMap &VRM;
  LiveVariables &LV;
  const RegisterClassInfo &RCI;
  DenseMap<Register, SmallVector<MCPhysReg>> Constraints;
  DenseMap<MachineBasicBlock *, BlockPlan> Blocks;
};

// The DP visits instruction boundaries in schedule order. Only completed
// instruction implementations, with the backing invariant restored, enter its
// tables. All operand choices and temporary contents belong to the local
// search.
class FunctionAllocator::BlockSearch {
public:
  BlockSearch(FunctionAllocator &Allocator, MachineBasicBlock &MBB);
  void run();

private:
  void initialize();
  void advance(const ProgramPoint &Before, ProgramPoint &After);
  bool restoreLiveOuts(Implementation &Plan, const ProgramPoint &Before);

  FunctionAllocator &Allocator;
  MachineBasicBlock &MBB;
  BlockPlan &Block;
};

// Search implementations of one instruction. The uses-to-defs bindings
// remain on the search stack; the outer DP sees only the compact outgoing
// allocation and the code/cost of each completed implementation.
class FunctionAllocator::InstructionSearch {
public:
  InstructionSearch(FunctionAllocator &Allocator, const ProgramPoint &Before,
                    const ProgramPoint &After, const RegisterContents &Incoming,
                    function_ref<void(Implementation)> Accept);
  void run();

private:
  void prepareUses(ArrayRef<MCPhysReg> Uses);
  void define(const Implementation &Prepared, ArrayRef<MCPhysReg> Operands);
  void finish(Implementation Plan);

  FunctionAllocator &Allocator;
  MachineInstr &MI;
  const RegisterContents &Incoming;
  const ProgramPoint &Before, &After;
  function_ref<void(Implementation)> Accept;
};

// Enumerate input or output operand assignments for the instruction-local
// search. Output choices retain the selected inputs to check ties and overlap.
class FunctionAllocator::OperandChoices {
public:
  OperandChoices(FunctionAllocator &Allocator, MachineInstr &MI,
                 const RegisterContents &Incoming, bool Defs,
                 ArrayRef<MCPhysReg> Uses,
                 function_ref<void(ArrayRef<MCPhysReg>)> Accept);
  void run();

private:
  void buildDomain(unsigned OpIdx);
  void addCopies(unsigned OpIdx, Register R);
  void addDestination(unsigned OpIdx, MCPhysReg R);
  bool isLegal(unsigned OpIdx, MCPhysReg R) const;
  bool isTiedUseAlias(const MachineOperand &Def,
                      const MachineOperand &Use) const;
  void search(unsigned OpIdx);

  FunctionAllocator &Allocator;
  MachineInstr &MI;
  const RegisterContents &Incoming;
  bool Defs;
  function_ref<void(ArrayRef<MCPhysReg>)> Accept;
  // Domains depend on the incoming contents, not on earlier operand choices.
  // Compute them once for this enumeration; only Assignment changes in search.
  SmallVector<SmallVector<MCPhysReg>> Domains;
  SmallVector<MCPhysReg> Assignment;
};

// Realize one operand assignment. All generated preservation and
// input preparation executes before MI. Inside MI, applying early defs,
// clobbers, and ordinary defs is forced and emits no extra instructions.
class FunctionAllocator::InstructionPlacement {
public:
  InstructionPlacement(FunctionAllocator &Allocator, const ProgramPoint &Before,
                       const ProgramPoint &After, Implementation &Plan,
                       ArrayRef<MCPhysReg> Operands);
  bool prepareInputs();
  bool execute();

private:
  bool repairCopy();
  bool repairRegSequence();
  bool prepareUses(bool InGPRs);
  void protectUses();
  bool preserveLiveThroughValues();
  bool applyInstruction();
  bool define(const MachineOperand &MO);

  FunctionAllocator &Allocator;
  MachineInstr &MI;
  Implementation &Plan;
  ArrayRef<MCPhysReg> Operands;
  const ProgramPoint &Before, &After;
  BitVector Locked;
};

FunctionAllocator::FunctionAllocator(MachineFunction &MF, const VirtRegMap &VRM,
                                     LiveVariables &LV,
                                     const RegisterClassInfo &RCI)
    : MF(MF), MRI(MF.getRegInfo()), TII(*MF.getSubtarget().getInstrInfo()),
      TRI(*MF.getSubtarget().getRegisterInfo()), VRM(VRM), LV(LV), RCI(RCI) {}

void FunctionAllocator::run() {
  collectConstraints();
  analyzeLiveness();
  // Keep SSA definitions intact until every block and edge has been planned;
  // rematerialization recipes and register-class queries refer to the original
  // MIR.
  for (MachineBasicBlock &MBB : MF)
    planBlock(MBB);
  emitEdges();
  for (MachineBasicBlock &MBB : MF)
    if (auto I = Blocks.find(&MBB); I != Blocks.end())
      emitSolution(MBB, I->second);
  eraseVirtualInstructions();
  recomputePhysicalLiveness();
}

void FunctionAllocator::collectConstraints() {
  // A physical COPY exposes a singleton constraint shared by all whole-value
  // aliases. VirtRegMap already records their common source.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isFullCopy() && MI.getOperand(0).getReg().isPhysical() &&
          MI.getOperand(1).getReg().isVirtual()) {
        auto &Regs = Constraints[VRM.getOriginal(MI.getOperand(1).getReg())];
        MCPhysReg R = MI.getOperand(0).getReg();
        if (!llvm::is_contained(Regs, R))
          Regs.push_back(R);
      }
}

void FunctionAllocator::analyzeLiveness() {
  for (MachineBasicBlock &MBB : MF) {
    auto &Points = Blocks[&MBB].Points;
    Points.emplace_back(nullptr); // Entry: all incoming backings are valid.
    bool HasTerminators = false;
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      if (MI.isTerminator() && !HasTerminators) {
        Points.emplace_back(nullptr); // Restore before any branch can execute.
        HasTerminators = true;
      }
      Points.emplace_back(&MI);
    }
    if (!HasTerminators)
      Points.emplace_back(nullptr);

    SparseBitVector<> LiveRegs = liveOuts(MBB);
    LivePhysRegs Phys(TRI);
    Phys.addLiveOutsNoPristines(MBB);
    for (ProgramPoint &Point : reverse(Points)) {
      Point.LiveRegs = LiveRegs;
      Point.FixedRegs = fixedRegisters(Phys);
      if (!Point.MI)
        continue;
      MachineInstr &MI = *Point.MI;
      for (const MachineOperand &MO : MI.all_defs())
        if (MO.getReg().isVirtual())
          LiveRegs.reset(MO.getReg().virtRegIndex());
      // PHI inputs are uses on predecessor edges, already included in liveOuts.
      if (!MI.isPHI())
        for (const MachineOperand &MO : MI.all_uses())
          if (MO.getReg().isVirtual() && !MO.isUndef())
            LiveRegs.set(MO.getReg().virtRegIndex());
      Phys.stepBackward(MI);
    }
  }
}

BitVector
FunctionAllocator::fixedRegisters(const LivePhysRegs &LiveRegs) const {
  // Compare/branch pseudos keep N/Z internal until late optimization.
  assert(!LiveRegs.contains(MOS::N) && !LiveRegs.contains(MOS::Z) &&
         "N/Z must not be live during MOS register allocation");
  BitVector FixedRegs(MOS::NUM_TARGET_REGS);
  for (MCPhysReg R : LiveRegs)
    for (MCPhysReg Part : registerParts(R))
      FixedRegs.set(Part);
  return FixedRegs;
}

void FunctionAllocator::planBlock(MachineBasicBlock &MBB) {
  BlockSearch(*this, MBB).run();
}

FunctionAllocator::BlockSearch::BlockSearch(FunctionAllocator &Allocator,
                                            MachineBasicBlock &MBB)
    : Allocator(Allocator), MBB(MBB),
      Block(Allocator.Blocks.find(&MBB)->second) {}

void FunctionAllocator::BlockSearch::run() {
  initialize();
  for (unsigned I = 1; I < Block.Points.size(); ++I)
    advance(Block.Points[I - 1], Block.Points[I]);
}

void FunctionAllocator::BlockSearch::initialize() {
  ProgramPoint &Entry = Block.Points.front();
  RegisterContents Registers;
  Allocator.forEachBacking(Entry.LiveRegs, [&](MCPhysReg R, ValueNumber V) {
    if (Entry.FixedRegs[R])
      Allocator.fail("pinned physical live-in overlaps a backing register");
    assert((!Registers.read(R).Reg || Registers.read(R) == V) &&
           "backing assignments interfere");
    Registers.define(R, V);
  });
  Entry.Allocations.try_emplace(
      Allocator.getAllocationState(Registers, Entry.LiveRegs),
      DPEntry{0, ~0u, {}});
}

void FunctionAllocator::BlockSearch::advance(const ProgramPoint &Before,
                                             ProgramPoint &After) {
  unsigned Previous = 0;
  for (const auto &Entry : Before.Allocations) {
    RegisterContents Incoming =
        Allocator.getRegisterContents(Entry.first, Before.LiveRegs);
    auto Retain = [&](Implementation Plan) {
      AllocationState State =
          Allocator.getAllocationState(Plan.Registers, After.LiveRegs);
      unsigned Cost = Entry.second.Cost + Plan.Cost;
      auto [I, Inserted] = After.Allocations.try_emplace(std::move(State));
      // Retain the first implementation on cost ties.
      if (Inserted || Cost < I->second.Cost)
        I->second = DPEntry{Cost, Previous, std::move(Plan.Instructions)};
    };
    if (After.MI) {
      InstructionSearch(Allocator, Before, After, Incoming, Retain).run();
    } else {
      Implementation Plan(std::move(Incoming));
      if (restoreLiveOuts(Plan, Before))
        Retain(std::move(Plan));
    }
    ++Previous;
  }
  if (After.Allocations.empty())
    Allocator.fail(After.MI ? "no supported placement continuation"
                            : "cannot restore live values to backing registers",
                   After.MI);
  NumStates += After.Allocations.size();
  LLVM_DEBUG(dbgs() << "bb." << MBB.getNumber() << ": "
                    << After.Allocations.size() << " states\n");
}

bool FunctionAllocator::BlockSearch::restoreLiveOuts(
    Implementation &Plan, const ProgramPoint &Before) {
  return Allocator.restoreBackingRegisters(Plan, Block.Points.back().LiveRegs,
                                           Before.LiveRegs, Before.FixedRegs) &&
         Allocator.restoreBackingInvariant(Plan, Before.LiveRegs,
                                           Before.FixedRegs, true);
}

bool FunctionAllocator::isRematerialized(const MachineInstr &MI) const {
  return (TII.isTriviallyReMaterializable(MI) &&
          MI.getOperand(0).getReg().isVirtual()) ||
         (MI.isCopy() && MI.getOperand(0).getReg().isVirtual() &&
          llvm::all_of(subRegIndices(MI.getOperand(0).getReg()),
                       [&](unsigned SubReg) {
                         return rematerialization(
                             getValueNumber(MI.getOperand(0).getReg(), SubReg));
                       }));
}

FunctionAllocator::InstructionSearch::InstructionSearch(
    FunctionAllocator &Allocator, const ProgramPoint &Before,
    const ProgramPoint &After, const RegisterContents &Incoming,
    function_ref<void(Implementation)> Accept)
    : Allocator(Allocator), MI(*After.MI), Incoming(Incoming), Before(Before),
      After(After), Accept(Accept) {}

void FunctionAllocator::InstructionSearch::run() {
  if (MI.isPHI()) {
    Implementation Plan(Incoming);
    Register R = MI.getOperand(0).getReg();
    if (!Allocator.MRI.use_nodbg_empty(R))
      for (unsigned SubReg : Allocator.subRegIndices(R))
        Plan.Registers.define(Allocator.getBackingRegister(R, SubReg),
                              Allocator.getValueNumber(R, SubReg));
    finish(std::move(Plan));
    return;
  }
  if (Allocator.isRematerialized(MI)) {
    finish(Implementation(Incoming));
    return;
  }
  OperandChoices(Allocator, MI, Incoming, false, {},
                 [&](ArrayRef<MCPhysReg> Uses) { prepareUses(Uses); })
      .run();
}

void FunctionAllocator::InstructionSearch::prepareUses(
    ArrayRef<MCPhysReg> Uses) {
  Implementation Prepared(Incoming);
  if (!InstructionPlacement(Allocator, Before, After, Prepared, Uses)
           .prepareInputs())
    return;
  OperandChoices(
      Allocator, MI, Prepared.Registers, true, Uses,
      [&](ArrayRef<MCPhysReg> Operands) { define(Prepared, Operands); })
      .run();
}

void FunctionAllocator::InstructionSearch::define(
    const Implementation &Prepared, ArrayRef<MCPhysReg> Operands) {
  Implementation Plan = Prepared;
  if (InstructionPlacement(Allocator, Before, After, Plan, Operands).execute())
    finish(std::move(Plan));
}

void FunctionAllocator::InstructionSearch::finish(Implementation Plan) {
  if (Allocator.restoreBackingInvariant(Plan, After.LiveRegs, After.FixedRegs,
                                        !MI.isTerminator()))
    Accept(std::move(Plan));
}

FunctionAllocator::OperandChoices::OperandChoices(
    FunctionAllocator &Allocator, MachineInstr &MI,
    const RegisterContents &Incoming, bool Defs, ArrayRef<MCPhysReg> Uses,
    function_ref<void(ArrayRef<MCPhysReg>)> Accept)
    : Allocator(Allocator), MI(MI), Incoming(Incoming), Defs(Defs),
      Accept(Accept), Domains(MI.getNumOperands()), Assignment(Uses) {
  Assignment.resize(MI.getNumOperands());
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.getReg().isPhysical())
      Assignment[I] = MO.getReg();
  }
}

void FunctionAllocator::OperandChoices::run() {
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.getReg().isVirtual() && MO.isDef() == Defs)
      buildDomain(I);
  }
  search(0);
}

void FunctionAllocator::OperandChoices::buildDomain(unsigned I) {
  const MachineOperand &MO = MI.getOperand(I);
  if (MO.getSubReg())
    Allocator.fail("subregister operands are not implemented", &MI);
  Register R = MO.getReg();
  if ((MI.isCopy() && I == 1) ||
      (MI.getOpcode() == TargetOpcode::REG_SEQUENCE && I != 0)) {
    // Transfers read from any available copy; these operands are not emitted.
    Domains[I].push_back(0);
    return;
  }
  Domains[I] = Allocator.destinations(R);
  addCopies(I, R);
  // An existing copy can satisfy an operand in place, even after evacuation.
  // COPY and tied definitions may also reuse their input's current location.
  if (MI.isCopy() && I == 0 && MI.getOperand(1).getReg().isVirtual())
    addCopies(I, MI.getOperand(1).getReg());
  if (MO.isDef() && MO.isTied()) {
    Register S = MI.getOperand(MI.findTiedOperandIdx(I)).getReg();
    if (S.isVirtual()) {
      addCopies(I, S);
      MCPhysReg H = Allocator.VRM.getPhys(S);
      if (H)
        addDestination(I, H);
    }
  }
  if (MI.isCopy() && MI.getOperand(1).getReg().isPhysical())
    Domains[I].push_back(MI.getOperand(1).getReg());
  if (MO.isUse() && MO.isTied()) {
    Register D = MI.getOperand(MI.findTiedOperandIdx(I)).getReg();
    if (D.isVirtual())
      for (MCPhysReg H : Allocator.destinations(D))
        addDestination(I, H);
  }
  if (!MI.isCopy() && MI.getOpcode() != TargetOpcode::REG_SEQUENCE) {
    const TargetRegisterClass *RC =
        MI.getRegClassConstraint(I, &Allocator.TII, &Allocator.TRI);
    if (!RC)
      RC = Allocator.MRI.getRegClass(R);
    llvm::erase_if(Domains[I], [&](MCPhysReg H) { return !RC->contains(H); });
  }
  if (Domains[I].empty())
    Allocator.fail("empty operand placement domain", &MI);
}

void FunctionAllocator::OperandChoices::addCopies(unsigned OpIdx, Register R) {
  for (MCPhysReg Location : Allocator.copies(R, Incoming))
    addDestination(OpIdx, Location);
}

void FunctionAllocator::OperandChoices::addDestination(unsigned OpIdx,
                                                       MCPhysReg R) {
  if (!llvm::is_contained(Domains[OpIdx], R))
    Domains[OpIdx].push_back(R);
}

bool FunctionAllocator::OperandChoices::isLegal(unsigned I, MCPhysReg R) const {
  const MachineOperand &MO = MI.getOperand(I);
  if (MI.isCopy() || MI.getOpcode() == TargetOpcode::REG_SEQUENCE ||
      !MO.isReg())
    return true;
  for (unsigned J = 0; J < MI.getNumOperands(); ++J) {
    if (J == I || !Assignment[J])
      continue;
    const MachineOperand &Other = MI.getOperand(J);
    if (!Other.isReg())
      continue;
    MCPhysReg OtherReg = Assignment[J];
    if (MO.isTied() && MI.findTiedOperandIdx(I) == J) {
      if (R != OtherReg)
        return false;
      continue;
    }
    if (!R || !OtherReg || !Allocator.TRI.regsOverlap(R, OtherReg) ||
        MO.isUndef() || Other.isUndef())
      continue;
    if (MO.isDef() && Other.isDef())
      return false;
    if (MO.isUse() && Other.isUse() &&
        (R != OtherReg || !Allocator.sameValue(MO.getReg(), Other.getReg())))
      return false;
    if ((MO.isEarlyClobber() && Other.isUse() && !isTiedUseAlias(MO, Other)) ||
        (Other.isEarlyClobber() && MO.isUse() && !isTiedUseAlias(Other, MO)))
      return false;
  }
  return true;
}

bool FunctionAllocator::OperandChoices::isTiedUseAlias(
    const MachineOperand &Def, const MachineOperand &Use) const {
  // LiveVariables can add implicit subregister kills beside a tied physical
  // use. They describe the same input, not independent early-clobber conflicts.
  if (!Def.isTied() || !Use.isImplicit() || !Use.isKill() ||
      !Use.getReg().isPhysical())
    return false;
  Register TiedReg =
      MI.getOperand(MI.findTiedOperandIdx(Def.getOperandNo())).getReg();
  return TiedReg.isPhysical() &&
         Allocator.TRI.isSubRegisterEq(TiedReg, Use.getReg());
}

void FunctionAllocator::OperandChoices::search(unsigned I) {
  if (I == MI.getNumOperands()) {
    Accept(Assignment);
    return;
  }
  if (Domains[I].empty()) {
    search(I + 1);
    return;
  }
  for (MCPhysReg R : Domains[I]) {
    if (!isLegal(I, R))
      continue;
    Assignment[I] = R;
    search(I + 1);
  }
  Assignment[I] = 0;
}

FunctionAllocator::InstructionPlacement::InstructionPlacement(
    FunctionAllocator &Allocator, const ProgramPoint &Before,
    const ProgramPoint &After, Implementation &Plan,
    ArrayRef<MCPhysReg> Operands)
    : Allocator(Allocator), MI(*After.MI), Plan(Plan), Operands(Operands),
      Before(Before), After(After), Locked(Before.FixedRegs) {}

bool FunctionAllocator::InstructionPlacement::prepareInputs() {
  // COPY and REG_SEQUENCE are realized after their destination is selected.
  // Their sources may come from any available copy.
  if (MI.isCopy() || MI.getOpcode() == TargetOpcode::REG_SEQUENCE)
    return true;
  // Memory and flag operands can need hardware scratch; prepare those first.
  return prepareUses(false) && prepareUses(true);
}

void FunctionAllocator::InstructionPlacement::protectUses() {
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.isUse() && !MO.isUndef())
      for (MCPhysReg R : Allocator.registerParts(Operands[I]))
        Locked.set(R);
  }
}

bool FunctionAllocator::InstructionPlacement::execute() {
  if (MI.isCopy() && (MI.getOperand(0).getReg().isVirtual() ||
                      MI.getOperand(1).getReg().isVirtual()))
    return repairCopy();
  if (MI.getOpcode() == TargetOpcode::REG_SEQUENCE)
    return repairRegSequence();
  protectUses();
  if (!preserveLiveThroughValues() || !applyInstruction())
    return false;
  Plan.Cost += Allocator.instructionCost(MI, Operands);
  Plan.Instructions.push_back(
      OperandAssignment{SmallVector<MCPhysReg>(Operands)});
  return Allocator.hasLiveValues(Plan.Registers, After.LiveRegs);
}

bool FunctionAllocator::InstructionPlacement::repairCopy() {
  Register D = MI.getOperand(0).getReg(), S = MI.getOperand(1).getReg();
  auto Dst = Allocator.registerParts(Operands[0]);
  if (Dst.empty())
    return false;
  if (S.isPhysical()) {
    auto Src = Allocator.registerParts(S);
    if (Src.size() != Dst.size() || !D.isVirtual())
      return false;
    // Capture the existing physical bits without emitting a transfer when the
    // chosen destination is the source. The fixed live range protects them.
    for (auto [SubReg, Reg] :
         llvm::zip_equal(Allocator.subRegIndices(D), Src)) {
      if (!Allocator.evacuate(Reg, Plan, Before.LiveRegs, Locked,
                              BitVector(MOS::NUM_TARGET_REGS)))
        return false;
      Plan.Registers.define(Reg, Allocator.getValueNumber(D, SubReg));
    }
    for (auto [SubReg, Reg] : llvm::zip_equal(Allocator.subRegIndices(D), Dst))
      if (!Allocator.placeValue(Allocator.getValueNumber(D, SubReg), Reg, Plan,
                                Before.LiveRegs, Locked))
        return false;
  } else {
    for (auto [SubReg, Reg] :
         llvm::zip_equal(Allocator.subRegIndices(S), Dst)) {
      if (!Allocator.placeValue(Allocator.getValueNumber(S, SubReg), Reg, Plan,
                                Before.LiveRegs, Locked))
        return false;
      Locked.set(Reg);
    }
    if (D.isVirtual() && !Allocator.sameValue(D, S))
      Allocator.fail("COPY lacks whole-value ancestry", &MI);
  }
  return Allocator.hasLiveValues(Plan.Registers, After.LiveRegs);
}

bool FunctionAllocator::InstructionPlacement::repairRegSequence() {
  auto Dst = Allocator.registerParts(Operands[0]);
  if (Dst.size() != 2)
    return false;
  for (unsigned I = 1; I < MI.getNumOperands(); I += 2) {
    unsigned SubReg = MI.getOperand(I + 1).getImm();
    MCPhysReg Reg = Allocator.TRI.getSubReg(Operands[0], SubReg);
    Register S = MI.getOperand(I).getReg();
    if (!Allocator.placeValue(Allocator.getValueNumber(S), Reg, Plan,
                              Before.LiveRegs, Locked))
      return false;
    Locked.set(Reg);
  }
  return Allocator.hasLiveValues(Plan.Registers, After.LiveRegs);
}

bool FunctionAllocator::InstructionPlacement::prepareUses(bool InGPRs) {
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse() || MO.isUndef() || !MO.getReg().isVirtual())
      continue;
    auto Parts = Allocator.registerParts(Operands[I]);
    if (Parts.empty() || MOS::GPRRegClass.contains(Parts.front()) != InGPRs)
      continue;
    for (auto [SubReg, Reg] :
         llvm::zip_equal(Allocator.subRegIndices(MO.getReg()), Parts)) {
      if (!Allocator.placeValue(Allocator.getValueNumber(MO.getReg(), SubReg),
                                Reg, Plan, Before.LiveRegs, Locked))
        return false;
      Locked.set(Reg);
    }
  }
  return true;
}

bool FunctionAllocator::InstructionPlacement::preserveLiveThroughValues() {
  // Evacuations must avoid every impending clobber, including destinations of
  // later operands. The small physical mask is shared by all evacuations.
  BitVector Clobbered(MOS::NUM_TARGET_REGS);
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.isDef())
      for (MCPhysReg Reg : Allocator.registerParts(Operands[I]))
        Clobbered.set(Reg);
    if (MO.isRegMask())
      for (MCPhysReg Reg = 1; Reg < MOS::NUM_TARGET_REGS; ++Reg)
        if (MO.clobbersPhysReg(Reg))
          Clobbered.set(Reg);
  }
  for (int Reg : Clobbered.set_bits())
    if (!Allocator.evacuate(Reg, Plan, After.LiveRegs, Locked, Clobbered))
      return false;
  return true;
}

bool FunctionAllocator::InstructionPlacement::applyInstruction() {
  // Repairs execute before MI, so preservation above protects every input.
  // Now simulate MI's effects. Killed inputs need no explicit erasure: their
  // copies remain usable until overwritten or forgotten at the end of the step.
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isEarlyClobber() && !define(MO))
      return false;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      for (MCPhysReg Reg = 1; Reg < MOS::NUM_TARGET_REGS; ++Reg)
        if (MO.clobbersPhysReg(Reg))
          Plan.Registers.clobber(Reg);
  for (const MachineOperand &MO : MI.all_defs())
    if (!MO.isEarlyClobber() && !define(MO))
      return false;
  return true;
}

bool FunctionAllocator::InstructionPlacement::define(const MachineOperand &MO) {
  auto Parts = Allocator.registerParts(Operands[MO.getOperandNo()]);
  for (auto [SubReg, Reg] :
       llvm::zip(Allocator.subRegIndices(MO.getReg()), Parts)) {
    if (MO.getReg().isPhysical()) {
      // Fixed liveness protects this result; a later COPY assigns its value
      // number when it captures the physical contents into an SSA register.
      Plan.Registers.clobber(Reg);
      continue;
    }
    bool TiedPhysicalUse =
        MO.isTied() && MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo()))
                           .getReg()
                           .isPhysical();
    if (After.FixedRegs[Reg] ||
        (MO.isEarlyClobber() && Before.FixedRegs[Reg] && !TiedPhysicalUse))
      return false;
    Plan.Registers.define(Reg, Allocator.getValueNumber(MO.getReg(), SubReg));
  }
  return true;
}

void FunctionAllocator::forEachBacking(
    const SparseBitVector<> &LiveRegs,
    function_ref<void(MCPhysReg, ValueNumber)> Visit) const {
  for (unsigned I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned SubReg : subRegIndices(R)) {
      ValueNumber V = getValueNumber(R, SubReg);
      if (rematerialization(V))
        continue;
      MCPhysReg Backing = getBackingRegister(R, SubReg);
      if (!Backing)
        fail("live value has no backing register");
      Visit(Backing, V);
    }
  }
}

AllocationState
FunctionAllocator::getAllocationState(const RegisterContents &Registers,
                                      const SparseBitVector<> &LiveRegs) const {
  AllocationState State;
  for (auto [I, R] : llvm::enumerate(HardwareRegs))
    State.Hardware[I] = Registers.read(R);
  forEachBacking(LiveRegs, [&](MCPhysReg R, ValueNumber V) {
    bool Valid = Registers.read(R) == V;
    assert((Valid || Registers.hasHardwareCopy(V)) &&
           "live value must be in hardware or its backing register");
    State.ValidBackings[R] = Valid;
  });
  return State;
}

RegisterContents FunctionAllocator::getRegisterContents(
    const AllocationState &State, const SparseBitVector<> &LiveRegs) const {
  RegisterContents Registers;
  for (auto [I, R] : llvm::enumerate(HardwareRegs))
    if (State.Hardware[I].Reg)
      Registers.define(R, State.Hardware[I]);
  forEachBacking(LiveRegs, [&](MCPhysReg R, ValueNumber V) {
    if (State.ValidBackings[R]) {
      assert((!Registers.read(R).Reg || Registers.read(R) == V) &&
             "backing assignments interfere");
      Registers.define(R, V);
    }
  });
  return Registers;
}

bool FunctionAllocator::restoreBackingInvariant(
    Implementation &Plan, const SparseBitVector<> &LiveRegs, BitVector Locked,
    bool CanInsert) {
  if (!hasLiveValues(Plan.Registers, LiveRegs))
    return false;
  // A repair can displace a value into arbitrary imaginary scratch. Before
  // retaining the result, give every live lane either its backing copy or a
  // hardware copy. Protect established backing copies while doing so.
  while (true) {
    MCPhysReg Missing = 0;
    ValueNumber V;
    forEachBacking(LiveRegs, [&](MCPhysReg R, ValueNumber Required) {
      if (Plan.Registers.read(R) == Required)
        Locked.set(R);
      else if (!Plan.Registers.hasHardwareCopy(Required) && R > Missing) {
        Missing = R;
        V = Required;
      }
    });
    if (!Missing)
      break;
    // In particular, don't insert instructions after a branch to repair its
    // outgoing state. Its inputs were restored before the terminators.
    if (!CanInsert)
      return false;
    // Either destination establishes the invariant. Keeping a hardware copy
    // can avoid an unnecessary store, and also works when backing is pinned.
    MCPhysReg BestReg = 0;
    Implementation Best = Plan;
    if (placeValue(V, Missing, Best, LiveRegs, Locked))
      BestReg = Missing;
    for (MCPhysReg L : HardwareRegs) {
      Implementation Trial = Plan;
      if (placeValue(V, L, Trial, LiveRegs, Locked) &&
          (!BestReg || Trial.Cost < Best.Cost)) {
        BestReg = L;
        Best = std::move(Trial);
      }
    }
    if (!BestReg)
      return false;
    Plan = std::move(Best);
    Locked.set(BestReg);
  }
  Plan.Registers.forget(
      [&](ValueNumber V) { return !isLiveValue(V, LiveRegs); });
  return true;
}

bool FunctionAllocator::restoreBackingRegisters(
    Implementation &Plan, const SparseBitVector<> &LiveRegs,
    const SparseBitVector<> &Preserve, BitVector Locked) {
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned SubReg : subRegIndices(R)) {
      ValueNumber A = getValueNumber(R, SubReg);
      if (rematerialization(A))
        continue;
      MCPhysReg H = getBackingRegister(R, SubReg);
      if (!H || !placeValue(A, H, Plan, Preserve, Locked))
        return false;
      Locked.set(H);
    }
  }
  return true;
}

bool FunctionAllocator::placeValue(ValueNumber V, MCPhysReg Dst,
                                   Implementation &Plan,
                                   const SparseBitVector<> &LiveRegs,
                                   BitVector Locked,
                                   const BitVector *Forbidden) {
  if (Plan.Registers.read(Dst) == V)
    return true;
  if (Locked[Dst])
    return false;
  BitVector Avoid = Forbidden ? *Forbidden : BitVector(MOS::NUM_TARGET_REGS);
  Implementation Best;
  Best.Cost = ~0u;
  auto Consider = [&](Implementation Trial) {
    if (Trial.Cost < Best.Cost)
      Best = std::move(Trial);
  };
  Implementation Remat = Plan;
  if (rematerialize(V, Dst, Remat, LiveRegs, Locked, Avoid))
    Consider(std::move(Remat));

  if (!evacuate(Dst, Plan, LiveRegs, Locked, Avoid)) {
    if (Best.Cost == ~0u)
      return false;
    Plan = std::move(Best);
    return true;
  }
  Locked.set(Dst);
  for (MCPhysReg Src : Plan.Registers.copies(V)) {
    Implementation Trial = Plan;
    if (copyRegister(Dst, Src, Trial))
      Consider(std::move(Trial));
  }

  // A byte of a rematerializable Imag16 may be wanted in hardware. Materialize
  // the Imag16 in imaginary storage first, then use the ordinary transfer path.
  if (!Plan.Registers.hasCopy(V) && subRegIndices(V.Reg).size() == 2 &&
      rematerialization(V)) {
    for (MCPhysReg Imag16 : destinations(V.Reg)) {
      MCPhysReg Byte = TRI.getSubReg(Imag16, V.SubReg);
      if (!Byte)
        continue;
      Implementation Trial = Plan;
      if (!rematerialize(V, Byte, Trial, LiveRegs, Locked, Avoid))
        continue;
      BitVector Unlocked = Locked;
      Unlocked.reset(Dst);
      if (placeValue(V, Dst, Trial, LiveRegs, Unlocked, &Avoid))
        Consider(std::move(Trial));
    }
  }

  // Memory-to-memory and X/Y transfers need hardware scratch. Reuse the same
  // placement logic to obtain the value there, including rematerialization.
  // Each recursive step locks another register, bounding scratch construction.
  for (MCPhysReg Scratch : {MOS::A, MOS::X, MOS::Y}) {
    if (Locked[Scratch])
      continue;
    Implementation Trial = Plan;
    if (placeValue(V, Scratch, Trial, LiveRegs, Locked, &Avoid) &&
        copyRegister(Dst, Scratch, Trial))
      Consider(std::move(Trial));
  }
  if (Best.Cost == ~0u)
    return false;
  Plan = std::move(Best);
  return true;
}

// Rematerialize the complete definition, even when only one byte was requested.
// The other byte is a clobber during preparation and an available copy
// afterward.
bool FunctionAllocator::rematerialize(ValueNumber V, MCPhysReg Reg,
                                      Implementation &Plan,
                                      const SparseBitVector<> &LiveRegs,
                                      BitVector Locked,
                                      const BitVector &Forbidden) {
  const MachineInstr *Def = rematerialization(V);
  if (!Def)
    return false;
  Register Source = V.Reg;
  MCPhysReg Dst = Reg;
  if (subRegIndices(Source).size() == 2) {
    Dst = TRI.getMatchingSuperReg(Dst, V.SubReg, &MOS::Imag16RegClass);
    if (!Dst)
      return false;
  }
  const TargetRegisterClass *RC = Def->getRegClassConstraint(0, &TII, &TRI);
  if (!RC)
    RC = MRI.getRegClass(Source);
  if (!RC->contains(Dst))
    return false;

  auto Parts = registerParts(Dst);
  if (Parts.size() != subRegIndices(Source).size())
    return false;
  BitVector Clobbered = Forbidden;
  for (MCPhysReg Part : Parts) {
    if (Locked[Part])
      return false;
    Clobbered.set(Part);
  }
  for (MCPhysReg Part : Parts)
    if (!evacuate(Part, Plan, LiveRegs, Locked, Clobbered))
      return false;
  SmallVector<MCPhysReg> Ops(Def->getNumOperands());
  Ops[0] = Dst;
  Plan.Instructions.push_back(Rematerialization{Def, Dst});
  Plan.Cost += instructionCost(*Def, Ops);
  for (auto [SubReg, Part] : llvm::zip_equal(subRegIndices(Source), Parts))
    Plan.Registers.define(Part, getValueNumber(Source, SubReg));
  return true;
}

bool FunctionAllocator::evacuate(MCPhysReg Reg, Implementation &Plan,
                                 const SparseBitVector<> &LiveRegs,
                                 BitVector Locked, const BitVector &Forbidden) {
  ValueNumber V = Plan.Registers.read(Reg);
  if (!isLiveValue(V, LiveRegs) || rematerialization(V))
    return true;
  for (MCPhysReg R : Plan.Registers.copies(V))
    if (R != Reg && !Forbidden[R])
      return true;
  Locked.set(Reg);
  // Any live alias's backing register can preserve these contents. Do not use
  // the ultimate source's backing assignment after its own live range has
  // ended.
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned SubReg : subRegIndices(R)) {
      if (getValueNumber(R, SubReg) != V)
        continue;
      MCPhysReg Backing = getBackingRegister(R, SubReg);
      if (!Backing || Locked[Backing] || Forbidden[Backing])
        continue;
      Implementation Trial = Plan;
      if (placeValue(V, Backing, Trial, LiveRegs, Locked, &Forbidden)) {
        Plan = std::move(Trial);
        return true;
      }
    }
  }
  // Only an unavailable backing register requires an arbitrary imaginary
  // destination. Choose empty storage on demand; Imag16 lanes can be evacuated
  // separately.
  for (MCPhysReg Scratch : RCI.getOrder(&MOS::Imag8RegClass)) {
    if (Locked[Scratch] || Forbidden[Scratch] ||
        isLiveValue(Plan.Registers.read(Scratch), LiveRegs))
      continue;
    Implementation Trial = Plan;
    if (placeValue(V, Scratch, Trial, LiveRegs, Locked, &Forbidden)) {
      Plan = std::move(Trial);
      return true;
    }
  }
  return false;
}

bool FunctionAllocator::copyRegister(MCPhysReg Dst, MCPhysReg Src,
                                     Implementation &Plan) {
  if (Dst == Src)
    return true;
  unsigned Opcode, Cost;
  if (MOS::GPRRegClass.contains(Dst) && MOS::Imag8RegClass.contains(Src)) {
    Opcode = MOS::LDImag8;
    Cost = 2;
  } else if (MOS::Imag8RegClass.contains(Dst) &&
             MOS::GPRRegClass.contains(Src)) {
    Opcode = MOS::STImag8;
    Cost = 2;
  } else if (MOS::GPRRegClass.contains(Dst) && MOS::GPRRegClass.contains(Src) &&
             (Dst == MOS::A || Src == MOS::A)) {
    Opcode = Src == MOS::A ? MOS::TA : MOS::T_A;
    Cost = 1;
  } else
    return false;
  Plan.Instructions.push_back(RegisterCopy{Opcode, Dst, Src});
  Plan.Cost += Cost;
  Plan.Registers.copy(Dst, Src);
  return true;
}

// The input opcode is fixed, but some pseudos expand differently depending on
// placement. Account for the byte costs relevant to sieve; the remaining fixed
// opcode costs do not affect the choice between placements of that operation.
unsigned FunctionAllocator::instructionCost(const MachineInstr &MI,
                                            ArrayRef<MCPhysReg> Ops) const {
  switch (MI.getOpcode()) {
  case TargetOpcode::IMPLICIT_DEF:
    return 0;
  case MOS::LDImm:
    return 2;
  case MOS::LDCImm:
  case MOS::CLV:
  case MOS::CL:
    return 1;
  case MOS::LDImm1:
    if (Ops[0] == MOS::C || (Ops[0] == MOS::V && !MI.getOperand(1).getImm()))
      return 1;
    if (Ops[0] == MOS::V) {
      const auto &ST = MF.getSubtarget<MOSSubtarget>();
      return ST.hasSPC700() ? 6 : ST.hasHUC6280() ? 2 : 3;
    }
    return 2;
  case MOS::LDZ:
    return MF.getSubtarget<MOSSubtarget>().hasHUC6280() &&
                   MOS::GPRRegClass.contains(Ops[0])
               ? 1
               : 2;
  case MOS::LDImm16Remat:
    // The scavenger may add preservation around its hardware scratch register.
    return MF.getSubtarget<MOSSubtarget>().hasSPC700() ? 6 : 8;
  case MOS::LDImm16SPC700:
    return 6;
  case MOS::ASL:
  case MOS::ROL:
    return Ops[0] == MOS::A ? 1 : 2;
  case MOS::IncMB: {
    unsigned Cost = 0;
    unsigned N = MI.getNumExplicitDefs();
    for (unsigned I = 0; I < N; ++I) {
      Cost += Ops[I] == MOS::A                      ? 3
              : MOS::Imag8RegClass.contains(Ops[I]) ? 2
                                                    : 1;
      if (I + 1 != N)
        Cost += 2;
    }
    return Cost;
  }
  default:
    return TII.getInstSizeInBytes(MI);
  }
}

SmallVector<MCPhysReg> FunctionAllocator::destinations(Register R) const {
  SmallVector<MCPhysReg> Result;
  // Hardware members of an operand class are meaningful instruction choices.
  // Ordinary values use their backing register or a concrete constraint for
  // imaginary storage. Rematerializable values have no backing assignment, so
  // constrained uses may need to acquire temporary storage here.
  if (subRegIndices(R).size() == 1) {
    if (TRI.getRegSizeInBits(*MRI.getRegClass(R)) == 1) {
      Result.push_back(MOS::C);
      Result.push_back(MOS::V);
    } else
      for (MCPhysReg H : {MOS::A, MOS::X, MOS::Y})
        Result.push_back(H);
  }
  if (MCPhysReg H = VRM.getPhys(R))
    Result.push_back(H);
  else if (rematerialization(getValueNumber(R, subRegIndices(R).front()))) {
    auto Order =
        RCI.getOrder(subRegIndices(R).size() == 2 ? &MOS::Imag16RegClass
                                                  : &MOS::Imag8RegClass);
    Result.append(Order.begin(), Order.end());
  }
  auto I = Constraints.find(VRM.getOriginal(R));
  if (I != Constraints.end())
    for (MCPhysReg C : I->second)
      if (!llvm::is_contained(Result, C))
        Result.push_back(C);
  return Result;
}

SmallVector<MCPhysReg>
FunctionAllocator::copies(Register R, const RegisterContents &S) const {
  SmallVector<MCPhysReg> Result;
  ValueNumber V = getValueNumber(R, subRegIndices(R).front());
  if (!V.Reg)
    return Result;
  for (MCPhysReg PhysReg : S.copies(V)) {
    if (subRegIndices(R).size() == 1) {
      Result.push_back(PhysReg);
      continue;
    }
    MCPhysReg Imag16 =
        TRI.getMatchingSuperReg(PhysReg, MOS::sublo, &MOS::Imag16RegClass);
    if (Imag16 && S.read(TRI.getSubReg(Imag16, MOS::subhi)) ==
                      getValueNumber(R, MOS::subhi))
      Result.push_back(Imag16);
  }
  return Result;
}

bool FunctionAllocator::hasLiveValues(const RegisterContents &S,
                                      const SparseBitVector<> &LiveRegs) const {
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned SubReg : subRegIndices(R))
      if (!rematerialization(getValueNumber(R, SubReg)) &&
          !S.hasCopy(getValueNumber(R, SubReg)))
        return false;
  }
  return true;
}

bool FunctionAllocator::isLiveValue(ValueNumber V,
                                    const SparseBitVector<> &LiveRegs) const {
  if (!V.Reg)
    return false;
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned SubReg : subRegIndices(R))
      if (getValueNumber(R, SubReg) == V)
        return true;
  }
  return false;
}

bool FunctionAllocator::sameValue(Register A, Register B) const {
  if (A == B)
    return true;
  if (!A.isVirtual() || !B.isVirtual() ||
      subRegIndices(A).size() != subRegIndices(B).size())
    return false;
  for (unsigned SubReg : subRegIndices(A))
    if (getValueNumber(A, SubReg) != getValueNumber(B, SubReg))
      return false;
  return true;
}

SparseBitVector<> FunctionAllocator::liveOuts(MachineBasicBlock &MBB) const {
  SparseBitVector<> LiveRegs;
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI.use_nodbg_empty(R) &&
        llvm::any_of(MBB.successors(), [&](MachineBasicBlock *Succ) {
          return LV.isLiveIn(R, *Succ);
        }))
      LiveRegs.set(I);
  }
  // PHI inputs are edge uses, not live-ins of the successor block.
  for (MachineBasicBlock *Succ : MBB.successors())
    for (MachineInstr &Phi : Succ->phis())
      for (unsigned I = 1; I < Phi.getNumOperands(); I += 2)
        if (Phi.getOperand(I + 1).getMBB() == &MBB &&
            !Phi.getOperand(I).isUndef())
          LiveRegs.set(Phi.getOperand(I).getReg().virtRegIndex());
  return LiveRegs;
}

ArrayRef<unsigned> FunctionAllocator::subRegIndices(Register R) const {
  static constexpr unsigned Whole[] = {0};
  static constexpr unsigned Bytes[] = {MOS::sublo, MOS::subhi};
  bool IsImag16 = R.isPhysical()
                      ? MOS::Imag16RegClass.contains(R)
                      : TRI.getRegSizeInBits(*MRI.getRegClass(R)) == 16;
  return IsImag16 ? ArrayRef<unsigned>(Bytes) : ArrayRef<unsigned>(Whole);
}

ValueNumber FunctionAllocator::getValueNumber(Register R,
                                              unsigned SubReg) const {
  assert(llvm::is_contained(subRegIndices(R), SubReg));
  // Copy ancestry preserves size and byte order. REG_SEQUENCE additionally
  // identifies a pair's selected byte with an entire Imag8 source value.
  R = VRM.getOriginal(R);
  const MachineInstr *Def = MRI.getVRegDef(R);
  if (Def && Def->getOpcode() == TargetOpcode::REG_SEQUENCE)
    for (unsigned I = 1; I < Def->getNumOperands(); I += 2) {
      const MachineOperand &Source = Def->getOperand(I);
      if (Def->getOperand(I + 1).getImm() == SubReg &&
          Source.getReg().isVirtual() && !Source.isUndef() &&
          !Source.getSubReg() &&
          TRI.getRegSizeInBits(*MRI.getRegClass(Source.getReg())) == 8)
        return ValueNumber(VRM.getOriginal(Source.getReg()));
    }
  // PHIs and physical-register captures introduce new values. In particular,
  // loop-carried PHI operands are not unconditionally equal to their result.
  return ValueNumber(R, SubReg);
}

SmallVector<MCPhysReg, 2> FunctionAllocator::registerParts(MCPhysReg R) const {
  if (MOS::Imag16RegClass.contains(R))
    return {MCPhysReg(TRI.getSubReg(R, MOS::sublo)),
            MCPhysReg(TRI.getSubReg(R, MOS::subhi))};
  if (MOS::Imag8RegClass.contains(R) || llvm::is_contained(HardwareRegs, R))
    return {R};
  return {};
}

MCPhysReg FunctionAllocator::getBackingRegister(Register R,
                                                unsigned SubReg) const {
  MCPhysReg BackingReg = VRM.getPhys(R);
  if (BackingReg && SubReg)
    return TRI.getSubReg(BackingReg, SubReg);
  return BackingReg;
}

const MachineInstr *FunctionAllocator::rematerialization(ValueNumber V) const {
  if (!V.Reg)
    return nullptr;
  const MachineInstr *Def = MRI.getVRegDef(V.Reg);
  return Def && TII.isTriviallyReMaterializable(*Def) ? Def : nullptr;
}

void FunctionAllocator::emitInstructions(
    MachineBasicBlock &MBB, MachineBasicBlock::iterator At, const DebugLoc &DL,
    ArrayRef<EmittedInstruction> Instructions, MachineInstr *MI) {
  for (const EmittedInstruction &Instruction : Instructions) {
    if (const auto *Copy = std::get_if<RegisterCopy>(&Instruction)) {
      BuildMI(MBB, At, DL, TII.get(Copy->Opcode), Copy->Dst).addReg(Copy->Src);
      ++NumTransfers;
    } else if (const auto *Remat =
                   std::get_if<Rematerialization>(&Instruction)) {
      TII.reMaterialize(MBB, At, Remat->Dst, 0, *Remat->Definition);
      ++NumTransfers;
    } else {
      const auto &Assignment = std::get<OperandAssignment>(Instruction);
      assert(MI && "operand assignments need an original instruction");
      for (unsigned I = 0; I < MI->getNumOperands(); ++I) {
        MachineOperand &MO = MI->getOperand(I);
        if (MO.isReg() && MO.getReg().isVirtual()) {
          MO.setReg(Assignment.Registers[I]);
          MO.setIsRenamable(false);
        }
      }
      At = std::next(MI->getIterator());
    }
  }
}

void FunctionAllocator::emitEdges() {
  // Snapshot PHI edges before inserting repair blocks changes the CFG.
  for (const PHIEdge &Edge : collectPHIEdges())
    emitPHIEdge(Edge);
}

SmallVector<FunctionAllocator::PHIEdge> FunctionAllocator::collectPHIEdges() {
  SmallVector<PHIEdge> Edges;
  for (MachineBasicBlock &MBB : MF)
    for (MachineBasicBlock *Pred : MBB.predecessors()) {
      PHIEdge E{Pred, &MBB, {}};
      for (MachineInstr &Phi : MBB.phis()) {
        Register D = Phi.getOperand(0).getReg();
        if (MRI.use_nodbg_empty(D))
          continue;
        for (unsigned I = 1; I < Phi.getNumOperands(); I += 2)
          if (Phi.getOperand(I + 1).getMBB() == Pred &&
              !Phi.getOperand(I).isUndef())
            E.Copies.push_back({D, Phi.getOperand(I).getReg()});
      }
      if (!E.Copies.empty())
        Edges.push_back(std::move(E));
    }
  return Edges;
}

void FunctionAllocator::emitPHIEdge(const PHIEdge &E) {
  Implementation Plan;
  const BlockPlan &B = Blocks[E.From];
  forEachBacking(B.Points.back().LiveRegs, [&](MCPhysReg R, ValueNumber V) {
    Plan.Registers.define(R, V);
  });
  BitVector Locked = B.Points.back().FixedRegs;
  // Protect completed destinations. Evacuation preserves sources of cycles
  // in temporary imaginary storage until their edge copy consumes them.
  for (auto [D, S] : E.Copies)
    for (unsigned SubReg : subRegIndices(D)) {
      MCPhysReg Backing = getBackingRegister(D, SubReg);
      if (!placeValue(getValueNumber(S, SubReg), Backing, Plan,
                      B.Points.back().LiveRegs, Locked))
        fail("parallel PHI repair needs unsupported preservation");
      Locked.set(Backing);
    }
  if (Plan.Instructions.empty())
    return;
  bool FallThrough = E.From->getFallThrough() == E.To;
  MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  E.From->ReplaceUsesOfBlockWith(E.To, MBB);
  if (FallThrough)
    BuildMI(*E.From, E.From->end(), DebugLoc(), TII.get(MOS::JMP)).addMBB(MBB);
  MBB->addSuccessor(E.To);
  emitInstructions(*MBB, MBB->end(), DebugLoc(), Plan.Instructions);
  BuildMI(*MBB, MBB->end(), DebugLoc(), TII.get(MOS::JMP)).addMBB(E.To);
}

void FunctionAllocator::emitSolution(MachineBasicBlock &MBB,
                                     const BlockPlan &Block) {
  const AllocationTable &Final = Block.Points.back().Allocations;
  auto Best = llvm::min_element(Final, [](const auto &A, const auto &B) {
    return A.second.Cost < B.second.Cost;
  });
  unsigned Index = std::distance(Final.begin(), Best);
  SmallVector<const DPEntry *> Selected(Block.Points.size());
  for (unsigned I = Block.Points.size(); I-- > 0;) {
    const DPEntry &Entry =
        std::next(Block.Points[I].Allocations.begin(), Index)->second;
    Selected[I] = &Entry;
    Index = Entry.Previous;
  }
  for (auto [Point, Entry] : llvm::zip_equal(Block.Points, Selected)) {
    MachineInstr *MI = Point.MI;
    auto At = MI ? MI->getIterator() : MBB.getFirstTerminator();
    emitInstructions(MBB, At, MI ? MI->getDebugLoc() : DebugLoc(),
                     Entry->Instructions, MI);
  }
}

void FunctionAllocator::eraseVirtualInstructions() {
  for (auto &[MBB, B] : Blocks)
    for (const ProgramPoint &Point : B.Points) {
      MachineInstr *MI = Point.MI;
      if (!MI)
        continue;
      if (MI->isPHI() ||
          (MI->isCopy() && (MI->getOperand(0).getReg().isVirtual() ||
                            MI->getOperand(1).getReg().isVirtual())) ||
          (TII.isTriviallyReMaterializable(*MI) &&
           MI->getOperand(0).getReg().isVirtual()) ||
          MI->getOpcode() == TargetOpcode::REG_SEQUENCE)
        MI->eraseFromParent();
    }
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isDebugInstr())
        for (MachineOperand &MO : MI.operands())
          if (MO.isReg() && MO.getReg().isVirtual())
            MO.setReg(0);
  MRI.clearVirtRegs();
}

void FunctionAllocator::recomputePhysicalLiveness() {
  SmallVector<MachineBasicBlock *> BlocksToUpdate;
  for (MachineBasicBlock &MBB : reverse(MF)) {
    MBB.clearLiveIns();
    BlocksToUpdate.push_back(&MBB);
  }
  fullyRecomputeLiveIns(BlocksToUpdate);
  for (MachineBasicBlock &MBB : MF)
    recomputeLivenessFlags(MBB);
}

void FunctionAllocator::fail(const Twine &Reason,
                             const MachineInstr *MI) const {
  errs() << "MOSRegAlloc: " << Reason << " in " << MF.getName() << '\n';
  if (MI)
    errs() << *MI;
  report_fatal_error("unsupported MOS register repair", false);
}

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;
  MOSRegAlloc();

  bool runOnMachineFunction(MachineFunction &MF) override;
  MachineFunctionProperties getRequiredProperties() const override;
  MachineFunctionProperties getSetProperties() const override;
  MachineFunctionProperties getClearedProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
};

MOSRegAlloc::MOSRegAlloc() : MachineFunctionPass(ID) {
  initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
}

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getRegInfo().getNumVirtRegs())
    return false;
  FunctionAllocator(MF, getAnalysis<VirtRegMapWrapperLegacy>().getVRM(),
                    getAnalysis<LiveVariablesWrapperPass>().getLV(),
                    getAnalysis<MachineRegisterClassInfoWrapperPass>().getRCI())
      .run();
  return true;
}

MachineFunctionProperties MOSRegAlloc::getRequiredProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

MachineFunctionProperties MOSRegAlloc::getSetProperties() const {
  return MachineFunctionProperties().setNoVRegs().setNoPHIs();
}

MachineFunctionProperties MOSRegAlloc::getClearedProperties() const {
  return MachineFunctionProperties().setIsSSA();
}

void MOSRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariablesWrapperPass>();
  AU.addRequired<VirtRegMapWrapperLegacy>();
  AU.addRequired<MachineRegisterClassInfoWrapperPass>();
}

} // namespace

char MOSRegAlloc::ID = 0;
INITIALIZE_PASS_BEGIN(MOSRegAlloc, DEBUG_TYPE, "MOS register constraint repair",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveVariablesWrapperPass)
INITIALIZE_PASS_DEPENDENCY(VirtRegMapWrapperLegacy)
INITIALIZE_PASS_DEPENDENCY(MachineRegisterClassInfoWrapperPass)
INITIALIZE_PASS_END(MOSRegAlloc, DEBUG_TYPE, "MOS register constraint repair",
                    false, false)
MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
