//===----------------------------------------------------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Realize SSA instructions by repairing their operand placement constraints.
/// MOSImagRegAlloc supplies backing registers through VirtRegMap. A block-local
/// dynamic program tracks hardware contents and backing validity, charging for
/// the instructions and transfers needed for each operation. A retained state
/// guarantees backing copies for live values absent from hardware. Instruction
/// repairs may use other imaginary locations temporarily.
/// Value numbers derived from split ancestry and REG_SEQUENCE identify equal
/// contents independently of SSA live ranges and their backing assignments.
/// Ordinary eviction uses a live range's backing register; a constrained access
/// to occupied backing storage can evacuate it locally. Each surviving live
/// range's backing register is restored before leaving the block, except for
/// contents that can be rematerialized.
///
/// This initial implementation retains the input schedule and supports the
/// Imag8/Imag16 operations used by sieve. Equal states retain only their
/// cheapest path; there is no beam pruning. Transfer construction chooses
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
#include <map>
#include <memory>
#include <utility>

using namespace llvm;

#define DEBUG_TYPE "mos-regalloc"
STATISTIC(NumStates, "Number of MOS register placement states retained");
STATISTIC(NumTransfers, "Number of MOS register repair instructions emitted");

namespace {

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

// A value number identifies an ultimate source SSA register and byte lane (or
// a one-bit flag). COPY and REG_SEQUENCE names can share these numbers without
// sharing backing assignments. Zero denotes unknown contents.
using ValueNumber = unsigned;
ValueNumber sourceValue(Register R, unsigned Lane = 0) {
  return 2 * R.virtRegIndex() + 1 + Lane;
}
Register sourceReg(ValueNumber V) {
  return Register::index2VirtReg((V - 1) / 2);
}

struct Transfer {
  unsigned Opcode;
  MCPhysReg Dst, Src;
  // A non-null recipe is emitted with TII.reMaterialize; Opcode/Src are unused.
  const MachineInstr *Rematerialization = nullptr;
};

constexpr std::array<MCPhysReg, 5> HardwareRegs = {MOS::A, MOS::X, MOS::Y,
                                                   MOS::C, MOS::V};

// Explicit contents while realizing an instruction. Locations begin with
// HardwareRegs, followed by individual imaginary bytes. Imag16 accesses use
// their two byte indices, so overwrites invalidate exactly the affected lanes.
class RegisterContents {
public:
  explicit RegisterContents(unsigned NumLocations = 0)
      : Contents(NumLocations) {}

  unsigned size() const { return Contents.size(); }
  ValueNumber read(unsigned L) const { return Contents[L]; }
  void define(unsigned L, ValueNumber V) { Contents[L] = V; }
  void clobber(unsigned L) { define(L, 0); }
  void copy(unsigned D, unsigned S) { define(D, read(S)); }
  bool hasCopy(ValueNumber V) const {
    return V && llvm::is_contained(Contents, V);
  }
  bool hasHardwareCopy(ValueNumber V) const {
    return V && llvm::is_contained(
                    ArrayRef(Contents).take_front(HardwareRegs.size()), V);
  }

private:
  SmallVector<ValueNumber, 40> Contents;
};

// A retained placement stores hardware contents and backing validity only.
// Backing contents themselves are determined by the live SSA names at this
// point. Every non-rematerializable live lane must have a valid backing copy
// whenever its value is absent from hardware. Equivalent names can have
// distinct backing registers, so validity belongs to locations rather than
// value numbers. Copies in other imaginary locations are temporary repair
// state. Comparisons and expansion use the same program point's backing map.
class PlacementState {
public:
  PlacementState(const RegisterContents &Registers,
                 const RegisterContents &Backings);
  RegisterContents expand(const RegisterContents &Backings) const;

  bool operator<(const PlacementState &Other) const {
    if (Hardware != Other.Hardware)
      return Hardware < Other.Hardware;
    auto A = ValidBackings.getData();
    auto B = Other.ValidBackings.getData();
    return std::lexicographical_compare(A.begin(), A.end(), B.begin(), B.end());
  }

private:
  std::array<ValueNumber, HardwareRegs.size()> Hardware = {};
  BitVector ValidBackings;
};

PlacementState::PlacementState(const RegisterContents &Registers,
                               const RegisterContents &Backings)
    : ValidBackings(Backings.size() - HardwareRegs.size()) {
  assert(Registers.size() == Backings.size());
  for (unsigned L = 0; L < Hardware.size(); ++L)
    Hardware[L] = Registers.read(L);
  for (unsigned L = Hardware.size(); L < Backings.size(); ++L) {
    ValueNumber V = Backings.read(L);
    if (!V)
      continue;
    bool Valid = Registers.read(L) == V;
    assert((Valid || Registers.hasHardwareCopy(V)) &&
           "live value must be in hardware or its backing register");
    ValidBackings[L - Hardware.size()] = Valid;
  }
}

RegisterContents
PlacementState::expand(const RegisterContents &Backings) const {
  assert(ValidBackings.size() + Hardware.size() == Backings.size());
  RegisterContents Registers = Backings;
  for (unsigned L = 0; L < Hardware.size(); ++L)
    Registers.define(L, Hardware[L]);
  for (unsigned L = Hardware.size(); L < Registers.size(); ++L)
    if (!ValidBackings[L - Hardware.size()])
      Registers.clobber(L);
  return Registers;
}

// A proposed operation and its preparation, starting from an incoming
// candidate.
struct RepairPlan {
  RegisterContents State;
  SmallVector<Transfer> Transfers;
  SmallVector<MCPhysReg> Operands;
  // Estimated encoded bytes for the path prefix, repairs, and operation.
  unsigned Cost = 0;
  // Restore the placement invariant after executing the operation.
  SmallVector<Transfer> AfterTransfers = {};
};

// Paths share their prefixes. MIR is not mutated until an entire block has a
// feasible exit, so failed candidates cannot leave partial repairs behind.
struct Trace {
  std::shared_ptr<const Trace> Previous;
  MachineInstr *MI = nullptr; // Null denotes the block-boundary restoration.
  SmallVector<Transfer> Transfers;
  SmallVector<MCPhysReg> Operands;
  SmallVector<Transfer> AfterTransfers = {};
};

// A retained placement and the path that reaches it.
struct Candidate {
  PlacementState State;
  unsigned Cost = 0;
  std::shared_ptr<const Trace> Path;
};

struct InstructionLiveness {
  // Virtual register indices, including PHI edge uses. The search revisits
  // these sets for every placement candidate.
  SparseBitVector<> LiveIns, LiveOuts;
  BitVector FixedLiveIns, FixedLiveOuts; // Indices in Locations.
};

// Keep original instructions alive for rematerialization and deferred emission.
struct BlockPlan {
  SparseBitVector<> LiveIns, LiveOuts;
  SmallVector<MachineInstr *> Instructions;
  BitVector FixedLiveOuts;
  std::shared_ptr<const Trace> Solution;
};

class RepairAllocator {
public:
  RepairAllocator(MachineFunction &MF, const VirtRegMap &VRM, LiveVariables &LV,
                  const RegisterClassInfo &RCI);
  void run();

private:
  class BlockSearch;
  class OperandSearch;
  class InstructionRepair;

  struct PHIEdge {
    MachineBasicBlock *From, *To;
    SmallVector<std::pair<Register, Register>> Copies;
  };

  void numberValues();
  void collectLocations();
  void collectConstraints();
  void analyzeLiveness();
  void planBlock(MachineBasicBlock &MBB);
  void emitEdges();
  SmallVector<PHIEdge> collectPHIEdges();
  void emitPHIEdge(const PHIEdge &Edge);
  void emitSolution(MachineBasicBlock &MBB, const Trace *Last);
  void eraseVirtualInstructions();
  void recomputePhysicalLiveness();

  ValueNumber numberValue(Register R, unsigned Lane);
  BitVector fixedLocations(const LivePhysRegs &LiveRegs) const;

  void enumeratePlans(MachineInstr &MI, const RepairPlan &C,
                      function_ref<void(RepairPlan)> Accept);
  RegisterContents backingContents(const SparseBitVector<> &LiveRegs) const;
  bool finishPlacement(RepairPlan &P, const RegisterContents &Backings,
                       const SparseBitVector<> &LiveRegs, BitVector Locked,
                       bool CanInsert);
  bool restore(RepairPlan &P, const SparseBitVector<> &LiveRegs,
               const SparseBitVector<> &Preserve, BitVector Locked);

  // Locked locations may be read but not changed by repair instructions.
  // Forbidden additionally protects locations about to be clobbered: evacuation
  // must not preserve a value in another member of that same clobber set.
  bool ensure(ValueNumber A, unsigned L, RepairPlan &P,
              const SparseBitVector<> &LiveRegs, BitVector Locked,
              const BitVector *Forbidden = nullptr);
  bool rematerialize(ValueNumber V, unsigned L, RepairPlan &P,
                     const SparseBitVector<> &LiveRegs, BitVector Locked,
                     const BitVector &Forbidden);
  bool evacuate(unsigned L, RepairPlan &P, const SparseBitVector<> &LiveRegs,
                BitVector Locked, const BitVector &Forbidden);
  bool transfer(unsigned D, unsigned S, RepairPlan &P);
  void emitTransfers(MachineBasicBlock &MBB, MachineBasicBlock::iterator At,
                     const DebugLoc &DL, ArrayRef<Transfer> Transfers);
  unsigned instructionCost(const MachineInstr &MI,
                           ArrayRef<MCPhysReg> Operands) const;

  SmallVector<MCPhysReg> destinations(Register R) const;
  SmallVector<MCPhysReg> copies(Register R, const RegisterContents &S) const;
  void forgetDeadValues(RegisterContents &S,
                        const SparseBitVector<> &LiveRegs) const;
  bool hasLiveValues(const RegisterContents &S,
                     const SparseBitVector<> &LiveRegs) const;
  bool isLiveValue(ValueNumber V, const SparseBitVector<> &LiveRegs) const;
  bool sameValue(Register A, Register B) const;
  SparseBitVector<> liveOuts(MachineBasicBlock &MBB) const;
  void addLocation(MCPhysReg R);
  void addConstraint(Register R, MCPhysReg Location);

  // One lane per byte; flags occupy a single lane too.
  unsigned numLanes(Register R) const;
  ValueNumber value(Register R, unsigned Lane = 0) const;
  SmallVector<unsigned, 2> locationIndices(MCPhysReg R) const;
  // Backing storage belongs to a live range, not to its value number.
  // Return the lane's index in Locations, or ~0u if unassigned.
  unsigned backingIndex(Register R, unsigned Lane = 0) const;
  bool isGPR(unsigned L) const;
  bool isImaginary(unsigned L) const;
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
  // Indexed by sourceValue(R, Lane). Entries identify the underlying contents,
  // while liveness, constraints, and backing assignments remain per SSA name.
  SmallVector<ValueNumber> ValueNumbers;
  SmallVector<MCPhysReg, 40> Locations;
  DenseMap<MCPhysReg, unsigned> LocationIndices;
  BitVector ScratchLocations;
  DenseMap<Register, SmallVector<MCPhysReg>> Constraints;
  DenseMap<MachineInstr *, InstructionLiveness> Liveness;
  DenseMap<MachineBasicBlock *, BlockPlan> Blocks;
};

// One DP step retains the cheapest path for each distinct placement. Keep the
// candidate vector and its index together so replacement cannot desynchronize
// them.
class PlacementFrontier {
public:
  void insert(const Candidate &Previous, MachineInstr *MI, RepairPlan Plan,
              PlacementState State);
  SmallVector<Candidate> takeCandidates();

private:
  SmallVector<Candidate> Candidates;
  std::map<PlacementState, unsigned> StateIndices;
};

// Search state lasts for one block. A null instruction advances through the
// backing-restoration step immediately before the terminators.
class RepairAllocator::BlockSearch {
public:
  BlockSearch(RepairAllocator &Allocator, MachineBasicBlock &MBB);
  std::shared_ptr<const Trace> run();

private:
  void initialize();
  void advance(MachineInstr *MI);
  bool restoreLiveOuts(RepairPlan &Plan);

  RegisterContents Backings;

  RepairAllocator &Allocator;
  MachineBasicBlock &MBB;
  const BlockPlan &Block;
  SmallVector<Candidate> Candidates;
};

// Enumerate operand assignments for one incoming placement. A recursive level
// owns one entry of Assignment; earlier entries constrain ties and overlap.
class RepairAllocator::OperandSearch {
public:
  OperandSearch(RepairAllocator &Allocator, MachineInstr &MI,
                const RepairPlan &Incoming,
                function_ref<void(RepairPlan)> Accept);
  void run();

private:
  void buildDomain(unsigned OpIdx);
  void addCopies(unsigned OpIdx, Register R);
  void addDestination(unsigned OpIdx, MCPhysReg R);
  bool isLegal(unsigned OpIdx, MCPhysReg R) const;
  bool isTiedUseAlias(const MachineOperand &Def,
                      const MachineOperand &Use) const;
  void search(unsigned OpIdx);

  RepairAllocator &Allocator;
  MachineInstr &MI;
  const RepairPlan &Incoming;
  function_ref<void(RepairPlan)> Accept;
  SmallVector<SmallVector<MCPhysReg>> Domains;
  SmallVector<MCPhysReg> Assignment;
};

// Prepare one concrete operand assignment. Locked locations protect prepared
// inputs; clobbered locations must all be known before choosing evacuations.
class RepairAllocator::InstructionRepair {
public:
  InstructionRepair(RepairAllocator &Allocator, MachineInstr &MI,
                    RepairPlan &Plan);
  bool run();

private:
  bool repairCopy();
  bool repairRegSequence();
  bool prepareUses(bool InGPRs);
  bool preserveLiveThroughValues();
  bool applyInstruction();
  bool define(const MachineOperand &MO);

  RepairAllocator &Allocator;
  MachineInstr &MI;
  RepairPlan &Plan;
  const InstructionLiveness &Liveness;
  BitVector Locked;
};

MOSRegAlloc::MOSRegAlloc() : MachineFunctionPass(ID) {
  initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
}

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  if (!MF.getRegInfo().getNumVirtRegs())
    return false;
  RepairAllocator(MF, getAnalysis<VirtRegMapWrapperLegacy>().getVRM(),
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

RepairAllocator::RepairAllocator(MachineFunction &MF, const VirtRegMap &VRM,
                                 LiveVariables &LV,
                                 const RegisterClassInfo &RCI)
    : MF(MF), MRI(MF.getRegInfo()), TII(*MF.getSubtarget().getInstrInfo()),
      TRI(*MF.getSubtarget().getRegisterInfo()), VRM(VRM), LV(LV), RCI(RCI) {}

void RepairAllocator::run() {
  numberValues();
  collectLocations();
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
      emitSolution(MBB, I->second.Solution.get());
  eraseVirtualInstructions();
  recomputePhysicalLiveness();
}

void RepairAllocator::numberValues() {
  ValueNumbers.resize(2 * MRI.getNumVirtRegs() + 1);
  for (unsigned I = 0; I < MRI.getNumVirtRegs(); ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI.def_empty(R))
      for (unsigned Lane = 0; Lane < numLanes(R); ++Lane)
        numberValue(R, Lane);
  }
}

ValueNumber RepairAllocator::numberValue(Register R, unsigned Lane) {
  ValueNumber &V = ValueNumbers[sourceValue(R, Lane)];
  if (V)
    return V;
  Register Original = VRM.getOriginal(R);
  if (Original != R)
    return V = numberValue(Original, Lane);
  const MachineInstr *Def = MRI.getVRegDef(R);
  if (Def && Def->getOpcode() == TargetOpcode::REG_SEQUENCE)
    for (unsigned I = 1; I < Def->getNumOperands(); I += 2) {
      const MachineOperand &Source = Def->getOperand(I);
      unsigned SubReg = Def->getOperand(I + 1).getImm();
      if (SubReg == (Lane == 0 ? MOS::sublo : MOS::subhi) &&
          Source.getReg().isVirtual() && !Source.isUndef() &&
          !Source.getSubReg() &&
          TRI.getRegSizeInBits(*MRI.getRegClass(Source.getReg())) == 8)
        return V = numberValue(Source.getReg(), 0);
    }
  // PHIs and physical-register captures introduce new values. In particular,
  // loop-carried PHI operands are not unconditionally equal to their result.
  return V = sourceValue(R, Lane);
}

void RepairAllocator::collectLocations() {
  for (MCPhysReg R : HardwareRegs)
    addLocation(R);
  for (MCPhysReg R : RCI.getOrder(&MOS::Imag8RegClass))
    addLocation(R);
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      for (const MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.getReg().isPhysical() &&
            (MOS::Imag8RegClass.contains(MO.getReg()) ||
             MOS::Imag16RegClass.contains(MO.getReg())))
          addLocation(MO.getReg());
  ScratchLocations.resize(Locations.size());
  for (unsigned I = 0; I < Locations.size(); ++I)
    if (isImaginary(I) && !MRI.isReserved(Locations[I]))
      ScratchLocations.set(I);
}

void RepairAllocator::collectConstraints() {
  // A physical COPY exposes a singleton constraint shared by all whole-value
  // aliases. VirtRegMap already records their common source.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isFullCopy() && MI.getOperand(0).getReg().isPhysical() &&
          MI.getOperand(1).getReg().isVirtual())
        addConstraint(MI.getOperand(1).getReg(), MI.getOperand(0).getReg());
}

void RepairAllocator::analyzeLiveness() {
  for (MachineBasicBlock &MBB : MF) {
    BlockPlan &B = Blocks[&MBB];
    B.LiveOuts = liveOuts(MBB);
    SparseBitVector<> LiveRegs = B.LiveOuts;
    LivePhysRegs Phys(TRI);
    Phys.addLiveOutsNoPristines(MBB);
    B.FixedLiveOuts = fixedLocations(Phys);
    for (MachineInstr &MI : reverse(MBB)) {
      if (MI.isDebugInstr())
        continue;
      B.Instructions.push_back(&MI);
      InstructionLiveness &L = Liveness[&MI];
      L.LiveOuts = LiveRegs;
      for (const MachineOperand &MO : MI.all_defs())
        if (MO.getReg().isVirtual())
          LiveRegs.reset(MO.getReg().virtRegIndex());
      // PHI inputs are uses on predecessor edges, already included in liveOuts.
      if (!MI.isPHI())
        for (const MachineOperand &MO : MI.all_uses())
          if (MO.getReg().isVirtual() && !MO.isUndef())
            LiveRegs.set(MO.getReg().virtRegIndex());
      L.LiveIns = LiveRegs;
      L.FixedLiveOuts = fixedLocations(Phys);
      Phys.stepBackward(MI);
      L.FixedLiveIns = fixedLocations(Phys);
    }
    B.LiveIns = LiveRegs;
    std::reverse(B.Instructions.begin(), B.Instructions.end());
  }
}

BitVector RepairAllocator::fixedLocations(const LivePhysRegs &LiveRegs) const {
  // Compare/branch pseudos keep N/Z internal until late optimization.
  assert(!LiveRegs.contains(MOS::N) && !LiveRegs.contains(MOS::Z) &&
         "N/Z must not be live during MOS register allocation");
  BitVector FixedLocations(Locations.size());
  for (unsigned I = 0; I < Locations.size(); ++I)
    if (LiveRegs.contains(Locations[I]))
      FixedLocations.set(I);
  return FixedLocations;
}

void RepairAllocator::planBlock(MachineBasicBlock &MBB) {
  Blocks[&MBB].Solution = BlockSearch(*this, MBB).run();
}

void PlacementFrontier::insert(const Candidate &Previous, MachineInstr *MI,
                               RepairPlan Plan, PlacementState State) {
  auto [I, Inserted] = StateIndices.emplace(State, Candidates.size());
  if (!Inserted && Candidates[I->second].Cost <= Plan.Cost)
    return;
  Candidate Next{
      std::move(State), Plan.Cost,
      std::make_shared<Trace>(
          Trace{Previous.Path, MI, std::move(Plan.Transfers),
                std::move(Plan.Operands), std::move(Plan.AfterTransfers)})};
  if (Inserted)
    Candidates.push_back(std::move(Next));
  else
    Candidates[I->second] = std::move(Next);
}

SmallVector<Candidate> PlacementFrontier::takeCandidates() {
  llvm::stable_sort(Candidates, [](const Candidate &A, const Candidate &B) {
    return A.Cost < B.Cost;
  });
  StateIndices.clear();
  return std::exchange(Candidates, {});
}

RepairAllocator::BlockSearch::BlockSearch(RepairAllocator &Allocator,
                                          MachineBasicBlock &MBB)
    : Allocator(Allocator), MBB(MBB),
      Block(Allocator.Blocks.find(&MBB)->second) {}

std::shared_ptr<const Trace> RepairAllocator::BlockSearch::run() {
  initialize();
  bool Restored = false;
  for (MachineInstr *MI : Block.Instructions) {
    if (MI->isTerminator() && !Restored) {
      advance(nullptr);
      Restored = true;
    }
    advance(MI);
  }
  if (!Restored)
    advance(nullptr);
  return Candidates.front().Path;
}

void RepairAllocator::BlockSearch::initialize() {
  Backings = Allocator.backingContents(Block.LiveIns);
  for (unsigned I : Block.LiveIns) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < Allocator.numLanes(R); ++Lane) {
      ValueNumber V = Allocator.value(R, Lane);
      if (Allocator.rematerialization(V))
        continue;
      unsigned Backing = Allocator.backingIndex(R, Lane);
      if (Backing == ~0u)
        Allocator.fail("block input has no backing register");
      if (!Block.Instructions.empty() &&
          Allocator.Liveness.find(Block.Instructions.front())
              ->second.FixedLiveIns[Backing])
        Allocator.fail("pinned physical live-in overlaps a backing register");
    }
  }
  Candidate Entry{PlacementState(Backings, Backings), 0, {}};
  Candidates.push_back(std::move(Entry));
}

void RepairAllocator::BlockSearch::advance(MachineInstr *MI) {
  const InstructionLiveness *L =
      MI ? &Allocator.Liveness.find(MI)->second : nullptr;
  RegisterContents NextBackings =
      MI ? Allocator.backingContents(L->LiveOuts) : Backings;
  PlacementFrontier Next;
  for (const Candidate &C : Candidates) {
    if (MI) {
      RepairPlan Incoming{C.State.expand(Backings), {}, {}, C.Cost};
      Allocator.enumeratePlans(*MI, Incoming, [&](RepairPlan Plan) {
        if (!Allocator.finishPlacement(Plan, NextBackings, L->LiveOuts,
                                       L->FixedLiveOuts, !MI->isTerminator()))
          return;
        PlacementState State(Plan.State, NextBackings);
        Next.insert(C, MI, std::move(Plan), std::move(State));
      });
    } else {
      RepairPlan Plan{C.State.expand(Backings), {}, {}, C.Cost};
      // Keep terminator inputs available at the restoration point, including
      // values which are not live out of the block.
      if (restoreLiveOuts(Plan)) {
        PlacementState State(Plan.State, Backings);
        Next.insert(C, nullptr, std::move(Plan), std::move(State));
      }
    }
  }
  Backings = std::move(NextBackings);
  Candidates = Next.takeCandidates();
  if (Candidates.empty())
    Allocator.fail(MI ? "no supported placement continuation"
                      : "cannot restore live values to backing registers",
                   MI);
  NumStates += Candidates.size();
  LLVM_DEBUG(dbgs() << "bb." << MBB.getNumber() << ": " << Candidates.size()
                    << " states, cost " << Candidates.front().Cost << '\n');
}

bool RepairAllocator::BlockSearch::restoreLiveOuts(RepairPlan &Plan) {
  auto Term = MBB.getFirstTerminator();
  const InstructionLiveness *L =
      Term == MBB.end() ? nullptr : &Allocator.Liveness.find(&*Term)->second;
  const BitVector &Locked = L ? L->FixedLiveIns : Block.FixedLiveOuts;
  const SparseBitVector<> &Preserve = L ? L->LiveIns : Block.LiveOuts;
  return Allocator.restore(Plan, Block.LiveOuts, Preserve, Locked) &&
         Allocator.finishPlacement(Plan, Backings, Preserve, Locked, true);
}

void RepairAllocator::emitEdges() {
  // Snapshot PHI edges before inserting repair blocks changes the CFG.
  for (const PHIEdge &Edge : collectPHIEdges())
    emitPHIEdge(Edge);
}

SmallVector<RepairAllocator::PHIEdge> RepairAllocator::collectPHIEdges() {
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

void RepairAllocator::emitPHIEdge(const PHIEdge &E) {
  RepairPlan P;
  P.State = RegisterContents(Locations.size());
  const BlockPlan &B = Blocks[E.From];
  for (int I : B.LiveOuts) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < numLanes(R); ++Lane)
      if (!rematerialization(value(R, Lane)))
        P.State.define(backingIndex(R, Lane), value(R, Lane));
  }
  BitVector Locked = B.FixedLiveOuts;
  // Protect completed destinations. Evacuation preserves sources of cycles
  // in temporary imaginary storage until their edge copy consumes them.
  for (auto [D, S] : E.Copies)
    for (unsigned Lane = 0; Lane < numLanes(D); ++Lane) {
      unsigned H = backingIndex(D, Lane);
      if (!ensure(value(S, Lane), H, P, B.LiveOuts, Locked))
        fail("parallel PHI repair needs unsupported preservation");
      Locked.set(H);
    }
  if (P.Transfers.empty())
    return;
  bool FallThrough = E.From->getFallThrough() == E.To;
  MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
  MF.push_back(MBB);
  E.From->ReplaceUsesOfBlockWith(E.To, MBB);
  if (FallThrough)
    BuildMI(*E.From, E.From->end(), DebugLoc(), TII.get(MOS::JMP)).addMBB(MBB);
  MBB->addSuccessor(E.To);
  emitTransfers(*MBB, MBB->end(), DebugLoc(), P.Transfers);
  BuildMI(*MBB, MBB->end(), DebugLoc(), TII.get(MOS::JMP)).addMBB(E.To);
}

void RepairAllocator::emitSolution(MachineBasicBlock &MBB, const Trace *Last) {
  SmallVector<const Trace *> Path;
  for (const Trace *T = Last; T; T = T->Previous.get())
    Path.push_back(T);
  for (const Trace *T : reverse(Path)) {
    auto At = T->MI ? T->MI->getIterator() : MBB.getFirstTerminator();
    emitTransfers(MBB, At, T->MI ? T->MI->getDebugLoc() : DebugLoc(),
                  T->Transfers);
    emitTransfers(MBB, T->MI ? std::next(At) : At,
                  T->MI ? T->MI->getDebugLoc() : DebugLoc(), T->AfterTransfers);
    if (!T->MI || T->Operands.empty() || T->MI->isCopy() ||
        T->MI->getOpcode() == TargetOpcode::REG_SEQUENCE)
      continue;
    for (unsigned I = 0; I < T->MI->getNumOperands(); ++I) {
      MachineOperand &MO = T->MI->getOperand(I);
      if (MO.isReg() && MO.getReg().isVirtual()) {
        MO.setReg(T->Operands[I]);
        MO.setIsRenamable(false);
      }
    }
  }
}

void RepairAllocator::eraseVirtualInstructions() {
  for (auto &[MBB, B] : Blocks)
    for (MachineInstr *MI : B.Instructions)
      if (MI->isPHI() ||
          (MI->isCopy() && (MI->getOperand(0).getReg().isVirtual() ||
                            MI->getOperand(1).getReg().isVirtual())) ||
          (TII.isTriviallyReMaterializable(*MI) &&
           MI->getOperand(0).getReg().isVirtual()) ||
          MI->getOpcode() == TargetOpcode::REG_SEQUENCE)
        MI->eraseFromParent();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isDebugInstr())
        for (MachineOperand &MO : MI.operands())
          if (MO.isReg() && MO.getReg().isVirtual())
            MO.setReg(0);
  MRI.clearVirtRegs();
}

void RepairAllocator::recomputePhysicalLiveness() {
  SmallVector<MachineBasicBlock *> BlocksToUpdate;
  for (MachineBasicBlock &MBB : reverse(MF)) {
    MBB.clearLiveIns();
    BlocksToUpdate.push_back(&MBB);
  }
  fullyRecomputeLiveIns(BlocksToUpdate);
  for (MachineBasicBlock &MBB : MF)
    recomputeLivenessFlags(MBB);
}

void RepairAllocator::enumeratePlans(MachineInstr &MI, const RepairPlan &C,
                                     function_ref<void(RepairPlan)> Accept) {
  if (MI.isPHI()) {
    RepairPlan P{C.State, {}, {}, C.Cost};
    Register R = MI.getOperand(0).getReg();
    if (!MRI.use_nodbg_empty(R))
      for (unsigned Lane = 0; Lane < numLanes(R); ++Lane)
        P.State.define(backingIndex(R, Lane), value(R, Lane));
    Accept(std::move(P));
    return;
  }
  if ((TII.isTriviallyReMaterializable(MI) &&
       MI.getOperand(0).getReg().isVirtual()) ||
      (MI.isCopy() && MI.getOperand(0).getReg().isVirtual() &&
       rematerialization(value(MI.getOperand(0).getReg())) &&
       (numLanes(MI.getOperand(0).getReg()) == 1 ||
        rematerialization(value(MI.getOperand(0).getReg(), 1))))) {
    RepairPlan P{C.State, {}, {}, C.Cost};
    Accept(std::move(P));
    return;
  }
  OperandSearch(*this, MI, C, Accept).run();
}

RepairAllocator::OperandSearch::OperandSearch(
    RepairAllocator &Allocator, MachineInstr &MI, const RepairPlan &Incoming,
    function_ref<void(RepairPlan)> Accept)
    : Allocator(Allocator), MI(MI), Incoming(Incoming), Accept(Accept),
      Domains(MI.getNumOperands()), Assignment(MI.getNumOperands()) {}

void RepairAllocator::OperandSearch::run() {
  for (unsigned I = 0; I < MI.getNumOperands(); ++I)
    buildDomain(I);
  search(0);
}

void RepairAllocator::OperandSearch::buildDomain(unsigned I) {
  const MachineOperand &MO = MI.getOperand(I);
  if (!MO.isReg() || !MO.getReg().isVirtual()) {
    Domains[I].push_back(MO.isReg() ? MCPhysReg(MO.getReg()) : 0);
    return;
  }
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

void RepairAllocator::OperandSearch::addCopies(unsigned OpIdx, Register R) {
  for (MCPhysReg Location : Allocator.copies(R, Incoming.State))
    addDestination(OpIdx, Location);
}

void RepairAllocator::OperandSearch::addDestination(unsigned OpIdx,
                                                    MCPhysReg R) {
  if (!llvm::is_contained(Domains[OpIdx], R))
    Domains[OpIdx].push_back(R);
}

bool RepairAllocator::OperandSearch::isLegal(unsigned I, MCPhysReg R) const {
  const MachineOperand &MO = MI.getOperand(I);
  if (MI.isCopy() || MI.getOpcode() == TargetOpcode::REG_SEQUENCE ||
      !MO.isReg())
    return true;
  for (unsigned J = 0; J < I; ++J) {
    const MachineOperand &Other = MI.getOperand(J);
    if (!Other.isReg())
      continue;
    // Fixed operands already describe the instruction, including redundant
    // implicit aliases. Only choices for virtual operands need testing here.
    if (MO.getReg().isPhysical() && Other.getReg().isPhysical())
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

bool RepairAllocator::OperandSearch::isTiedUseAlias(
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

void RepairAllocator::OperandSearch::search(unsigned I) {
  if (I == MI.getNumOperands()) {
    RepairPlan Plan{Incoming.State, {}, Assignment, Incoming.Cost};
    if (InstructionRepair(Allocator, MI, Plan).run())
      Accept(std::move(Plan));
    return;
  }
  for (MCPhysReg R : Domains[I]) {
    if (!isLegal(I, R))
      continue;
    Assignment[I] = R;
    search(I + 1);
  }
}

RepairAllocator::InstructionRepair::InstructionRepair(
    RepairAllocator &Allocator, MachineInstr &MI, RepairPlan &Plan)
    : Allocator(Allocator), MI(MI), Plan(Plan),
      Liveness(Allocator.Liveness.find(&MI)->second),
      Locked(Liveness.FixedLiveIns) {}

bool RepairAllocator::InstructionRepair::run() {
  if (MI.isCopy() && (MI.getOperand(0).getReg().isVirtual() ||
                      MI.getOperand(1).getReg().isVirtual()))
    return repairCopy();
  if (MI.getOpcode() == TargetOpcode::REG_SEQUENCE)
    return repairRegSequence();

  // Memory and flag operands may require A/X/Y scratch. Lock each prepared
  // input before preparing the next, with A/X/Y inputs prepared last.
  if (!prepareUses(/*InGPRs=*/false) || !prepareUses(/*InGPRs=*/true))
    return false;
  if (!preserveLiveThroughValues() || !applyInstruction())
    return false;
  Plan.Cost += Allocator.instructionCost(MI, Plan.Operands);
  return Allocator.hasLiveValues(Plan.State, Liveness.LiveOuts);
}

bool RepairAllocator::InstructionRepair::repairCopy() {
  Register D = MI.getOperand(0).getReg(), S = MI.getOperand(1).getReg();
  auto Dst = Allocator.locationIndices(Plan.Operands[0]);
  if (Dst.empty())
    return false;
  if (S.isPhysical()) {
    auto Src = Allocator.locationIndices(S);
    if (Src.size() != Dst.size() || !D.isVirtual())
      return false;
    // Capture the existing physical bits without emitting a transfer when the
    // chosen destination is the source. The fixed live range protects them.
    for (unsigned I = 0; I < Src.size(); ++I) {
      if (!Allocator.evacuate(Src[I], Plan, Liveness.LiveIns, Locked,
                              BitVector(Allocator.Locations.size())))
        return false;
      Plan.State.define(Src[I], Allocator.value(D, I));
    }
    for (unsigned I = 0; I < Dst.size(); ++I)
      if (!Allocator.ensure(Allocator.value(D, I), Dst[I], Plan,
                            Liveness.LiveIns, Locked))
        return false;
  } else {
    for (unsigned I = 0; I < Dst.size(); ++I) {
      if (!Allocator.ensure(Allocator.value(S, I), Dst[I], Plan,
                            Liveness.LiveIns, Locked))
        return false;
      Locked.set(Dst[I]);
    }
    if (D.isVirtual() && !Allocator.sameValue(D, S))
      Allocator.fail("COPY lacks whole-value ancestry", &MI);
  }
  return Allocator.hasLiveValues(Plan.State, Liveness.LiveOuts);
}

bool RepairAllocator::InstructionRepair::repairRegSequence() {
  auto Dst = Allocator.locationIndices(Plan.Operands[0]);
  if (Dst.size() != 2)
    return false;
  for (unsigned I = 1; I < MI.getNumOperands(); I += 2) {
    unsigned Lane = MI.getOperand(I + 1).getImm() == MOS::sublo ? 0 : 1;
    Register S = MI.getOperand(I).getReg();
    if (!Allocator.ensure(Allocator.value(S), Dst[Lane], Plan, Liveness.LiveIns,
                          Locked))
      return false;
    Locked.set(Dst[Lane]);
  }
  return Allocator.hasLiveValues(Plan.State, Liveness.LiveOuts);
}

bool RepairAllocator::InstructionRepair::prepareUses(bool InGPRs) {
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse() || MO.isUndef() || !MO.getReg().isVirtual())
      continue;
    auto U = Allocator.locationIndices(Plan.Operands[I]);
    if (U.empty() || Allocator.isGPR(U.front()) != InGPRs)
      continue;
    for (unsigned Lane = 0; Lane < U.size(); ++Lane) {
      if (!Allocator.ensure(Allocator.value(MO.getReg(), Lane), U[Lane], Plan,
                            Liveness.LiveIns, Locked))
        return false;
      Locked.set(U[Lane]);
    }
  }
  return true;
}

bool RepairAllocator::InstructionRepair::preserveLiveThroughValues() {
  // Evacuations must avoid every impending clobber, including destinations of
  // later operands. The small physical mask is shared by all evacuations.
  BitVector Clobbered(Allocator.Locations.size());
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (MO.isReg() && MO.isDef())
      for (unsigned U : Allocator.locationIndices(Plan.Operands[I]))
        Clobbered.set(U);
    if (MO.isRegMask())
      for (unsigned U = 0; U < Allocator.Locations.size(); ++U)
        if (MO.clobbersPhysReg(Allocator.Locations[U]))
          Clobbered.set(U);
  }
  for (int U : Clobbered.set_bits())
    if (!Allocator.evacuate(U, Plan, Liveness.LiveOuts, Locked, Clobbered))
      return false;
  return true;
}

bool RepairAllocator::InstructionRepair::applyInstruction() {
  // Repairs execute before MI, so preservation above protects every input.
  // Now simulate MI's effects. Killed inputs need no explicit erasure: their
  // copies remain usable until overwritten or forgotten at the end of the step.
  for (const MachineOperand &MO : MI.all_defs())
    if (MO.isEarlyClobber() && !define(MO))
      return false;
  for (const MachineOperand &MO : MI.operands())
    if (MO.isRegMask())
      for (unsigned L = 0; L < Allocator.Locations.size(); ++L)
        if (MO.clobbersPhysReg(Allocator.Locations[L]))
          Plan.State.clobber(L);
  for (const MachineOperand &MO : MI.all_defs())
    if (!MO.isEarlyClobber() && !define(MO))
      return false;
  return true;
}

bool RepairAllocator::InstructionRepair::define(const MachineOperand &MO) {
  auto Indices = Allocator.locationIndices(Plan.Operands[MO.getOperandNo()]);
  for (unsigned Lane = 0; Lane < Indices.size(); ++Lane) {
    unsigned L = Indices[Lane];
    if (MO.getReg().isPhysical()) {
      // Fixed liveness protects this result; a later COPY assigns its value
      // number when it captures the physical contents into an SSA register.
      Plan.State.clobber(L);
      continue;
    }
    bool TiedPhysicalUse =
        MO.isTied() && MI.getOperand(MI.findTiedOperandIdx(MO.getOperandNo()))
                           .getReg()
                           .isPhysical();
    if (Liveness.FixedLiveOuts[L] ||
        (MO.isEarlyClobber() && Liveness.FixedLiveIns[L] && !TiedPhysicalUse))
      return false;
    Plan.State.define(L, Allocator.value(MO.getReg(), Lane));
  }
  return true;
}

RegisterContents
RepairAllocator::backingContents(const SparseBitVector<> &LiveRegs) const {
  RegisterContents Backings(Locations.size());
  for (unsigned I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < numLanes(R); ++Lane) {
      ValueNumber V = value(R, Lane);
      if (rematerialization(V))
        continue;
      unsigned L = backingIndex(R, Lane);
      if (L == ~0u)
        fail("live value has no backing register");
      assert((!Backings.read(L) || Backings.read(L) == V) &&
             "backing assignments interfere");
      Backings.define(L, V);
    }
  }
  return Backings;
}

bool RepairAllocator::finishPlacement(RepairPlan &P,
                                      const RegisterContents &Backings,
                                      const SparseBitVector<> &LiveRegs,
                                      BitVector Locked, bool CanInsert) {
  if (!hasLiveValues(P.State, LiveRegs))
    return false;
  unsigned NumPreparationTransfers = P.Transfers.size();
  // A repair can displace a value into arbitrary imaginary scratch. Before
  // retaining the result, give every live lane either its backing copy or a
  // hardware copy. Protect established backing copies while doing so.
  while (true) {
    unsigned Missing = ~0u;
    for (unsigned L = HardwareRegs.size(); L < Backings.size(); ++L) {
      ValueNumber V = Backings.read(L);
      if (!V)
        continue;
      if (P.State.read(L) == V)
        Locked.set(L);
      else if (!P.State.hasHardwareCopy(V))
        Missing = L;
    }
    if (Missing == ~0u)
      break;
    // In particular, don't insert instructions after a branch to repair its
    // outgoing state. Its inputs were restored before the terminators.
    if (!CanInsert)
      return false;
    ValueNumber V = Backings.read(Missing);
    // Either destination establishes the invariant. Keeping a hardware copy
    // can avoid an unnecessary store, and also works when backing is pinned.
    unsigned BestLocation = ~0u;
    RepairPlan Best = P;
    if (ensure(V, Missing, Best, LiveRegs, Locked))
      BestLocation = Missing;
    for (unsigned L = 0; L < HardwareRegs.size(); ++L) {
      RepairPlan Trial = P;
      if (ensure(V, L, Trial, LiveRegs, Locked) &&
          (BestLocation == ~0u || Trial.Cost < Best.Cost)) {
        BestLocation = L;
        Best = std::move(Trial);
      }
    }
    if (BestLocation == ~0u)
      return false;
    P = std::move(Best);
    Locked.set(BestLocation);
  }
  for (unsigned I = NumPreparationTransfers; I < P.Transfers.size(); ++I)
    P.AfterTransfers.push_back(std::move(P.Transfers[I]));
  P.Transfers.resize(NumPreparationTransfers);
  forgetDeadValues(P.State, LiveRegs);
  return true;
}

bool RepairAllocator::restore(RepairPlan &P, const SparseBitVector<> &LiveRegs,
                              const SparseBitVector<> &Preserve,
                              BitVector Locked) {
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < numLanes(R); ++Lane) {
      ValueNumber A = value(R, Lane);
      if (rematerialization(A))
        continue;
      unsigned H = backingIndex(R, Lane);
      if (H == ~0u || !ensure(A, H, P, Preserve, Locked))
        return false;
      Locked.set(H);
    }
  }
  return true;
}

bool RepairAllocator::ensure(ValueNumber V, unsigned L, RepairPlan &P,
                             const SparseBitVector<> &LiveRegs,
                             BitVector Locked, const BitVector *Forbidden) {
  if (P.State.read(L) == V)
    return true;
  if (Locked[L])
    return false;
  BitVector Avoid = Forbidden ? *Forbidden : BitVector(Locations.size());
  RepairPlan Best;
  Best.Cost = ~0u;
  auto Consider = [&](RepairPlan Trial) {
    if (Trial.Cost < Best.Cost)
      Best = std::move(Trial);
  };
  RepairPlan Remat = P;
  if (rematerialize(V, L, Remat, LiveRegs, Locked, Avoid))
    Consider(std::move(Remat));

  if (!evacuate(L, P, LiveRegs, Locked, Avoid)) {
    if (Best.Cost == ~0u)
      return false;
    P = std::move(Best);
    return true;
  }
  Locked.set(L);
  for (unsigned S = 0; S < Locations.size(); ++S) {
    if (P.State.read(S) != V)
      continue;
    RepairPlan Trial = P;
    if (transfer(L, S, Trial))
      Consider(std::move(Trial));
  }

  // A byte of a rematerializable Imag16 may be wanted in hardware. Materialize
  // the Imag16 in imaginary storage first, then use the ordinary transfer path.
  if (!P.State.hasCopy(V) && numLanes(sourceReg(V)) == 2 &&
      rematerialization(V)) {
    unsigned Lane = (V - 1) % 2;
    for (MCPhysReg Imag16 : destinations(sourceReg(V))) {
      auto U = locationIndices(Imag16);
      if (U.size() != 2)
        continue;
      RepairPlan Trial = P;
      if (!rematerialize(V, U[Lane], Trial, LiveRegs, Locked, Avoid))
        continue;
      BitVector Unlocked = Locked;
      Unlocked.reset(L);
      if (ensure(V, L, Trial, LiveRegs, Unlocked, &Avoid))
        Consider(std::move(Trial));
    }
  }

  // Memory-to-memory and X/Y transfers need hardware scratch. Reuse the same
  // placement logic to obtain the value there, including rematerialization.
  // Each recursive step locks another location, bounding scratch construction.
  for (MCPhysReg R : {MOS::A, MOS::X, MOS::Y}) {
    unsigned H = LocationIndices.lookup(R);
    if (Locked[H])
      continue;
    RepairPlan Trial = P;
    if (ensure(V, H, Trial, LiveRegs, Locked, &Avoid) && transfer(L, H, Trial))
      Consider(std::move(Trial));
  }
  if (Best.Cost == ~0u)
    return false;
  P = std::move(Best);
  return true;
}

// Rematerialize the complete definition, even when only one byte was requested.
// The other byte is a clobber during preparation and an available copy
// afterward.
bool RepairAllocator::rematerialize(ValueNumber V, unsigned L, RepairPlan &P,
                                    const SparseBitVector<> &LiveRegs,
                                    BitVector Locked,
                                    const BitVector &Forbidden) {
  const MachineInstr *Def = rematerialization(V);
  if (!Def)
    return false;
  Register Source = sourceReg(V);
  MCPhysReg Dst = Locations[L];
  if (numLanes(Source) == 2) {
    unsigned Lane = (V - 1) % 2;
    Dst = TRI.getMatchingSuperReg(Dst, Lane ? MOS::subhi : MOS::sublo,
                                  &MOS::Imag16RegClass);
    if (!Dst)
      return false;
  }
  const TargetRegisterClass *RC = Def->getRegClassConstraint(0, &TII, &TRI);
  if (!RC)
    RC = MRI.getRegClass(Source);
  if (!RC->contains(Dst))
    return false;

  auto U = locationIndices(Dst);
  if (U.size() != numLanes(Source))
    return false;
  BitVector Clobbered = Forbidden;
  for (unsigned Location : U) {
    if (Locked[Location])
      return false;
    Clobbered.set(Location);
  }
  for (unsigned Location : U)
    if (!evacuate(Location, P, LiveRegs, Locked, Clobbered))
      return false;
  SmallVector<MCPhysReg> Ops(Def->getNumOperands());
  Ops[0] = Dst;
  P.Transfers.push_back({0, Dst, 0, Def});
  P.Cost += instructionCost(*Def, Ops);
  for (unsigned Lane = 0; Lane < U.size(); ++Lane)
    P.State.define(U[Lane], value(Source, Lane));
  return true;
}

bool RepairAllocator::evacuate(unsigned L, RepairPlan &P,
                               const SparseBitVector<> &LiveRegs,
                               BitVector Locked, const BitVector &Forbidden) {
  ValueNumber A = P.State.read(L);
  if (!isLiveValue(A, LiveRegs) || rematerialization(A))
    return true;
  for (unsigned I = 0; I < Locations.size(); ++I)
    if (I != L && !Forbidden[I] && P.State.read(I) == A)
      return true;
  Locked.set(L);
  // Any live alias's backing register can preserve these contents. Do not use
  // the ultimate source's backing assignment after its own live range has
  // ended.
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < numLanes(R); ++Lane) {
      if (value(R, Lane) != A)
        continue;
      unsigned H = backingIndex(R, Lane);
      if (H == ~0u || Locked[H] || Forbidden[H])
        continue;
      RepairPlan Trial = P;
      if (ensure(A, H, Trial, LiveRegs, Locked, &Forbidden)) {
        P = std::move(Trial);
        return true;
      }
    }
  }
  // Only an unavailable backing register requires an arbitrary imaginary
  // destination. Choose empty storage on demand; Imag16 lanes can be evacuated
  // separately.
  for (int T : ScratchLocations.set_bits()) {
    if (Locked[T] || Forbidden[T] || isLiveValue(P.State.read(T), LiveRegs))
      continue;
    RepairPlan Trial = P;
    if (ensure(A, T, Trial, LiveRegs, Locked, &Forbidden)) {
      P = std::move(Trial);
      return true;
    }
  }
  return false;
}

bool RepairAllocator::transfer(unsigned D, unsigned S, RepairPlan &P) {
  MCPhysReg Dst = Locations[D], Src = Locations[S];
  if (D == S)
    return true;
  unsigned Opcode, Cost;
  if (isGPR(D) && isImaginary(S)) {
    Opcode = MOS::LDImag8;
    Cost = 2;
  } else if (isImaginary(D) && isGPR(S)) {
    Opcode = MOS::STImag8;
    Cost = 2;
  } else if (isGPR(D) && isGPR(S) && (Dst == MOS::A || Src == MOS::A)) {
    Opcode = Src == MOS::A ? MOS::TA : MOS::T_A;
    Cost = 1;
  } else
    return false;
  P.Transfers.push_back({Opcode, Dst, Src});
  P.Cost += Cost;
  P.State.copy(D, S);
  return true;
}

void RepairAllocator::emitTransfers(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator At,
                                    const DebugLoc &DL,
                                    ArrayRef<Transfer> Transfers) {
  for (const Transfer &T : Transfers) {
    if (T.Rematerialization)
      TII.reMaterialize(MBB, At, T.Dst, 0, *T.Rematerialization);
    else
      BuildMI(MBB, At, DL, TII.get(T.Opcode), T.Dst).addReg(T.Src);
    ++NumTransfers;
  }
}

// The input opcode is fixed, but some pseudos expand differently depending on
// placement. Account for the byte costs relevant to sieve; the remaining fixed
// opcode costs do not affect the choice between placements of that operation.
unsigned RepairAllocator::instructionCost(const MachineInstr &MI,
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

SmallVector<MCPhysReg> RepairAllocator::destinations(Register R) const {
  SmallVector<MCPhysReg> Result;
  // Hardware members of an operand class are meaningful instruction choices.
  // Ordinary values use their backing register or a concrete constraint for
  // imaginary storage. Rematerializable values have no backing assignment, so
  // constrained uses may need to acquire temporary storage here.
  if (numLanes(R) == 1) {
    if (TRI.getRegSizeInBits(*MRI.getRegClass(R)) == 1) {
      Result.push_back(MOS::C);
      Result.push_back(MOS::V);
    } else
      for (MCPhysReg H : {MOS::A, MOS::X, MOS::Y})
        Result.push_back(H);
  }
  if (MCPhysReg H = VRM.getPhys(R))
    Result.push_back(H);
  else if (rematerialization(value(R))) {
    auto Order = RCI.getOrder(numLanes(R) == 2 ? &MOS::Imag16RegClass
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
RepairAllocator::copies(Register R, const RegisterContents &S) const {
  SmallVector<MCPhysReg> Result;
  ValueNumber V = value(R);
  if (!V)
    return Result;
  for (unsigned L = 0; L < Locations.size(); ++L) {
    if (S.read(L) != V)
      continue;
    if (numLanes(R) == 1) {
      Result.push_back(Locations[L]);
      continue;
    }
    MCPhysReg Imag16 =
        TRI.getMatchingSuperReg(Locations[L], MOS::sublo, &MOS::Imag16RegClass);
    if (!Imag16)
      continue;
    auto Hi = LocationIndices.find(TRI.getSubReg(Imag16, MOS::subhi));
    if (Hi != LocationIndices.end() && S.read(Hi->second) == value(R, 1))
      Result.push_back(Imag16);
  }
  return Result;
}

void RepairAllocator::forgetDeadValues(
    RegisterContents &S, const SparseBitVector<> &LiveRegs) const {
  for (unsigned L = 0; L < S.size(); ++L)
    if (!isLiveValue(S.read(L), LiveRegs))
      S.clobber(L);
}

bool RepairAllocator::hasLiveValues(const RegisterContents &S,
                                    const SparseBitVector<> &LiveRegs) const {
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < numLanes(R); ++Lane)
      if (!rematerialization(value(R, Lane)) && !S.hasCopy(value(R, Lane)))
        return false;
  }
  return true;
}

bool RepairAllocator::isLiveValue(ValueNumber V,
                                  const SparseBitVector<> &LiveRegs) const {
  if (!V)
    return false;
  for (int I : LiveRegs) {
    Register R = Register::index2VirtReg(I);
    for (unsigned Lane = 0; Lane < numLanes(R); ++Lane)
      if (value(R, Lane) == V)
        return true;
  }
  return false;
}

bool RepairAllocator::sameValue(Register A, Register B) const {
  if (A == B)
    return true;
  if (!A.isVirtual() || !B.isVirtual() || numLanes(A) != numLanes(B))
    return false;
  for (unsigned Lane = 0; Lane < numLanes(A); ++Lane)
    if (value(A, Lane) != value(B, Lane))
      return false;
  return true;
}

SparseBitVector<> RepairAllocator::liveOuts(MachineBasicBlock &MBB) const {
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

void RepairAllocator::addLocation(MCPhysReg R) {
  if (MOS::Imag16RegClass.contains(R)) {
    addLocation(TRI.getSubReg(R, MOS::sublo));
    addLocation(TRI.getSubReg(R, MOS::subhi));
  } else if (R && !LocationIndices.count(R)) {
    LocationIndices[R] = Locations.size();
    Locations.push_back(R);
  }
}

void RepairAllocator::addConstraint(Register R, MCPhysReg Location) {
  auto &C = Constraints[VRM.getOriginal(R)];
  if (!llvm::is_contained(C, Location))
    C.push_back(Location);
}

unsigned RepairAllocator::numLanes(Register R) const {
  return TRI.getRegSizeInBits(*MRI.getRegClass(R)) == 16 ? 2 : 1;
}

ValueNumber RepairAllocator::value(Register R, unsigned Lane) const {
  return ValueNumbers[sourceValue(R, Lane)];
}

SmallVector<unsigned, 2> RepairAllocator::locationIndices(MCPhysReg R) const {
  if (MOS::Imag16RegClass.contains(R))
    return {LocationIndices.lookup(TRI.getSubReg(R, MOS::sublo)),
            LocationIndices.lookup(TRI.getSubReg(R, MOS::subhi))};
  auto I = LocationIndices.find(R);
  if (I == LocationIndices.end())
    return {};
  return {I->second};
}

unsigned RepairAllocator::backingIndex(Register R, unsigned Lane) const {
  MCPhysReg BackingReg = VRM.getPhys(R);
  if (!BackingReg)
    return ~0u;
  return locationIndices(BackingReg)[Lane];
}

bool RepairAllocator::isGPR(unsigned L) const {
  return MOS::GPRRegClass.contains(Locations[L]);
}

bool RepairAllocator::isImaginary(unsigned L) const {
  return MOS::Imag8RegClass.contains(Locations[L]);
}

const MachineInstr *RepairAllocator::rematerialization(ValueNumber V) const {
  if (!V)
    return nullptr;
  const MachineInstr *Def = MRI.getVRegDef(sourceReg(V));
  return Def && TII.isTriviallyReMaterializable(*Def) ? Def : nullptr;
}

void RepairAllocator::fail(const Twine &Reason, const MachineInstr *MI) const {
  errs() << "MOSRegAlloc: " << Reason << " in " << MF.getName() << '\n';
  if (MI)
    errs() << *MI;
  report_fatal_error("unsupported MOS register repair", false);
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
