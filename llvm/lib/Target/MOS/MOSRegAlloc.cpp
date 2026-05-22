//===-- MOSRegAlloc.cpp - MOS Register Allocation -------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Heuristic combined scheduling + register allocation for the MOS 6502.
//
// Implements the cluster-merging algorithm: each instruction starts as its own
// cluster, and clusters are iteratively merged via insertion until a single
// cluster remains. The final cluster's schedule is the block's instruction
// sequence. Physical registers are assigned top-down greedily.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MOS.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

using MBBIterator = MachineBasicBlock::iterator;
using LiveSet = SmallSet<Register, 8>;

/// A contiguous range of instructions in the MBB, representing a group of
/// instructions that have been scheduled together.
struct Cluster {
  explicit Cluster(iterator_range<MBBIterator> Range) : Range(Range) {}

  iterator_range<MBBIterator> Range;
  /// Registers live at the end of this cluster's allocation.
  LiveSet LiveOut;

  bool empty() const { return Range.empty(); }
  MBBIterator begin() const { return Range.begin(); }
  MBBIterator end() const { return Range.end(); }
};

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setNoVRegs().setNoPHIs();
  }

  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineRegisterInfo *MRI = nullptr;
  const TargetRegisterInfo *TRI = nullptr;
  RegisterClassInfo RCI;

  void validate(MachineBasicBlock &MBB);

  /// All clusters; indexed by cluster ID. Emptied clusters have empty Range.
  /// Cluster 0 is the livein cluster: an empty cluster whose LiveOut
  /// publishes the block's livein physregs to consumers via merging.
  static constexpr unsigned LiveInClusterIdx = 0;
  SmallVector<Cluster, 0> Clusters;
  /// Maps each instruction to the ID of the cluster that contains it.
  DenseMap<MachineInstr *, unsigned> MICluster;
  void initClusters(MachineBasicBlock &MBB);

  /// Effective register sets accumulated across all allocations. Indexed by
  /// virtual register. Persists past V's death because narrowing chains may
  /// later revisit V (an active value running short on slack can shed an
  /// already-dead adversary's contribution by narrowing it). Used by
  /// assignRegisters to pick physregs.
  IndexedMap<BitVector, VirtReg2IndexFunctor> EffectiveRC;
  /// Per-vreg interferences: the set of values (vregs or physregs) live at
  /// V's definition. V's total interference (and thus slack) is computed on
  /// demand by summing each interferer's contribution against V's current
  /// EffectiveRC — so any later narrowing of an interferer is reflected
  /// automatically. Persists past V's death (chained narrowing may revisit
  /// V).
  IndexedMap<SmallVector<Register, 8>, VirtReg2IndexFunctor> Interferences;

  /// Maps a tied def D to the root vreg of its slot — i.e. the original tied
  /// use's vreg that owns the slot's EffectiveRC / Interferences / Live entry.
  /// Always stores the resolved root, so lookups are one level deep.
  DenseMap<Register, Register> TiedRoot;
  Register tiedRoot(Register R) const {
    if (R.isPhysical())
      return R;
    auto It = TiedRoot.find(R);
    return It == TiedRoot.end() ? R : It->second;
  }

  void initKillDeadFlags(MachineBasicBlock &MBB);
  void normalizeTiedRegs(MachineBasicBlock &MBB);

  void allocate(iterator_range<MBBIterator> MIs, LiveSet &Live);
  void narrowToFit(Register Def, const BitVector &DefEff,
                   ArrayRef<Register> DefInters, const LiveSet &Live);
  int getSlack(const BitVector &EffRC, ArrayRef<Register> Inters) const {
    int Slack = EffRC.count();
    for (Register U : Inters)
      Slack -= maxInterference(EffRC, U);
    return Slack;
  }
  /// Worst-case number of EffRC's registers that adversary U can block when
  /// U is colored. U is colored to one register from its own effective set
  /// (the singleton {U} for a physreg, or EffectiveRC[U] for a vreg); the
  /// answer is the max over that set of |{r ∈ EffRC : aliases U's color}|.
  unsigned maxInterference(const BitVector &EffRC, Register U) const;
  BitVector allocatableRegs(const TargetRegisterClass *RC);
  BitVector aliasSet(const BitVector &Regs);

  void schedule(MachineBasicBlock &MBB);
  void schedulePhysRegs(MachineBasicBlock &MBB);
  void scheduleVRegs(MachineBasicBlock &MBB);
  void linearizeClusters(MachineBasicBlock &MBB);
  void mergeClusters(MachineBasicBlock &MBB, unsigned DefC, unsigned UseC);
  void setKillFlags(iterator_range<MBBIterator> MIs, unsigned ClusterIdx);

  void assignRegisters(MachineBasicBlock &MBB);
};

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  MRI = &MF.getRegInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  RCI.runOnMachineFunction(MF);

  if (MRI->getNumVirtRegs() == 0)
    return false;

  LLVM_DEBUG(dbgs() << "MOS RegAlloc: " << MF.getName() << " ("
                    << MRI->getNumVirtRegs() << " vregs)\n");

  assert(MF.size() == 1 && "Multiple basic blocks not yet supported");
  MachineBasicBlock &MBB = MF.front();
  validate(MBB);
  initKillDeadFlags(MBB);
  normalizeTiedRegs(MBB);
  initClusters(MBB);
  schedule(MBB);
  assignRegisters(MBB);
  MRI->clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

void MOSRegAlloc::validate(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    assert(MI.getOpcode() != TargetOpcode::REG_SEQUENCE &&
           "REG_SEQUENCE not yet supported");
    assert(MI.getOpcode() != TargetOpcode::INSERT_SUBREG &&
           "INSERT_SUBREG not yet supported");
    assert(MI.getOpcode() != TargetOpcode::EXTRACT_SUBREG &&
           "EXTRACT_SUBREG not yet supported");

    for (const MachineOperand &MO : MI.operands()) {
      assert(!MO.isRegMask() && "Regmasks not yet supported");
      if (!MO.isReg())
        continue;
      assert(!MO.isEarlyClobber() && "Earlyclobber not yet supported");
      // No vreg defs in terminators: this makes any tied operand pair on
      // a terminator necessarily physreg-tied, so normalizeTiedRegs never
      // has to insert a COPY inside the terminator span.
      if (MI.isTerminator() && MO.isDef())
        assert(!MO.getReg().isVirtual() &&
               "Vreg defs on terminators not yet supported");
    }
  }
}

/// Reset register flags to a trustworthy baseline. Incoming kill flags
/// describe the input instruction order; scheduling reorders instructions,
/// so they cannot be trusted and are cleared. The pass instead sets each
/// kill flag itself at the moment the kill fact is established:
/// normalizeTiedRegs marks tied uses (single-use by construction), and
/// setKillFlags marks each value's last use once a merge gathers all its
/// uses into one cluster — a fact no later merge can falsify, since merges
/// only append. Deadness, by contrast, is order-stable — a register with no
/// uses — so it is derived and set once here.
void MOSRegAlloc::initKillDeadFlags(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    for (MachineOperand &MO : MI.all_uses())
      MO.setIsKill(false);
    for (MachineOperand &MO : MI.all_defs()) {
      if (MO.isDead())
        continue;
      Register R = MO.getReg();
      // A def is dead iff nothing reads it. A register read nowhere —
      // including through an aliasing register (a use of $p reads $c) — has
      // only dead defs under any schedule; this is the one order-independent
      // physreg deadness fact. Finer deadness is per-def (a def unread before
      // the next write is dead even if the register has other uses) and thus
      // schedule-dependent. That case is conservatively missed, consistent
      // with the kill handling, which is also register-keyed: a physreg
      // defined more than once in the block is treated as a single value,
      // live from its first def to its last use. Over-extending liveness this
      // way can cost colorability, but never correctness.
      bool NoUses = MRI->use_nodbg_empty(R);
      if (R.isPhysical())
        for (MCRegAliasIterator AI(R.asMCReg(), TRI, /*IncludeSelf=*/false);
             AI.isValid() && NoUses; ++AI)
          NoUses = MRI->use_nodbg_empty(*AI);
      if (NoUses)
        MO.setIsDead(true);
    }
  }
}

/// Normalize tied uses so the allocator's core sees every tie as
/// single-use + class-matching the tied def. For each tied use that is
/// either multi-use or whose vreg class differs from the tied def's,
/// insert `V' = COPY V` before MI and rewire the tied use to V'. V' is
/// given the tied def's register class, so after this pass every tied
/// pair has matching classes and the tied use is single-use. That makes
/// every tied use its value's last use by construction, so its kill flag
/// is set here, at the moment the invariant is established.
void MOSRegAlloc::normalizeTiedRegs(MachineBasicBlock &MBB) {
  const TargetInstrInfo *TII = MBB.getParent()->getSubtarget().getInstrInfo();
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    for (unsigned I = 0, E = MI.getNumOperands(); I != E; ++I) {
      MachineOperand &MO = MI.getOperand(I);
      if (!MO.isReg() || !MO.isUse() || !MO.isTied())
        continue;
      Register V = MO.getReg();
      // Tied physreg pairs need no normalization: the slot is just
      // "this physreg stays live through the MI."
      if (V.isPhysical())
        continue;

      unsigned DefIdx = MI.findTiedOperandIdx(I);
      const TargetRegisterClass *VRC = MRI->getRegClass(V);
      const TargetRegisterClass *DRC =
          MRI->getRegClass(MI.getOperand(DefIdx).getReg());

      // No COPY needed only when V already matches the def's class and is
      // single-use (tied destruction lands on its natural last use).
      if (VRC == DRC && MRI->hasOneNonDBGUse(V)) {
        MO.setIsKill(true);
        continue;
      }

      // V's own kill is established later, once a merge gathers all its
      // uses, so the COPY's use of V carries no flag.
      Register VPrime = MRI->createVirtualRegister(DRC);
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(TargetOpcode::COPY), VPrime)
          .addReg(V);
      MO.setReg(VPrime);
      MO.setIsKill(true);
    }
  }
}

/// Create one singleton cluster per instruction and allocate it. Cluster 0
/// (LiveInClusterIdx) is an empty cluster whose LiveOut publishes the
/// block's livein physregs; consumers of liveins are merged into it later
/// by schedule, exactly like any other physreg def/use.
void MOSRegAlloc::initClusters(MachineBasicBlock &MBB) {
  Clusters.clear();
  MICluster.clear();
  EffectiveRC.clear();
  EffectiveRC.resize(MRI->getNumVirtRegs());
  Interferences.clear();
  Interferences.resize(MRI->getNumVirtRegs());
  TiedRoot.clear();

  Cluster LiveIns(make_range(MBB.begin(), MBB.begin()));
  for (const auto &LI : MBB.liveins())
    LiveIns.LiveOut.insert(LI.PhysReg);
  Clusters.push_back(std::move(LiveIns));

  auto FirstTerm = MBB.getFirstTerminator();
  for (auto It = MBB.begin(); It != FirstTerm; ++It) {
    MachineInstr &MI = *It;
    MICluster[&MI] = Clusters.size();
    Cluster C(make_range(MI.getIterator(), std::next(MI.getIterator())));
    allocate(C.Range, C.LiveOut);
    Clusters.push_back(std::move(C));
  }

  // Terminator span: one atomic cluster, never split. Its contiguity is
  // load-bearing — nothing may be inserted between its MIs.
  if (FirstTerm != MBB.end()) {
    Cluster Term(make_range(FirstTerm, MBB.end()));
    for (MachineInstr &MI : Term)
      MICluster[&MI] = Clusters.size();
    allocate(Term.Range, Term.LiveOut);
    Clusters.push_back(std::move(Term));
  }
}

/// Allocate register space for instructions in the given range. For each new
/// def, snapshots the live set as the def's interferences and ensures
/// slack >= 1, narrowing live values out of the new def's alias set until it
/// recovers. Updates this->EffectiveRC and this->Interferences; Live tracks
/// the currently-live vreg set.
///
/// Slack model: each vreg V's interferences (the set of vregs live at V's
/// def) are fixed at V's def. V's slack is computed on demand by getSlack
/// against the current EffectiveRC of each interferer, so any later
/// narrowing of an interferer is reflected automatically. Top-down
/// assignment colors V before any later def, so later defs are constrained
/// to avoid V; they do not consume V's slack. Narrowing of V is the one
/// event that charges V — it directly shrinks EffectiveRC[V].
void MOSRegAlloc::allocate(iterator_range<MBBIterator> MIs, LiveSet &Live) {
  for (MachineInstr &MI : MIs) {
    for (const MachineOperand &MO : MI.all_uses()) {
      // Kill flags are trustworthy here: incoming ones were cleared
      // (initKillDeadFlags), and the pass sets one only as each kill fact is
      // established (normalizeTiedRegs, setKillFlags).
      if (!MO.isKill() || MO.isUndef())
        continue;
      // Tied uses don't end the slot — the tied def reuses it. The def loop
      // below records the new vreg's identity in TiedRoot. For other uses,
      // Live holds the slot's root, which may be the tied use's vreg behind
      // an intermediate tied def — redirect via tiedRoot.
      if (!MO.isTied())
        Live.erase(tiedRoot(MO.getReg()));
    }

    for (const MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();
      if (MO.isTied()) {
        if (R.isPhysical())
          continue;  // Slot is just "physreg stays live through MI."
        unsigned UseIdx = MI.findTiedOperandIdx(MO.getOperandNo());
        const MachineOperand &UseMO = MI.getOperand(UseIdx);
        assert(UseMO.getReg().isVirtual() &&
               "Tied physreg operands not supported");
        assert(UseMO.isKill() && "Tied use must be killed (normalizeTiedRegs)");
        assert(MRI->getRegClass(R) == MRI->getRegClass(UseMO.getReg()) &&
               "Tied operands must have matching register classes");
        TiedRoot[R] = tiedRoot(UseMO.getReg());
        LLVM_DEBUG(dbgs() << "    Tied def " << printReg(R, TRI) << " -> slot "
                          << printReg(TiedRoot[R], TRI) << "\n");
        continue;
      }
      if (R.isVirtual()) {
        EffectiveRC[R] = allocatableRegs(MRI->getRegClass(R));
        Interferences[R].assign(Live.begin(), Live.end());
        Live.insert(R);
        narrowToFit(R, EffectiveRC[R], Interferences[R], Live);

        LLVM_DEBUG(dbgs() << "    Slack for " << printReg(R, TRI) << " ("
                          << TRI->getRegClassName(MRI->getRegClass(R))
                          << "): " << getSlack(EffectiveRC[R], Interferences[R])
                          << " (eff=" << EffectiveRC[R].count() << ")\n");
      } else {
        // Physreg def: nothing to store in EffectiveRC / Interferences (a
        // physreg has a fixed singleton effective and no allocation choice).
        // Run the same colorability check against scratch values so that any
        // live vreg overlapping R is narrowed out of its alias set.
        BitVector PhysEff(TRI->getNumRegs());
        PhysEff.set(R);
        SmallVector<Register, 8> PhysInters(Live.begin(), Live.end());
        Live.insert(R);
        narrowToFit(R, PhysEff, PhysInters, Live);

        LLVM_DEBUG(dbgs() << "    Physreg def " << printReg(R, TRI) << "\n");
      }
    }

    // Dead defs' slots don't extend past this MI. Run after the full def
    // loop so intra-MI later defs still saw the dead def as an interferer.
    // Redirect through tiedRoot for tied + dead defs (rare): the slot's
    // root is the original tied use's vreg, not the tied def itself.
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.isDead())
        Live.erase(tiedRoot(MO.getReg()));
  }
}

/// Ensure Def is colorable given DefEff and DefInters by repeatedly narrowing
/// overlapping live vregs out of Def's alias set. Def is the new value (vreg
/// or physreg); DefEff is its effective register set; DefInters is the set
/// of values live at Def's definition.
void MOSRegAlloc::narrowToFit(Register Def, const BitVector &DefEff,
                              ArrayRef<Register> DefInters,
                              const LiveSet &Live) {
  // Slack < 1 means Def's worst-case interference has eaten its whole
  // effective set: we can't guarantee a free register. Narrow live vregs
  // out of Def's alias set until it recovers, picking each victim so that
  // it also remains colorable after losing those registers.
  while (getSlack(DefEff, DefInters) < 1) {
    BitVector RAlias = aliasSet(DefEff);

    // Pick the smallest V whose post-narrow effective set
    // EffectiveRC[V] \ RAlias is both:
    //   - strictly smaller than EffectiveRC[V] (so narrowing helps Def),
    //   - large enough to keep V colorable.
    // Apply the colorability filter before the tie-break: picking the
    // smallest V outright could land on a victim that loses colorability
    // when a slightly larger V could absorb the narrow safely.
    Register NarrowReg;
    BitVector NarrowEffRC;
    for (Register V : Live) {
      if (V == Def)
        continue;
      if (V.isPhysical())
        continue; // physregs have a fixed singleton effective; not narrowable
      BitVector NewEffRC = EffectiveRC[V];
      NewEffRC.reset(RAlias);
      if (NewEffRC == EffectiveRC[V])
        continue;
      if (getSlack(NewEffRC, Interferences[V]) < 1)
        continue;
      if (!NarrowReg.isValid() || V < NarrowReg) {
        NarrowReg = V;
        NarrowEffRC = std::move(NewEffRC);
      }
    }
    assert(NarrowReg.isValid() &&
           "Allocation failed: no colorably-narrowable live value");

    // V's interferences set stays as-is. Its slack drops naturally since
    // EffectiveRC[V] shrinks. Def's slack rises on the next getSlack call:
    // V's contribution to Def's interference is recomputed against the new
    // (smaller, RAlias-disjoint) EffectiveRC[V] and goes to 0. Any other
    // vreg that has V in its interferences is updated the same way the
    // next time its slack is queried.
    EffectiveRC[NarrowReg] = std::move(NarrowEffRC);

    LLVM_DEBUG(dbgs() << "    Narrow " << printReg(NarrowReg, TRI) << " to "
                      << EffectiveRC[NarrowReg].count() << " regs (slack "
                      << getSlack(EffectiveRC[NarrowReg],
                                  Interferences[NarrowReg])
                      << ")\n");
  }
}

/// Build a BitVector of allocatable physical registers for a register class.
BitVector MOSRegAlloc::allocatableRegs(const TargetRegisterClass *RC) {
  BitVector BV(TRI->getNumRegs());
  for (MCPhysReg R : RCI.getOrder(RC))
    BV.set(R);
  return BV;
}

unsigned MOSRegAlloc::maxInterference(const BitVector &EffRC,
                                      Register U) const {
  if (U.isPhysical()) {
    unsigned Blocked = 0;
    for (unsigned R : EffRC.set_bits())
      if (TRI->regsOverlap(MCPhysReg(R), U.asMCReg()))
        Blocked++;
    return Blocked;
  }
  unsigned Max = 0;
  for (unsigned R : EffectiveRC[U].set_bits())
    Max = std::max(Max, maxInterference(EffRC, Register(MCPhysReg(R))));
  return Max;
}

/// Compute the alias set of a register set: all physical registers that
/// overlap with any register in the input set.
BitVector MOSRegAlloc::aliasSet(const BitVector &Regs) {
  BitVector Result(TRI->getNumRegs());
  for (unsigned R : Regs.set_bits())
    for (MCRegAliasIterator AI(MCPhysReg(R), TRI, /*IncludeSelf=*/true);
         AI.isValid(); ++AI)
      Result.set(*AI);
  return Result;
}

void MOSRegAlloc::schedule(MachineBasicBlock &MBB) {
  schedulePhysRegs(MBB);
  scheduleVRegs(MBB);
  linearizeClusters(MBB);

  assert(llvm::count_if(Clusters,
                        [](const Cluster &C) { return !C.empty(); }) == 1 &&
         "Not all clusters merged into one");

  // MBB is now in final schedule order.
  LLVM_DEBUG({
    dbgs() << "  Final schedule:\n";
    for (MachineInstr &MI : MBB)
      dbgs() << "    " << MI;
  });
}

/// Walk MBB in original order, tracking which cluster currently "owns" each
/// live physreg. Liveins start owned by the livein cluster; physreg defs
/// update ownership to the defining instruction's cluster. Each physreg
/// use merges the using instruction's cluster into its owner. This handles
/// liveins, implicit-defs, and explicit physreg defs uniformly.
///
/// TODO: Physreg def-use bindings are fixed by the input order: a use must
/// stay after its reaching def and before the next write of any aliasing
/// register. That is preserved here only by construction — this sweep runs
/// in input order and merges each use into its owner immediately, sealing
/// def+use groups before any other reordering — not by any modeled
/// constraint. A scheduler that picks merges in a different order must
/// enforce these barriers explicitly; allocate validates only defs, so
/// nothing stops an unrelated merge from relocating a physreg use ahead of
/// its def.
void MOSRegAlloc::schedulePhysRegs(MachineBasicBlock &MBB) {
  DenseMap<MCPhysReg, unsigned> PhysRegOwner;
  for (const auto &LI : MBB.liveins())
    PhysRegOwner[LI.PhysReg] = LiveInClusterIdx;

  // Each merge here only relocates MI itself (its cluster is still a
  // singleton when we visit it), so MI's original successor is untouched.
  // make_early_inc_range captures the successor before the merge runs.
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
    // Terminators live in the terminator cluster — an atomic cluster
    // anchored at MBB.end(). Driving a merge from a terminator's physreg
    // use would splice the terminator cluster mid-block; linearizeClusters
    // folds it in last instead.
    if (MI.isTerminator())
      continue;
    for (const MachineOperand &MO : MI.all_uses()) {
      if (!MO.getReg().isPhysical())
        continue;
      auto It = PhysRegOwner.find(MO.getReg().asMCReg());
      assert(It != PhysRegOwner.end() &&
             "Physreg use has no prior def or livein");
      unsigned DefC = It->second;
      unsigned UseC = MICluster[&MI];
      if (UseC == DefC)
        continue;
      LLVM_DEBUG(dbgs() << "  Merge cluster " << UseC << " into " << DefC
                        << " for " << printReg(MO.getReg(), TRI) << "\n");
      mergeClusters(MBB, DefC, UseC);
    }
    for (const MachineOperand &MO : MI.all_defs()) {
      if (!MO.getReg().isPhysical())
        continue;
      PhysRegOwner[MO.getReg().asMCReg()] = MICluster[&MI];
    }
  }
}

/// For each vreg, merge the def cluster with each use cluster.
void MOSRegAlloc::scheduleVRegs(MachineBasicBlock &MBB) {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    LLVM_DEBUG(dbgs() << "  Considering " << printReg(R, TRI) << "\n");
    if (MRI->reg_nodbg_empty(R))
      continue;

    for (MachineOperand &MO : MRI->use_nodbg_operands(R)) {
      if (MO.isUndef())
        continue;
      MachineInstr &UseMI = *MO.getParent();
      // Terminator vreg uses don't drive merges — a merge involving the
      // terminator cluster would splice it mid-block; linearizeClusters
      // folds it in last.
      if (UseMI.isTerminator())
        continue;
      unsigned DefC = MICluster[MRI->getVRegDef(R)];
      unsigned UseC = MICluster[&UseMI];

      LLVM_DEBUG(dbgs() << "    DefC=" << DefC << " UseC=" << UseC);
      if (DefC == UseC) {
        LLVM_DEBUG(dbgs() << " — same cluster, skip\n");
        continue;
      }
      LLVM_DEBUG(dbgs() << "\n");

      LLVM_DEBUG(dbgs() << "    Merge cluster " << UseC << " into " << DefC
                        << " for " << printReg(R, TRI) << "\n");
      mergeClusters(MBB, DefC, UseC);
    }
  }
}

/// Final linearization. Anything left disconnected after physreg / vreg
/// scheduling — independent def-use components, plus an empty K_livein —
/// gets merged adjacently in MBB order. Walk MBB; whenever MICluster
/// changes, merge the new cluster into the running current cluster. Each
/// merge hits mergeClusters' else-branch (UseBegin == Def.end()) since the
/// walk is in current layout order, so no splice happens — only Range and
/// LiveOut bookkeeping.
void MOSRegAlloc::linearizeClusters(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return;
  unsigned CurC = MICluster[&MBB.front()];
  for (MachineInstr &MI : MBB) {
    unsigned C = MICluster[&MI];
    if (C == CurC)
      continue;
    LLVM_DEBUG(dbgs() << "  Linearize: merge cluster " << C << " into "
                      << CurC << "\n");
    mergeClusters(MBB, CurC, C);
  }
}

/// Merge cluster UseC into cluster DefC: relocate UseC's instructions to
/// directly follow DefC's last instruction, transfer their MICluster
/// mapping to DefC, set kill flags on the relocated instructions, and
/// allocate them against DefC's running LiveOut. UseC is left empty.
void MOSRegAlloc::mergeClusters(MachineBasicBlock &MBB, unsigned DefC,
                                unsigned UseC) {
  Cluster &Def = Clusters[DefC];
  Cluster &Use = Clusters[UseC];
  auto UseBegin = Use.begin();
  for (MachineInstr &MI : Use)
    MICluster[&MI] = DefC;
  if (UseBegin != Def.end()) {
    // Fix the predecessor cluster whose end sentinel is Use.begin().
    unsigned PredC = MICluster[&*std::prev(UseBegin)];
    Clusters[PredC].Range = make_range(Clusters[PredC].begin(), Use.end());
    MBB.splice(Def.end(), &MBB, UseBegin, Use.end());
    // For non-empty Def, the splice naturally extends Def.Range: Def.end()
    // is a stable iterator and the list path from Def.begin() to Def.end()
    // now traverses the spliced instructions. For empty Def, begin == end
    // and the spliced nodes land *before* begin, outside the range. Set
    // begin to the freshly-spliced first instruction (UseBegin still points
    // to it after splice).
    if (Def.empty())
      Def.Range = make_range(UseBegin, Def.end());
  } else {
    Def.Range = make_range(Def.begin(), Use.end());
  }
  Use.Range = make_range(Use.end(), Use.end());
  Use.LiveOut.clear();

  // Update kill flags and allocate only the new (use) instructions.
  auto NewMIs = make_range(UseBegin, Def.end());
  setKillFlags(NewMIs, DefC);
  allocate(NewMIs, Def.LiveOut);
}

/// Set kill flags for new instructions added to a cluster. A use of V is a
/// kill iff it is the last use of V in MIs and all of V's uses are in the
/// cluster. Existing kill flags in the cluster are not affected. A kill set
/// here never goes stale: merges only append, so once V's uses are all
/// gathered, the last of them remains V's last use in any final order.
void MOSRegAlloc::setKillFlags(iterator_range<MBBIterator> MIs,
                               unsigned ClusterIdx) {
  DenseMap<Register, MachineOperand *> LastUse;
  for (MachineInstr &MI : MIs)
    for (MachineOperand &MO : MI.all_uses())
      LastUse[MO.getReg()] = &MO;

  for (auto &[V, MO] : LastUse) {
    bool AllInside =
        llvm::all_of(MRI->use_nodbg_instructions(V), [&](MachineInstr &U) {
          return MICluster[&U] == ClusterIdx;
        });
    if (AllInside)
      MO->setIsKill(true);
  }
}

/// Assign physical registers top-down greedily, replacing vregs in place.
/// Uses the effective sets computed during allocation.
void MOSRegAlloc::assignRegisters(MachineBasicBlock &MBB) {
  BitVector LiveUnits(TRI->getNumRegUnits());

  // Liveins occupy their physregs at MBB entry.
  for (const auto &LI : MBB.liveins())
    for (MCRegUnit Unit : TRI->regunits(MCPhysReg(LI.PhysReg)))
      LiveUnits.set(static_cast<unsigned>(Unit));

  for (MachineInstr &MI : MBB) {
    // Free killed uses. Prior defs' replaceRegWith already made these physregs.
    for (const MachineOperand &MO : MI.all_uses()) {
      if (!MO.isKill() || MO.isUndef())
        continue;
      assert(MO.getReg().isPhysical() && "Use not yet replaced");
      for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg()))
        LiveUnits.reset(static_cast<unsigned>(Unit));
    }

    // Assign defs: pick first free physreg from the effective set. Physreg
    // defs (e.g. implicit-def $c) keep their fixed register; just mark it
    // live.
    for (const MachineOperand &MO : MI.all_defs()) {
      Register Reg = MO.getReg();
      MCPhysReg Assigned = 0;
      if (Reg.isPhysical()) {
        Assigned = Reg.asMCReg();
      } else if (MO.isTied()) {
        // Tied def inherits its slot's physreg from the tied use, which has
        // already been rewritten to a physreg by the slot root's
        // replaceRegWith (or it was a livein physreg).
        unsigned UseIdx = MI.findTiedOperandIdx(MO.getOperandNo());
        Assigned = MI.getOperand(UseIdx).getReg().asMCReg();
        MRI->replaceRegWith(Reg, Assigned);
      } else {
        const BitVector &Eff = EffectiveRC[Reg];
        for (unsigned PhysReg : Eff.set_bits()) {
          bool Free = llvm::none_of(TRI->regunits(MCPhysReg(PhysReg)),
                                    [&](MCRegUnit U) {
                                      return LiveUnits.test(
                                          static_cast<unsigned>(U));
                                    });
          if (Free) {
            Assigned = MCPhysReg(PhysReg);
            break;
          }
        }
        assert(Assigned && "No free register in effective set");
        LLVM_DEBUG(dbgs() << "    Assign " << printReg(Reg, TRI) << " -> "
                          << printReg(Assigned, TRI) << "\n");
        MRI->replaceRegWith(Reg, Assigned);
      }
      for (MCRegUnit Unit : TRI->regunits(Assigned))
        LiveUnits.set(static_cast<unsigned>(Unit));
    }

    // Free dead defs' physregs after the full def loop so intra-MI later
    // defs still saw them as occupied for their free-physreg search.
    for (const MachineOperand &MO : MI.all_defs())
      if (MO.isDead())
        for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg()))
          LiveUnits.reset(static_cast<unsigned>(Unit));
  }
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS_BEGIN(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                      false)
INITIALIZE_PASS_END(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                    false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
