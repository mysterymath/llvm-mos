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
// A dependence graph fixed by the input order (data, anti, and output edges
// over registers) constrains the merging: a merge must keep the cluster-level
// graph acyclic (canMerge), and the final linearization folds clusters in
// topological order, so every schedule the allocator can produce respects the
// input's def-use bindings.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MOS.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
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
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

using MBBIterator = MachineBasicBlock::iterator;
using LiveSet = SmallSet<Register, 8>;

/// Dependence kinds, mirroring SDep's vocabulary; an Order kind is reserved
/// for future memory/barrier edges. Stored as bits: a cluster-level edge
/// carries the union of the kinds of all instruction dependencies it stands
/// for. Kinds encode two independent properties, and consumers filter by
/// the masks below: whether the edge constrains the final order, and
/// whether it proposes merging its endpoints (gathering a value's def and
/// uses into one cluster is the allocator's goal).
enum DepKinds : unsigned {
  DepData = 1 << 0,   ///< Regular data dependence (aka true-dependence).
  DepAnti = 1 << 1,   ///< A register anti-dependence (aka WAR).
  DepOutput = 1 << 2, ///< A register output-dependence (aka WAW).
  /// A read of a block livein value, hung off the livein cluster. Drives
  /// merging — that is how livein liveness reaches the reader's allocation
  /// — but constrains nothing: block entry precedes every instruction, and
  /// reads of the same livein commute. The ordering content of a livein is
  /// carried by its readers' Anti edges to the next aliasing def.
  DepLiveIn = 1 << 3,
};
/// Kinds that constrain the final instruction order.
constexpr unsigned DepConstraintKinds = DepData | DepAnti | DepOutput;
/// Kinds that propose merging the edge's endpoints.
constexpr unsigned DepMergeKinds = DepData | DepLiveIn;

/// A contiguous range of instructions in the MBB, representing a group of
/// instructions that have been scheduled together.
struct Cluster {
  explicit Cluster(iterator_range<MBBIterator> Range) : Range(Range) {}

  iterator_range<MBBIterator> Range;
  /// Registers live at the end of this cluster's allocation.
  LiveSet LiveOut;

  /// A cluster-level dependence edge: every instruction of cluster Succ
  /// must be scheduled after every instruction of this cluster (clusters
  /// are atomic). Kinds is a DepKinds bitmask.
  struct Dep {
    unsigned Succ;
    unsigned Kinds;
  };
  /// This cluster's dependence successors. Built over the initial singleton
  /// clusters by computeDepGraph and contracted by commitMerge as clusters
  /// merge.
  SmallVector<Dep, 4> Succs;

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
  /// The terminator cluster's ID, or ~0u if the block has no terminators.
  unsigned TermClusterIdx;
  SmallVector<Cluster, 0> Clusters;

  /// Maps each instruction to the ID of the cluster that contains it.
  DenseMap<MachineInstr *, unsigned> MICluster;

  void initClusters(MachineBasicBlock &MBB);
  void computeDepGraph(MachineBasicBlock &MBB);
  void addDep(unsigned Pred, unsigned Succ, unsigned Kinds);
  void addVRegDeps();
  void addPhysRegDeps(MachineBasicBlock &MBB);
  void contractDeps(unsigned DefC, unsigned UseC);
  bool canMerge(unsigned DefC, unsigned UseC);

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

  /// Count of defs in the last allocate() pass that could not be colored
  /// without inserting a copy. allocate() resets it to 0 at entry and
  /// narrowToFit bumps it when a def has no colorably-narrowable victim, so
  /// it is purely the output of the most recent allocate(). Callers inspect
  /// it after: tryMerge scores the trial by it, while non-merge callers
  /// (initClusters) and commitMerge assert it is 0 because their allocations
  /// must be colorable.
  unsigned Cost = 0;

  /// A merge trial's outcome plus the post-merge state needed to make it
  /// real. tryMerge fills this in for the cheapest trial seen so far;
  /// commitMerge swaps the saved maps into the live state directly, without
  /// re-running allocate(). Cost == ~0u means "no candidate yet".
  struct PendingMerge {
    unsigned DefC = 0;
    unsigned UseC = 0;
    unsigned Cost = ~0u;

    // Post-merge state, captured from the live maps at the end of a winning
    // trial (before tryMerge reverts them).
    IndexedMap<BitVector, VirtReg2IndexFunctor> EffectiveRC;
    IndexedMap<SmallVector<Register, 8>, VirtReg2IndexFunctor> Interferences;
    DenseMap<Register, Register> TiedRoot;
    DenseMap<MachineInstr *, unsigned> MICluster;
    LiveSet LiveOutAfter;
    SmallVector<MachineOperand *, 4> KilledMOs;

    void reset() { *this = PendingMerge(); }
  };
  /// The cheapest merge trial seen in the current scheduleClusters scan.
  PendingMerge BestMerge;

  void schedule(MachineBasicBlock &MBB);
  void scheduleClusters(MachineBasicBlock &MBB);
  void linearizeClusters(MachineBasicBlock &MBB);
  void mergeClusters(MachineBasicBlock &MBB, unsigned DefC, unsigned UseC);
  void tryMerge(unsigned DefC, unsigned UseC);
  void commitMerge(MachineBasicBlock &MBB, PendingMerge PM);
  void setKillFlags(iterator_range<MBBIterator> MIs, unsigned ClusterIdx,
                    SmallVectorImpl<MachineOperand *> *KilledMOs = nullptr);

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

  if (MF.size() != 1)
    report_fatal_error("multiple basic blocks not yet supported");
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
    if (MI.getOpcode() == TargetOpcode::REG_SEQUENCE)
      report_fatal_error("REG_SEQUENCE not yet supported");
    if (MI.getOpcode() == TargetOpcode::INSERT_SUBREG)
      report_fatal_error("INSERT_SUBREG not yet supported");
    if (MI.getOpcode() == TargetOpcode::EXTRACT_SUBREG)
      report_fatal_error("EXTRACT_SUBREG not yet supported");
    // These must not reorder with anything, which the dependence graph does
    // not model. (Terminators are exempt: the terminator span is an atomic
    // cluster that is always scheduled last, in its input order.)
    if (MI.isPosition())
      report_fatal_error("position-like instructions not yet supported");
    if (!MI.isTerminator() && MI.hasUnmodeledSideEffects())
      report_fatal_error(
          "instructions with unmodeled side effects not yet supported");

    for (const MachineOperand &MO : MI.operands()) {
      if (MO.isRegMask())
        report_fatal_error("regmasks not yet supported");
      if (!MO.isReg())
        continue;
      if (MO.isEarlyClobber())
        report_fatal_error("earlyclobber not yet supported");
      // No vreg defs in terminators: this makes any tied operand pair on
      // a terminator necessarily physreg-tied, so normalizeTiedRegs never
      // has to insert a COPY inside the terminator span.
      if (MI.isTerminator() && MO.isDef() && MO.getReg().isVirtual())
        report_fatal_error("vreg defs on terminators not yet supported");
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

/// Create one singleton cluster per instruction, allocate it, and connect
/// the clusters with their dependence edges (computeDepGraph). Cluster 0
/// (LiveInClusterIdx) is an empty cluster whose LiveOut publishes the
/// block's livein physregs; consumers of liveins are merged into it later
/// by schedule, exactly like any other physreg def/use.
void MOSRegAlloc::initClusters(MachineBasicBlock &MBB) {
  TermClusterIdx = ~0u;
  Clusters.clear();
  MICluster.clear();
  EffectiveRC.clear();
  EffectiveRC.resize(MRI->getNumVirtRegs());
  Interferences.clear();
  Interferences.resize(MRI->getNumVirtRegs());
  TiedRoot.clear();

  Clusters.emplace_back(make_range(MBB.begin(), MBB.begin()));
  Cluster &LiveIns = Clusters.back();
  for (const auto &LI : MBB.liveins())
    LiveIns.LiveOut.insert(LI.PhysReg);

  auto FirstTerm = MBB.getFirstTerminator();
  for (auto It = MBB.begin(); It != FirstTerm; ++It) {
    MachineInstr &MI = *It;
    Clusters.emplace_back(
        make_range(MI.getIterator(), std::next(MI.getIterator())));
    Cluster &C = Clusters.back();
    MICluster[&MI] = &C - Clusters.begin();
    allocate(C.Range, C.LiveOut);
    if (Cost != 0)
      report_fatal_error("single-instruction cluster is not colorable");
  }

  // Terminator span: one atomic cluster, never split. Its contiguity is
  // load-bearing — nothing may be inserted between its MIs.
  if (FirstTerm != MBB.end()) {
    TermClusterIdx = Clusters.size();
    Clusters.emplace_back(make_range(FirstTerm, MBB.end()));
    Cluster &Term = Clusters.back();
    for (MachineInstr &MI : Term)
      MICluster[&MI] = &Term - Clusters.begin();
    allocate(Term.Range, Term.LiveOut);
    if (Cost != 0)
      report_fatal_error("terminator cluster is not colorable");
  }

  computeDepGraph(MBB);
}

/// Build the initial dependence graph: the ordering facts the input order
/// imposes, which any final schedule must respect. Cluster merging preserves
/// these WRT the original instructions.
///
///   - Vreg Data edges: def -> each non-undef use
///   - Physreg Data edges: reaching def -> use, per regunit. This binds each
///     physreg use to the one def (or livein) whose value it reads — the
///     binding is never recomputed, so no later layout can rebind it.
///   - Physreg Anti edges: each use since the last aliasing def -> that next
///     def (the use must read before the new value clobbers).
///   - Physreg Output edges: aliasing def -> next def. Dead defs included;
///     their writes still clobber.
///
/// Block liveins contribute DepLiveIn edges from the livein cluster to
/// their readers. These drive merges but do not constrain order (see DepKinds).
///
/// Edges are recorded at cluster level (Cluster::Succs) over the initial
/// singleton clusters. All dependencies point forward in input order, so
/// the graph starts as a DAG; canMerge keeps its constraint edges acyclic
/// as commitMerge contracts merged clusters.
///
/// TODO: Memory dependencies are not modeled — the scheduler can reorder
/// aliasing loads and stores. They belong here as Order edges (store->load,
/// load->store, store->store), but with no MMOs on MOS memory ops,
/// MachineInstr::mayAlias is conservatively true and the edges would
/// serialize all memory traffic. Needs alias precision (MMOs from ISel or a
/// target-specific absolute-address disambiguator) first.
///
/// TODO: Anti and Output edges have no positive test yet. Register-keyed
/// liveness treats a multiply-written physreg as one value spanning first
/// def to last use, so every shape whose ordering they would constrain is
/// uncolorable today (the gathering merge costs 1 and commitMerge reports
/// "merge requires a copy"). Once copy insertion makes such shapes
/// colorable, add MIR tests pinning use-before-next-write and
/// def-before-next-def order.
void MOSRegAlloc::computeDepGraph(MachineBasicBlock &MBB) {
  addVRegDeps();
  addPhysRegDeps(MBB);
}

/// Record the cluster-level dependence edge Pred -> Succ, unioning Kinds
/// into the existing edge if there is one (each successor list stays
/// duplicate-free). Self-edges are dropped: order within a cluster is
/// already fixed, so an edge from a cluster to itself records nothing.
/// They come up in a few cases, e.g. dependencies between terminators.
void MOSRegAlloc::addDep(unsigned Pred, unsigned Succ, unsigned Kinds) {
  if (Pred == Succ)
    return;
  for (Cluster::Dep &E : Clusters[Pred].Succs) {
    if (E.Succ == Succ) {
      E.Kinds |= Kinds;
      return;
    }
  }
  Clusters[Pred].Succs.push_back({Succ, Kinds});
}

/// Add each vreg's Data edges: def -> each non-undef use. SSA's single def
/// means vregs have no Anti or Output edges.
void MOSRegAlloc::addVRegDeps() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;
    unsigned DefC = MICluster[MRI->getVRegDef(R)];
    for (MachineOperand &MO : MRI->use_nodbg_operands(R))
      if (!MO.isUndef())
        addDep(DefC, MICluster[MO.getParent()], DepData);
  }
}

/// Add the physreg Data, Anti, and Output edges by walking instructions
/// forward. Uses of liveins get DepLiveIn edges from the livein cluster.
void MOSRegAlloc::addPhysRegDeps(MachineBasicBlock &MBB) {
  struct UnitState {
    MachineInstr *LastDef = nullptr;
    SmallVector<MachineInstr *, 4> UsesSinceLastDef;
  };
  SmallVector<UnitState, 0> Units(TRI->getNumRegUnits());

  for (MachineInstr &MI : MBB) {
    unsigned C = MICluster[&MI];
    for (const MachineOperand &MO : MI.all_uses()) {
      Register R = MO.getReg();
      if (!R.isPhysical() || MO.isUndef())
        continue;
      for (MCRegUnit Unit : TRI->regunits(R.asMCReg())) {
        UnitState &S = Units[static_cast<unsigned>(Unit)];
        if (S.LastDef)
          addDep(MICluster[S.LastDef], C, DepData);
        else
          addDep(LiveInClusterIdx, C, DepLiveIn);
        S.UsesSinceLastDef.push_back(&MI);
      }
    }
    for (const MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();
      if (!R.isPhysical())
        continue;
      for (MCRegUnit Unit : TRI->regunits(R.asMCReg())) {
        UnitState &S = Units[static_cast<unsigned>(Unit)];
        for (MachineInstr *UseMI : S.UsesSinceLastDef)
          addDep(MICluster[UseMI], C, DepAnti);
        if (S.LastDef)
          addDep(MICluster[S.LastDef], C, DepOutput);
        S.LastDef = &MI;
        S.UsesSinceLastDef.clear();
      }
    }
  }
}

/// Contract UseC into DefC in the cluster dependence graph, mirroring their
/// merge. Edges between the pair become internal: Def->Use is satisfied by
/// the append order, and Use->Def cannot exist (canMerge). UseC's external
/// successors transfer to DefC, and every other cluster's edge to UseC is
/// remapped to DefC, unioning kinds with any existing edge.
void MOSRegAlloc::contractDeps(unsigned DefC, unsigned UseC) {
  llvm::erase_if(Clusters[DefC].Succs,
                 [&](const Cluster::Dep &E) { return E.Succ == UseC; });
  for (const Cluster::Dep &E : Clusters[UseC].Succs)
    addDep(DefC, E.Succ, E.Kinds);
  Clusters[UseC].Succs.clear();
  for (unsigned C = 0, End = Clusters.size(); C != End; ++C) {
    if (C == DefC || C == UseC)
      continue;
    auto &Succs = Clusters[C].Succs;
    auto *It = llvm::find_if(
        Succs, [&](const Cluster::Dep &E) { return E.Succ == UseC; });
    if (It == Succs.end())
      continue;
    addDep(C, DefC, It->Kinds);
    Succs.erase(It);
  }

  LLVM_DEBUG({
    dbgs() << "    Dep succs of " << DefC << ":";
    for (const Cluster::Dep &E : Clusters[DefC].Succs) {
      dbgs() << ' ' << E.Succ << '(';
      if (E.Kinds & DepData)
        dbgs() << 'd';
      if (E.Kinds & DepAnti)
        dbgs() << 'a';
      if (E.Kinds & DepOutput)
        dbgs() << 'o';
      if (E.Kinds & DepLiveIn)
        dbgs() << 'l';
      dbgs() << ')';
    }
    dbgs() << '\n';
  });
}

/// Would appending UseC to DefC violate an ordering barrier? The merge
/// permanently commits "all of DefC, then all of UseC, adjacent" — in
/// dependence-graph terms it contracts the two clusters into one node M. For
/// this to create a cycle, there would need to be a path from M to some other
/// part of the graph back to M. There cannot be any paths from UseC to DefC, so
/// there must be a path from DefC to some other part of the graph to UseC. The
/// merge is thus legal iff all paths from DefC to UseC are direct.
bool MOSRegAlloc::canMerge(unsigned DefC, unsigned UseC) {
  SmallDenseSet<unsigned, 16> Visited;
  SmallVector<unsigned, 16> Worklist;
  for (const Cluster::Dep &E : Clusters[DefC].Succs)
    if ((E.Kinds & DepConstraintKinds) && E.Succ != UseC &&
        Visited.insert(E.Succ).second)
      Worklist.push_back(E.Succ);
  while (!Worklist.empty()) {
    for (const Cluster::Dep &E : Clusters[Worklist.pop_back_val()].Succs) {
      if (!(E.Kinds & DepConstraintKinds))
        continue;
      if (E.Succ == UseC)
        return false;
      if (Visited.insert(E.Succ).second)
        Worklist.push_back(E.Succ);
    }
  }
  return true;
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
  // Cost reflects this pass alone; narrowToFit bumps it per uncolorable def.
  Cost = 0;
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
          continue; // Slot is just "physreg stays live through MI."
        unsigned UseIdx = MI.findTiedOperandIdx(MO.getOperandNo());
        const MachineOperand &UseMO = MI.getOperand(UseIdx);
        if (!UseMO.getReg().isVirtual())
          report_fatal_error("tied physreg operands not yet supported");
        assert(UseMO.isKill() && "tied use must be killed (normalizeTiedRegs)");
        assert(MRI->getRegClass(R) == MRI->getRegClass(UseMO.getReg()) &&
               "tied operands must have matching register classes");
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
    if (!NarrowReg.isValid()) {
      LLVM_DEBUG({
        dbgs() << "    !!! No narrowable victim for " << printReg(Def, TRI)
               << " (DefEff=" << DefEff.count()
               << ", slack=" << getSlack(DefEff, DefInters) << ")\n";
        dbgs() << "      DefInters: ";
        for (Register U : DefInters) {
          dbgs() << printReg(U, TRI);
          if (U.isVirtual())
            dbgs() << "(eff=" << EffectiveRC[U].count() << ")";
          dbgs() << " ";
        }
        dbgs() << "\n      Live: ";
        for (Register V : Live) {
          dbgs() << printReg(V, TRI);
          if (V.isVirtual())
            dbgs() << "(eff=" << EffectiveRC[V].count()
                   << ",iSlk=" << getSlack(EffectiveRC[V], Interferences[V])
                   << ")";
          dbgs() << " ";
        }
        dbgs() << "\n";
      });
      // No colorable victim. Bump Cost (a copy would be needed to color
      // this def) and stop narrowing for it. Callers inspect Cost: tryMerge
      // scores the trial by it; initClusters and commitMerge assert it is 0.
      ++Cost;
      break;
    }

    // V's interferences set stays as-is. Its slack drops naturally since
    // EffectiveRC[V] shrinks. Def's slack rises on the next getSlack call:
    // V's contribution to Def's interference is recomputed against the new
    // (smaller, RAlias-disjoint) EffectiveRC[V] and goes to 0. Any other
    // vreg that has V in its interferences is updated the same way the
    // next time its slack is queried.
    EffectiveRC[NarrowReg] = std::move(NarrowEffRC);

    LLVM_DEBUG(
        dbgs() << "    Narrow " << printReg(NarrowReg, TRI) << " to "
               << EffectiveRC[NarrowReg].count() << " regs (slack "
               << getSlack(EffectiveRC[NarrowReg], Interferences[NarrowReg])
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
  scheduleClusters(MBB);
  linearizeClusters(MBB);

  assert(llvm::count_if(Clusters,
                        [](const Cluster &C) { return !C.empty(); }) == 1 &&
         "not all clusters merged into one");

  // MBB is now in final schedule order.
  LLVM_DEBUG({
    dbgs() << "  Final schedule:\n";
    for (MachineInstr &MI : MBB)
      dbgs() << "    " << MI;
  });
}

/// Cost-aware merge selection. Each iteration enumerates every candidate
/// merge — the graph's merge-driving edges (DepMergeKinds): Data edges,
/// physreg and vreg alike, plus the livein cluster's DepLiveIn edges —
/// dry-running each legal one via tryMerge, which scores it by the
/// number of defs that would need a copy to be colorable and keeps the
/// cheapest in BestMerge. We then commit that winner and rescan, since a
/// commit can shift other candidates' costs through narrowing cascades and
/// can turn ordering-illegal candidates legal (canMerge) once an intervening
/// cluster merges into one side. We stop when no committable edge remains
/// (BestMerge.Cost == ~0u); whatever is left, linearizeClusters folds in
/// dependence-respecting topological order.
///
/// Copy insertion is not implemented yet, so commitMerge requires the winner
/// to be colorable as-is (Cost 0). Merges with higher cost become committable
/// once we can insert the copies they need.
void MOSRegAlloc::scheduleClusters(MachineBasicBlock &MBB) {
  while (true) {
    BestMerge.reset();

    for (unsigned C = 0, End = Clusters.size(); C != End; ++C) {
      for (const Cluster::Dep &E : Clusters[C].Succs) {
        if (!(E.Kinds & DepMergeKinds))
          continue;
        // The terminator cluster is atomic and anchored at MBB.end(); merging
        // it as a use side would splice it mid-block. linearizeClusters folds
        // it in last instead.
        if (E.Succ == TermClusterIdx)
          continue;
        if (!canMerge(C, E.Succ)) {
          LLVM_DEBUG(dbgs() << "  Skip merge " << E.Succ << " into " << C
                            << ": ordering barrier\n");
          continue;
        }
        tryMerge(C, E.Succ);
      }
    }

    // No committable edge remains: every def-use-connected cluster that can
    // legally merge has merged.
    if (BestMerge.Cost == ~0u)
      break;
    LLVM_DEBUG(dbgs() << "  Commit merge " << BestMerge.UseC << " into "
                      << BestMerge.DefC << " (cost " << BestMerge.Cost
                      << ")\n");
    commitMerge(MBB, std::move(BestMerge));
  }
}

/// Final linearization. Whatever scheduleClusters leaves — independent
/// def-use components and pairs whose merges stayed ordering-illegal — gets
/// folded into one cluster here. Nodes are merged in topological order; ties go
/// to the earliest current layout position: when layout is already consistent
/// (the common case) this reproduces it, and it keeps the terminator cluster —
/// which has only incoming edges and sits at layout end — last.
///
/// Consecutive clusters of a topological order are always legal to fold: any
/// cluster constrained to sit between them would appear between them in
/// every topological order.
void MOSRegAlloc::linearizeClusters(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return;

  bool MergedAny = true;
  while (MergedAny) {
    MergedAny = false;
    // Merge into the live in block the first cluster that has no predecessors
    // except the livein block.
    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end(); I != E;
         I = Clusters[MICluster[&*I]].end()) {
      unsigned C = MICluster[&*I];
      if (C == LiveInClusterIdx)
        continue;

      bool HasPred = false;
      for (unsigned OtherC = LiveInClusterIdx + 1, OtherE = Clusters.size();
           !HasPred && OtherC != OtherE; ++OtherC) {
        if (llvm::any_of(Clusters[OtherC].Succs, [&](Cluster::Dep &E) {
              return (E.Kinds & DepConstraintKinds) && E.Succ == C;
            }))
          HasPred = true;
      }
      if (HasPred)
        continue;

      LLVM_DEBUG(dbgs() << "  Linearize: merge cluster " << C << " into "
                        << LiveInClusterIdx << "\n");
      mergeClusters(MBB, LiveInClusterIdx, C);
      MergedAny = true;
    }
  }
}

/// Unconditionally merge cluster UseC into cluster DefC: dry-run it with
/// tryMerge to capture its effects, then commitMerge them. Used by
/// linearizeClusters, which folds the remaining clusters in topological
/// order and so always wants to take the one merge it names (rather than
/// picking the cheapest among competing candidates, as scheduleClusters does
/// by calling tryMerge / commitMerge directly).
void MOSRegAlloc::mergeClusters(MachineBasicBlock &MBB, unsigned DefC,
                                unsigned UseC) {
  assert(canMerge(DefC, UseC) && "merge violates an ordering barrier");
  BestMerge.reset();
  tryMerge(DefC, UseC);
  commitMerge(MBB, std::move(BestMerge));
}

/// Dry-run a merge of cluster UseC into DefC: compute its cost (the number of
/// defs that would need a copy to be colorable) and, if it is cheaper than
/// the current BestMerge, capture the resulting post-merge state into
/// BestMerge for commitMerge to swap in later. Real allocator state is fully
/// reverted before returning, so trials are independent and repeatable.
///
/// No splice happens here: the use instructions are allocated in their
/// current location against a copy of Def's running live set. allocate()
/// processes the same MI objects either way, so this yields the same
/// EffectiveRC / Interferences / Cost as if they had been spliced after Def.
void MOSRegAlloc::tryMerge(unsigned DefC, unsigned UseC) {
  Cluster &Def = Clusters[DefC];
  Cluster &Use = Clusters[UseC];

  // Snapshot every piece of persistent state a trial mutates, so we can
  // revert after scoring. (allocate touches EffectiveRC / Interferences /
  // TiedRoot; we reassign MICluster below; setKillFlags flips kill flags,
  // tracked separately via KilledMOs.)
  auto SavedERC = EffectiveRC;
  auto SavedInters = Interferences;
  auto SavedTiedRoot = TiedRoot;
  auto SavedMICluster = MICluster;

  // Reassign the use instructions to DefC so setKillFlags sees the merged
  // cluster membership when deciding which uses die.
  for (MachineInstr &MI : Use)
    MICluster[&MI] = DefC;

  // Compute kill flags as if merged, recording the operands flipped so we
  // can undo them (commitMerge re-applies BestMerge.KilledMOs).
  SmallVector<MachineOperand *, 4> KilledMOs;
  setKillFlags(Use.Range, DefC, &KilledMOs);

  // Allocate the use instructions against a copy of Def's running live set.
  LiveSet TmpLive = Def.LiveOut;
  allocate(Use.Range, TmpLive);

  LLVM_DEBUG(dbgs() << "  tryMerge " << UseC << " into " << DefC << ": cost "
                    << Cost << " (best " << BestMerge.Cost << ")\n");

  // If this trial is the cheapest so far, capture its post-merge state.
  // Move the mutated maps out (they are restored from the snapshots below).
  if (Cost < BestMerge.Cost) {
    BestMerge.DefC = DefC;
    BestMerge.UseC = UseC;
    BestMerge.Cost = Cost;
    BestMerge.EffectiveRC = std::move(EffectiveRC);
    BestMerge.Interferences = std::move(Interferences);
    BestMerge.TiedRoot = std::move(TiedRoot);
    BestMerge.MICluster = std::move(MICluster);
    BestMerge.LiveOutAfter = std::move(TmpLive);
    BestMerge.KilledMOs = KilledMOs;
  }

  // Revert all real state to baseline. On a win the maps were moved out
  // above, so this reassigns from the snapshots either way. Kill flags are
  // cleared regardless of win/loss — the real IR returns to baseline and
  // commitMerge re-sets the winning trial's flags from BestMerge.KilledMOs.
  EffectiveRC = std::move(SavedERC);
  Interferences = std::move(SavedInters);
  TiedRoot = std::move(SavedTiedRoot);
  MICluster = std::move(SavedMICluster);
  for (MachineOperand *MO : KilledMOs)
    MO->setIsKill(false);
}

/// Make a merge captured by tryMerge real. Swaps the recorded post-merge
/// maps into the live state (no re-allocation), re-applies the kill flags
/// the trial computed, then performs the splice and cluster-range
/// bookkeeping tryMerge deliberately skipped. UseC is left empty.
void MOSRegAlloc::commitMerge(MachineBasicBlock &MBB, PendingMerge PM) {
  if (PM.Cost != 0)
    report_fatal_error("merge requires a copy, which is not yet supported");
  unsigned DefC = PM.DefC;
  unsigned UseC = PM.UseC;
  Cluster &Def = Clusters[DefC];
  Cluster &Use = Clusters[UseC];

  // Swap the captured post-merge state into the live maps, replacing the
  // baseline tryMerge reverted to.
  EffectiveRC = std::move(PM.EffectiveRC);
  Interferences = std::move(PM.Interferences);
  TiedRoot = std::move(PM.TiedRoot);
  MICluster = std::move(PM.MICluster);

  // Re-apply the kill flags the trial computed (tryMerge cleared them).
  for (MachineOperand *MO : PM.KilledMOs)
    MO->setIsKill(true);

  // Splice UseC's instructions to follow DefC, mirroring the layout the
  // dry-run allocation assumed, and fix up the affected cluster ranges.
  auto UseBegin = Use.begin();
  if (UseBegin != Def.end()) {
    // Fix the predecessor cluster whose end sentinel is Use.begin(). At
    // MBB.begin() there is no predecessor: only empty clusters can be
    // anchored there, and an empty (it, it) range stays empty wherever its
    // node moves.
    if (UseBegin != MBB.begin()) {
      unsigned PredC = MICluster[&*std::prev(UseBegin)];
      Clusters[PredC].Range = make_range(Clusters[PredC].begin(), Use.end());
    }
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
  // TODO: When UseC is the livein cluster, this clear drops the liveins
  // from the liveness model: the trial allocated Use.Range against Def's
  // live set alone, and liveins have no defining instruction in the range
  // to re-add them. Values allocated afterwards can thus miss their livein
  // interference. The model only ever errs optimistic here —
  // assignRegisters' regunit tracking turns any resulting miscoloring into
  // "no free register in effective set" rather than a silent clobber — but
  // colorable inputs can be spuriously rejected. The trial should instead
  // allocate against the union of Def's live set and Use's surviving
  // liveins.
  Use.Range = make_range(Use.end(), Use.end());
  Use.LiveOut.clear();
  Def.LiveOut = std::move(PM.LiveOutAfter);

  // Mirror the merge in the cluster dependence graph.
  contractDeps(DefC, UseC);
}

/// Set kill flags for new instructions added to a cluster. A use of V is a
/// kill iff it is the last use of V in MIs and all of V's uses are in the
/// cluster. Existing kill flags in the cluster are not affected. A kill set
/// by a committed merge never goes stale: merges only append, so once V's
/// uses are all gathered, the last of them remains V's last use in any final
/// order. A dry run's kills hold only if its merge commits, so if KilledMOs
/// is non-null, each operand whose kill flag is flipped false→true is
/// appended to it (so the caller can undo the writes by clearing the kill
/// flags on those operands; commitMerge re-applies the winner's).
void MOSRegAlloc::setKillFlags(iterator_range<MBBIterator> MIs,
                               unsigned ClusterIdx,
                               SmallVectorImpl<MachineOperand *> *KilledMOs) {
  DenseMap<Register, MachineOperand *> LastUse;
  for (MachineInstr &MI : MIs)
    for (MachineOperand &MO : MI.all_uses())
      LastUse[MO.getReg()] = &MO;

  for (auto &[V, MO] : LastUse) {
    bool AllInside =
        llvm::all_of(MRI->use_nodbg_instructions(V), [&](MachineInstr &U) {
          return MICluster[&U] == ClusterIdx;
        });
    if (AllInside && !MO->isKill()) {
      MO->setIsKill(true);
      if (KilledMOs)
        KilledMOs->push_back(MO);
    }
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
      assert(MO.getReg().isPhysical() && "use not yet replaced");
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
          bool Free = llvm::none_of(
              TRI->regunits(MCPhysReg(PhysReg)), [&](MCRegUnit U) {
                return LiveUnits.test(static_cast<unsigned>(U));
              });
          if (Free) {
            Assigned = MCPhysReg(PhysReg);
            break;
          }
        }
        if (!Assigned)
          report_fatal_error("no free register in effective set");
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
