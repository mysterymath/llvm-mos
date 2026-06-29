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
// A dependence graph fixed by the input order (data edges over registers;
// register anti/output edges are intentionally not modeled — see
// computeDepGraph) constrains the merging: a merge must keep the cluster-level
// graph acyclic (canMerge), and the final linearization folds clusters in
// topological order, so every schedule the allocator can produce respects the
// input's def-use bindings.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
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

/// Dependence kinds, mirroring SDep's vocabulary. Stored as bits: a
/// cluster-level edge carries the union of the kinds of all instruction
/// dependencies it stands for. Kinds encode two independent properties, and
/// consumers filter by the masks below: whether the edge constrains the final
/// order, and whether it proposes merging its endpoints (gathering a value's
/// def and uses into one cluster is the allocator's goal). Data does both.
///
/// Only Data edges exist today. The two masks below are kept distinct so that
/// constrain-only edges — register anti/output, or future memory/barrier edges
/// — can be reintroduced without touching the consumers that filter by them.
enum DepKinds : unsigned {
  DepData = 1 << 0, ///< Regular data dependence (aka true-dependence).
};
/// Kinds that constrain the final instruction order.
constexpr unsigned DepConstraintKinds = DepData;
/// Kinds that propose merging the edge's endpoints.
constexpr unsigned DepMergeKinds = DepData;

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
  /// Cluster 0 is the livein cluster: a source cluster, pinned at the block
  /// front, whose LiveOut publishes the block's livein physregs. It has no
  /// dependence edges, so the greedy mergeClusters never touches it;
  /// linearizeClusters then folds every other cluster into it, allocating
  /// each against its (livein-carrying) LiveOut.
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
  void fusePhysRegCopies(MachineBasicBlock &MBB);

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

  /// Minimum residual slack (headroom before a forced copy) over the defs of
  /// the last allocate() pass — the secondary, pressure-aware merge key under
  /// Cost. allocate() resets it to INT_MAX and folds in each vreg def's
  /// post-narrow slack. Among equal-Cost trials tryMerge prefers the one that
  /// keeps this higher, i.e. the schedule that stays furthest from a copy.
  /// Because a held register depresses the slack of every later def it
  /// interferes with, maximizing it also steers toward register-freeing
  /// merges, so directionality needs no separate heuristic. Class-agnostic:
  /// tight classes show low slack through their small effective sets, with no
  /// class named here.
  int TrialMinSlack = std::numeric_limits<int>::max();

  /// A merge trial's outcome plus the post-merge state needed to make it
  /// real. tryMerge fills this in for the cheapest trial seen so far;
  /// commitMerge swaps the saved maps into the live state directly, without
  /// re-running allocate(). Cost == ~0u means "no candidate yet".
  struct PendingMerge {
    unsigned DefC = 0;
    unsigned UseC = 0;
    unsigned Cost = ~0u;
    /// Secondary key: the captured trial's TrialMinSlack; higher is better.
    /// INT_MIN until a candidate is captured, so any real trial wins the tie.
    int MinSlack = std::numeric_limits<int>::min();

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
  /// The cheapest merge trial seen in the current merge scan.
  PendingMerge BestMerge;

  void schedule(MachineBasicBlock &MBB);
  void mergeClusters(MachineBasicBlock &MBB);
  void linearizeClusters(MachineBasicBlock &MBB);
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
  fusePhysRegCopies(MBB);
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

/// Replace Run — each a single-def, single-use COPY — with one PCOPY inserted at
/// Run.front()'s position. Operand layout is all defs in run order followed by
/// all uses in run order, so def[i] pairs with use[i]; the variadic operands
/// carry no explicit tie, so the expansion recovers the pairing by position.
/// Each def/use operand is copied wholesale, so every flag the original COPY
/// carried (dead/kill/undef/renamable/subreg) survives unchanged.
static void buildPCopyFromRun(MachineBasicBlock &MBB,
                              ArrayRef<MachineInstr *> Run,
                              const TargetInstrInfo &TII) {
  MachineInstrBuilder MIB =
      BuildMI(MBB, Run.front()->getIterator(), Run.front()->getDebugLoc(),
              TII.get(MOS::PCOPY));
  for (const MachineInstr *MI : Run)
    MIB.add(MI->getOperand(0));
  for (MachineInstr *MI : make_early_inc_range(Run)) {
    MIB.add(MI->getOperand(1));
    MI->eraseFromParent();
  }
}

/// A COPY that moves a value into or out of a physical register: the copies that
/// liveins, returns, calling conventions, and inline asm use to cross a
/// register-allocation boundary. Pure vreg-to-vreg copies are left alone for the
/// cluster coalescer to handle; fusing them would gain nothing, as the register
/// anti/output edges that PCOPY exists to dissolve are physreg-only.
static bool isPhysRegCopy(const MachineInstr &MI) {
  return MI.isCopy() && (MI.getOperand(0).getReg().isPhysical() ||
                         MI.getOperand(1).getReg().isPhysical());
}

/// Fuse each maximal run of consecutive physreg copies into one parallel copy
/// (PCOPY). A parallel copy is the faithful representation of such a run: the
/// sources are read and the destinations written all at once, with no order
/// among the moves to model. That is what lets computeDepGraph drop the register
/// anti/output edges that existed only to sequence them. The run is extended
/// only while it stays a valid parallel copy — no copy may read or overwrite a
/// register an earlier copy in the run already wrote, since either would make
/// the all-at-once reading differ from the original sequential one (e.g.
/// `$a = COPY $x; $x = COPY $a` is not a swap and must not fuse). Only runs of
/// length >= 2 are fused; a lone copy has no internal order to lose. The PCOPY
/// survives as a post-RA pseudo; ExpandPostRAPseudos lowers it to individual
/// moves once the operands are physregs. assignRegisters biases the operand
/// assignment so the moves never form a permutation cycle, keeping that lowering
/// temporary-free.
void MOSRegAlloc::fusePhysRegCopies(MachineBasicBlock &MBB) {
  const TargetInstrInfo &TII = *MBB.getParent()->getSubtarget().getInstrInfo();

  for (auto It = MBB.begin(), End = MBB.end(); It != End;) {
    if (!isPhysRegCopy(*It)) {
      ++It;
      continue;
    }

    // Grab consecutive physreg copies while the run stays a valid parallel copy:
    // a copy whose source or destination overlaps an earlier copy's destination
    // would carry a sequential dependence the parallel form can't preserve, so
    // it ends the run and starts the next one.
    SmallVector<MachineInstr *, 4> Run;
    auto WrittenEarlier = [&](Register R) {
      return llvm::any_of(Run, [&](const MachineInstr *M) {
        return TRI->regsOverlap(M->getOperand(0).getReg(), R);
      });
    };
    for (; It != End && isPhysRegCopy(*It); ++It) {
      Register Dst = It->getOperand(0).getReg();
      Register Src = It->getOperand(1).getReg();
      if (WrittenEarlier(Src) || WrittenEarlier(Dst))
        break;
      Run.push_back(&*It);
    }

    if (Run.size() >= 2)
      buildPCopyFromRun(MBB, Run, TII);
  }
}

/// Create one singleton cluster per instruction, allocate it, and connect
/// the clusters with their dependence edges (computeDepGraph). Cluster 0
/// (LiveInClusterIdx) is an empty cluster whose LiveOut publishes the
/// block's livein physregs; linearizeClusters later folds every other cluster
/// into it, so liveins reach each allocation through its LiveOut.
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
///
/// A physreg use with no reaching def reads a block livein; it gets no edge at
/// all (the livein cluster has no defining instruction to depend on). Livein
/// liveness reaches such a use in linearizeClusters, which folds its cluster
/// into the livein cluster and allocates it against that cluster's LiveOut.
///
/// Edges are recorded at cluster level (Cluster::Succs) over the initial
/// singleton clusters. All dependencies point forward in input order, so
/// the graph starts as a DAG; canMerge keeps its constraint edges acyclic
/// as commitMerge contracts merged clusters.
///
/// Register anti (WAR) and output (WAW) edges are deliberately not modeled. The
/// only such hazard in real input is a livein physreg read before it is
/// overwritten for a return value, and that ordering is preserved structurally:
/// the livein reader is a dependence source folded at the block front, while
/// the return writer keeps its Data edge to the terminator, which is folded
/// last — so the read always precedes the write. fusePhysRegCopies further
/// collapses each run of physreg copies into one atomic PCOPY. A colorable
/// interior physreg WAR/WAW ordered only by anti/output would be a real hazard,
/// but register-keyed liveness makes such a shape cost a copy, which commitMerge
/// already rejects, so none can be silently misordered.
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

/// Add the physreg Data edges by walking instructions forward, tracking the
/// last def of each regunit. A use with a reaching def gets a Data edge from
/// it; a use with no reaching def reads a block livein and gets no edge (the
/// livein cluster has no defining instruction). Register anti/output edges are
/// deliberately not modeled — see computeDepGraph.
void MOSRegAlloc::addPhysRegDeps(MachineBasicBlock &MBB) {
  SmallVector<MachineInstr *, 0> LastDef(TRI->getNumRegUnits(), nullptr);

  for (MachineInstr &MI : MBB) {
    unsigned C = MICluster[&MI];
    for (const MachineOperand &MO : MI.all_uses()) {
      Register R = MO.getReg();
      if (!R.isPhysical() || MO.isUndef())
        continue;
      for (MCRegUnit Unit : TRI->regunits(R.asMCReg()))
        if (MachineInstr *Def = LastDef[static_cast<unsigned>(Unit)])
          addDep(MICluster[Def], C, DepData);
    }
    for (const MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();
      if (!R.isPhysical())
        continue;
      for (MCRegUnit Unit : TRI->regunits(R.asMCReg()))
        LastDef[static_cast<unsigned>(Unit)] = &MI;
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
    // Erase before addDep: addDep may push_back onto this same Succs vector
    // (C -> DefC), reallocating it and invalidating It. Capture the kinds and
    // drop the C -> UseC edge first, then add the remapped edge.
    unsigned Kinds = It->Kinds;
    Succs.erase(It);
    addDep(C, DefC, Kinds);
  }

  LLVM_DEBUG({
    dbgs() << "    Dep succs of " << DefC << ":";
    for (const Cluster::Dep &E : Clusters[DefC].Succs) {
      dbgs() << ' ' << E.Succ << '(';
      if (E.Kinds & DepData)
        dbgs() << 'd';
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
  // Cost and TrialMinSlack reflect this pass alone; narrowToFit bumps Cost per
  // uncolorable def, and the def loop folds each colorable def's residual slack
  // into TrialMinSlack.
  Cost = 0;
  TrialMinSlack = std::numeric_limits<int>::max();
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

        // Fold this def's residual headroom into the trial's min slack. A def
        // allocated while a register is held sees the held value among its
        // interferers, so its slack is depressed — that is what penalizes
        // holding and rewards freeing, with no class named.
        int Slack = getSlack(EffectiveRC[R], Interferences[R]);
        TrialMinSlack = std::min(TrialMinSlack, Slack);
        LLVM_DEBUG(dbgs() << "    Slack for " << printReg(R, TRI) << " ("
                          << TRI->getRegClassName(MRI->getRegClass(R))
                          << "): " << Slack
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

/// Reduce the block to a single cluster — its final schedule — in two
/// cost-driven passes over the same merge machinery. mergeClusters greedily
/// gathers values along data edges; linearizeClusters then folds whatever
/// remains into the livein cluster. Both pick the cheapest legal merge each
/// step; they differ only in which candidates they propose.
void MOSRegAlloc::schedule(MachineBasicBlock &MBB) {
  mergeClusters(MBB);
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

/// Greedy cost-aware gathering. Each iteration enumerates every candidate
/// merge — the graph's merge-driving edges (DepMergeKinds, i.e. Data edges,
/// physreg and vreg alike) — dry-running each legal one via tryMerge, which
/// scores it by the number of defs that would need a copy to be colorable and
/// keeps the cheapest in BestMerge. We then commit that winner and rescan,
/// since a commit can shift other candidates' costs through narrowing cascades
/// and can turn ordering-illegal candidates legal (canMerge) once an
/// intervening cluster merges into one side. We stop when no committable edge
/// remains (BestMerge.Cost == ~0u); whatever is left, linearizeClusters folds
/// into the livein cluster, by the same cost criterion.
///
/// Copy insertion is not implemented yet, so commitMerge requires the winner
/// to be colorable as-is (Cost 0). Merges with higher cost become committable
/// once we can insert the copies they need.
void MOSRegAlloc::mergeClusters(MachineBasicBlock &MBB) {
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

/// Final linearization. Whatever mergeClusters leaves — independent def-use
/// components and pairs whose merges stayed ordering-illegal — gets folded
/// into the livein cluster here, which is the accumulator: appending at its
/// end grows the schedule from the block's front. Each round, among the
/// topologically ready clusters (those whose predecessors are all already
/// folded in), fold the cheapest, so the same cost criterion that drives
/// mergeClusters governs these merges too; ties break toward the lower cluster
/// id, which is input order. Folding consecutive clusters of a topological
/// order is always legal: any cluster constrained to sit between them would
/// appear between them in every topological order.
///
/// The terminator is held back until everything else is folded: its
/// instructions must land at the block's end, and the fold appends at the
/// accumulator's end, so it has to be the last thing appended.
void MOSRegAlloc::linearizeClusters(MachineBasicBlock &MBB) {
  if (MBB.empty())
    return;

  // commitMerge keeps the livein cluster pinned at the block front, so it is a
  // valid fold sink here without any re-anchoring.
  assert(Clusters[LiveInClusterIdx].empty() &&
         "livein cluster should be untouched by the greedy phase");

  while (true) {
    BestMerge.reset();
    for (unsigned C = LiveInClusterIdx + 1, End = Clusters.size(); C != End;
         ++C) {
      if (Clusters[C].empty() || C == TermClusterIdx)
        continue;
      // Ready iff no other (non-livein) cluster still constrains C to follow
      // it. The livein cluster is excluded from the scan: it is the sink C
      // folds into, so an edge from it is what we are resolving, not a barrier.
      bool HasPred = false;
      for (unsigned P = LiveInClusterIdx + 1, PE = Clusters.size();
           !HasPred && P != PE; ++P)
        HasPred = llvm::any_of(Clusters[P].Succs, [&](const Cluster::Dep &E) {
          return (E.Kinds & DepConstraintKinds) && E.Succ == C;
        });
      if (HasPred)
        continue;
      tryMerge(LiveInClusterIdx, C);
    }
    if (BestMerge.Cost == ~0u)
      break;
    LLVM_DEBUG(dbgs() << "  Linearize: merge cluster " << BestMerge.UseC
                      << " into " << LiveInClusterIdx << " (cost "
                      << BestMerge.Cost << ")\n");
    commitMerge(MBB, std::move(BestMerge));
  }

  // Fold the terminator last, into the now-complete accumulator.
  if (TermClusterIdx != ~0u && !Clusters[TermClusterIdx].empty()) {
    assert(canMerge(LiveInClusterIdx, TermClusterIdx) &&
           "terminator fold violates an ordering barrier");
    BestMerge.reset();
    tryMerge(LiveInClusterIdx, TermClusterIdx);
    LLVM_DEBUG(dbgs() << "  Linearize: merge terminator " << TermClusterIdx
                      << " into " << LiveInClusterIdx << " (cost "
                      << BestMerge.Cost << ")\n");
    commitMerge(MBB, std::move(BestMerge));
  }
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
  // For the livein cluster as Def this is its LiveOut, which carries the block
  // liveins; the livein cluster is never the Use side (it has no incoming
  // edges), so Def.LiveOut is always the complete incoming liveness here.
  LiveSet TmpLive = Def.LiveOut;
  allocate(Use.Range, TmpLive);

  LLVM_DEBUG(dbgs() << "  tryMerge " << UseC << " into " << DefC << ": cost "
                    << Cost << " minslack " << TrialMinSlack << " (best "
                    << BestMerge.Cost << "/" << BestMerge.MinSlack << ")\n");

  // Capture this trial if it beats the best so far: lower Cost wins outright,
  // and among equal Cost the higher TrialMinSlack — more headroom before a
  // forced copy — wins. Move the mutated maps out (restored from the snapshots
  // below).
  if (Cost < BestMerge.Cost ||
      (Cost == BestMerge.Cost && TrialMinSlack > BestMerge.MinSlack)) {
    BestMerge.DefC = DefC;
    BestMerge.UseC = UseC;
    BestMerge.Cost = Cost;
    BestMerge.MinSlack = TrialMinSlack;
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
  // The livein cluster is never the Use side — it has no incoming edges, so
  // nothing ever folds it into another cluster — so this clear never discards
  // the block liveins. They live on as the Def.LiveOut of the livein cluster
  // until linearizeClusters folds everything into it, allocating each cluster
  // against those liveins.
  Use.Range = make_range(Use.end(), Use.end());
  Use.LiveOut.clear();
  Def.LiveOut = std::move(PM.LiveOutAfter);

  // Mirror the merge in the cluster dependence graph.
  contractDeps(DefC, UseC);

  // Keep the livein cluster pinned at the block front. It holds no instruction
  // to anchor to, and an ilist has no stable front sentinel (only MBB.end()),
  // so its iterators would drift if this splice relocated the instruction they
  // happened to point at. Re-derive them from the current front so the position
  // linearizeClusters folds into stays correct. (Folding into it makes it
  // non-empty, leaving its real range untouched.)
  if (Clusters[LiveInClusterIdx].empty())
    Clusters[LiveInClusterIdx].Range = make_range(MBB.begin(), MBB.begin());
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

  // PCOPY pairs def[i] with use[i]. Where one side is a physreg and the other a
  // vreg, prefer to give that vreg the physreg: the move then collapses to an
  // identity, so the parallel copy can never become a permutation cycle (a
  // register swap) — which ExpandPostRAPseudos has no temporary to break. The
  // hint is keyed by the still-virtual operand, so compute it before the loop
  // rewrites any vregs.
  DenseMap<Register, MCPhysReg> Hint;
  for (const MachineInstr &MI : MBB) {
    if (MI.getOpcode() != MOS::PCOPY)
      continue;
    unsigned Half = MI.getNumOperands() / 2;
    for (unsigned I = 0; I != Half; ++I) {
      Register Def = MI.getOperand(I).getReg();
      Register Use = MI.getOperand(Half + I).getReg();
      if (Def.isVirtual() && Use.isPhysical())
        Hint[Def] = Use.asMCReg();
      else if (Use.isVirtual() && Def.isPhysical())
        Hint[Use] = Def.asMCReg();
    }
  }

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
        auto IsFree = [&](MCPhysReg P) {
          return llvm::none_of(TRI->regunits(P), [&](MCRegUnit U) {
            return LiveUnits.test(static_cast<unsigned>(U));
          });
        };
        // Take the PCOPY hint when it is allocatable for this slot and free, so
        // the paired boundary move collapses to an identity (keeps the parallel
        // copy acyclic). Otherwise fall back to the first free register.
        auto HintIt = Hint.find(Reg);
        if (HintIt != Hint.end() && HintIt->second < Eff.size() &&
            Eff.test(HintIt->second) && IsFree(HintIt->second))
          Assigned = HintIt->second;
        for (unsigned PhysReg : Eff.set_bits()) {
          if (Assigned)
            break;
          if (IsFree(MCPhysReg(PhysReg)))
            Assigned = MCPhysReg(PhysReg);
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
