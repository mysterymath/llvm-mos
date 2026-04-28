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
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
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
  initClusters(MBB);
  schedule(MBB);
  assignRegisters(MBB);
  MRI->clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

void MOSRegAlloc::validate(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    assert(!MI.isTerminator() && "Terminators not yet supported");
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
      assert(!MO.isTied() && "Tied operands not yet supported");
      if (MO.isUse())
        assert(!MO.isUndef() && "Undef uses not yet supported");
      if (MO.isDef())
        assert(!MO.isDead() && "Dead defs not yet supported");
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

  Cluster LiveIns(make_range(MBB.begin(), MBB.begin()));
  for (const auto &LI : MBB.liveins())
    LiveIns.LiveOut.insert(LI.PhysReg);
  Clusters.push_back(std::move(LiveIns));

  for (MachineInstr &MI : MBB) {
    MICluster[&MI] = Clusters.size();
    Cluster C(make_range(MI.getIterator(), std::next(MI.getIterator())));
    allocate(C.Range, C.LiveOut);
    Clusters.push_back(std::move(C));
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
      if (!MO.isKill())
        continue;
      Live.erase(MO.getReg());
    }

    for (const MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();
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
void MOSRegAlloc::schedulePhysRegs(MachineBasicBlock &MBB) {
  DenseMap<MCPhysReg, unsigned> PhysRegOwner;
  for (const auto &LI : MBB.liveins())
    PhysRegOwner[LI.PhysReg] = LiveInClusterIdx;

  // Each merge here only relocates MI itself (its cluster is still a
  // singleton when we visit it), so MI's original successor is untouched.
  // make_early_inc_range captures the successor before the merge runs.
  for (MachineInstr &MI : llvm::make_early_inc_range(MBB)) {
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

    for (MachineInstr &UseMI : MRI->use_nodbg_instructions(R)) {
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
/// cluster. Existing kill flags in the cluster are not affected.
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
      if (!MO.isKill())
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
  }
}

} // namespace

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
