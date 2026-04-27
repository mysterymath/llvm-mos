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
  /// Per-vreg interferences: the set of vregs live at V's definition. V's
  /// total interference (and thus slack) is computed on demand by summing
  /// maxRegInterference(V, EffectiveRC[U]) over each U in this set against
  /// U's current EffectiveRC — so any later narrowing of a U is reflected
  /// automatically. Persists past V's death (chained narrowing may revisit
  /// V).
  IndexedMap<SmallVector<Register, 8>, VirtReg2IndexFunctor> Interferences;

  void allocate(iterator_range<MBBIterator> MIs, LiveSet &Live);
  int getSlack(Register V) const { return getSlack(V, EffectiveRC[V]); }
  int getSlack(Register V, const BitVector &EffRC) const {
    int Slack = EffRC.count();
    for (Register U : Interferences[V])
      Slack -= maxRegInterference(EffRC, EffectiveRC[U]);
    return Slack;
  }
  BitVector allocatableRegs(const TargetRegisterClass *RC);
  unsigned maxRegInterference(const BitVector &DefRegs,
                              const BitVector &LiveRegs) const;
  BitVector aliasSet(const BitVector &Regs);

  void mergeClusters(MachineBasicBlock &MBB);
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
  mergeClusters(MBB);
  assignRegisters(MBB);
  MRI->clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

void MOSRegAlloc::validate(MachineBasicBlock &MBB) {
  assert(MBB.livein_empty() && "Block liveins not yet supported");

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
      assert(!MO.getReg().isPhysical() &&
             "Physical register operands not yet supported");
      assert(!MO.isEarlyClobber() && "Earlyclobber not yet supported");
      assert(!MO.isTied() && "Tied operands not yet supported");
      if (MO.isUse())
        assert(!MO.isUndef() && "Undef uses not yet supported");
      if (MO.isDef())
        assert(!MO.isDead() && "Dead virtual reg defs not yet supported");
    }
  }
}

/// Create one singleton cluster per instruction and allocate it.
void MOSRegAlloc::initClusters(MachineBasicBlock &MBB) {
  Clusters.clear();
  MICluster.clear();
  EffectiveRC.clear();
  EffectiveRC.resize(MRI->getNumVirtRegs());
  Interferences.clear();
  Interferences.resize(MRI->getNumVirtRegs());
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
      EffectiveRC[R] = allocatableRegs(MRI->getRegClass(R));
      Interferences[R].assign(Live.begin(), Live.end());
      Live.insert(R);

      // Slack < 1 means R's worst-case interference has eaten its whole
      // effective set: we can't guarantee a free register. Narrow live
      // values out of R's alias set until R recovers, picking each victim
      // so that it also remains colorable after losing those registers.
      while (getSlack(R) < 1) {
        BitVector RAlias = aliasSet(EffectiveRC[R]);

        // Pick the smallest V whose post-narrow effective set
        // EffectiveRC[V] \ RAlias is both:
        //   - strictly smaller than EffectiveRC[V] (so narrowing helps R),
        //   - large enough to keep V colorable.
        // Apply the colorability filter before the tie-break: picking the
        // smallest V outright could land on a victim that loses
        // colorability when a slightly larger V could absorb the narrow
        // safely.
        Register NarrowReg;
        BitVector NarrowEffRC;
        for (Register V : Live) {
          if (V == R)
            continue;
          BitVector NewEffRC = EffectiveRC[V];
          NewEffRC.reset(RAlias);
          if (NewEffRC == EffectiveRC[V])
            continue;
          if (getSlack(V, NewEffRC) < 1)
            continue;
          if (!NarrowReg.isValid() || V < NarrowReg) {
            NarrowReg = V;
            NarrowEffRC = std::move(NewEffRC);
          }
        }
        assert(NarrowReg.isValid() &&
               "Allocation failed: no colorably-narrowable live value");

        // V's interferences set stays as-is. Its slack drops naturally
        // since EffectiveRC[V] shrinks. R's slack rises naturally on the
        // next getSlack call: V's contribution
        // maxRegInterference(R, EffectiveRC[V]) is recomputed against the
        // new (smaller, RAlias-disjoint) EffectiveRC[V] and goes to 0. Any
        // other vreg that has V in its interferences is updated the same
        // way the next time its slack is queried.
        EffectiveRC[NarrowReg] = std::move(NarrowEffRC);

        LLVM_DEBUG(dbgs() << "    Narrow " << printReg(NarrowReg, TRI) << " to "
                          << EffectiveRC[NarrowReg].count() << " regs (slack "
                          << getSlack(NarrowReg) << ")\n");
      }

      LLVM_DEBUG(dbgs() << "    Slack for " << printReg(R, TRI) << " ("
                        << TRI->getRegClassName(MRI->getRegClass(R))
                        << "): " << getSlack(R)
                        << " (eff=" << EffectiveRC[R].count() << ")\n");
    }
  }
}

/// Build a BitVector of allocatable physical registers for a register class.
BitVector MOSRegAlloc::allocatableRegs(const TargetRegisterClass *RC) {
  BitVector BV(TRI->getNumRegs());
  for (MCPhysReg R : RCI.getOrder(RC))
    BV.set(R);
  return BV;
}

/// The max number of registers in DefRegs that a single register from
/// LiveRegs can block via aliasing.
unsigned MOSRegAlloc::maxRegInterference(const BitVector &DefRegs,
                                         const BitVector &LiveRegs) const {
  unsigned Max = 0;
  for (unsigned LReg : LiveRegs.set_bits()) {
    unsigned Blocked = 0;
    for (unsigned DReg : DefRegs.set_bits())
      if (TRI->regsOverlap(MCPhysReg(LReg), MCPhysReg(DReg)))
        Blocked++;
    Max = std::max(Max, Blocked);
  }
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

void MOSRegAlloc::mergeClusters(MachineBasicBlock &MBB) {
  // For each vreg, merge the def cluster with each use cluster.
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

      // Merge UseC into DefC right after DefC's last instruction.
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
  }

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

  for (MachineInstr &MI : MBB) {
    // Free killed uses. Prior defs' replaceRegWith already made these physregs.
    for (const MachineOperand &MO : MI.all_uses()) {
      if (!MO.isKill())
        continue;
      assert(MO.getReg().isPhysical() && "Use not yet replaced");
      for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg()))
        LiveUnits.reset(static_cast<unsigned>(Unit));
    }

    // Assign defs: pick first free physreg from the effective set.
    for (const MachineOperand &MO : MI.all_defs()) {
      Register Reg = MO.getReg();
      assert(Reg.isVirtual() && "Def already replaced");

      const BitVector &Eff = EffectiveRC[Reg];

      MCPhysReg Assigned = 0;
      for (unsigned PhysReg : Eff.set_bits()) {
        bool Free =
            llvm::none_of(TRI->regunits(MCPhysReg(PhysReg)), [&](MCRegUnit U) {
              return LiveUnits.test(static_cast<unsigned>(U));
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
