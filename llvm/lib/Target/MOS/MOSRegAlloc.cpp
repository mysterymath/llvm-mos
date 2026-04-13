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

  /// EffectiveRC register sets accumulated across all allocations.
  /// Indexed by virtual register. Used by assignRegisters to pick physregs.
  IndexedMap<BitVector, VirtReg2IndexFunctor> EffectiveRC;
  void allocate(iterator_range<MBBIterator> MIs, LiveSet &Live);
  BitVector allocatableRegs(const TargetRegisterClass *RC);
  unsigned computeSqueeze(Register R, const LiveSet &Live);
  unsigned maxRegInterference(const BitVector &DefRegs,
                              const BitVector &LiveRegs);
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
  for (MachineInstr &MI : MBB) {
    MICluster[&MI] = Clusters.size();
    Cluster C(make_range(MI.getIterator(), std::next(MI.getIterator())));
    allocate(C.Range, C.LiveOut);
    Clusters.push_back(std::move(C));
  }
}

/// Allocate register space for instructions in the given range, ensuring
/// colorability via worst^1 squeeze with one-level narrowing. Live is the
/// set of live registers entering the range; it is updated in place.
/// Effective register sets are written to this->EffectiveRC.
void MOSRegAlloc::allocate(iterator_range<MBBIterator> MIs, LiveSet &Live) {
  for (MachineInstr &MI : MIs) {
    // Retire killed uses.
    for (const MachineOperand &MO : MI.all_uses()) {
      if (!MO.isKill())
        continue;
      Live.erase(MO.getReg());
    }

    // Add new defs and ensure slack >= 1.
    for (const MachineOperand &MO : MI.all_defs()) {
      Register R = MO.getReg();

      EffectiveRC[R] = allocatableRegs(MRI->getRegClass(R));
      Live.insert(R);

      unsigned Squeeze = computeSqueeze(R, Live);
      int Slack = EffectiveRC[R].count() - Squeeze;

      // Narrowing: shrink an overlapping live value's effective set.
      while (Slack < 1) {
        BitVector DAlias = aliasSet(EffectiveRC[R]);

        // Pick the first narrowable live value (deterministic: smallest Reg).
        Register NarrowReg;
        for (Register Other : Live) {
          if (Other == R)
            continue;
          if (maxRegInterference(EffectiveRC[R], EffectiveRC[Other]) == 0)
            continue;
          BitVector Remaining = EffectiveRC[Other];
          Remaining.reset(DAlias);
          if (Remaining.none())
            continue;
          if (!NarrowReg.isValid() || Other < NarrowReg)
            NarrowReg = Other;
        }
        assert(NarrowReg.isValid() &&
               "Allocation failed: no value can be narrowed");

        // Narrow: remove all registers that alias d's effective set.
        EffectiveRC[NarrowReg].reset(DAlias);
        LLVM_DEBUG(dbgs() << "    Narrow " << printReg(NarrowReg, TRI) << " to "
                          << EffectiveRC[NarrowReg].count() << " regs\n");

        // One-level check: assert the narrowed value still has slack >= 1.
        unsigned USqueeze = 0;
        for (Register Other2 : Live) {
          if (Other2 == NarrowReg)
            continue;
          USqueeze +=
              maxRegInterference(EffectiveRC[NarrowReg], EffectiveRC[Other2]);
        }
        assert((int)(EffectiveRC[NarrowReg].count() - USqueeze) >= 1 &&
               "Narrowed value lost colorability");

        Squeeze = computeSqueeze(R, Live);
        Slack = EffectiveRC[R].count() - Squeeze;
      }

      LLVM_DEBUG(dbgs() << "    Slack for " << printReg(R, TRI) << " ("
                        << TRI->getRegClassName(MRI->getRegClass(R))
                        << "): " << Slack << " (eff=" << EffectiveRC[R].count()
                        << ")\n");
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

/// Sum of maxRegInterference for R against all other live values.
unsigned MOSRegAlloc::computeSqueeze(Register R, const LiveSet &Live) {
  unsigned Squeeze = 0;
  for (Register Other : Live) {
    if (Other == R)
      continue;
    Squeeze += maxRegInterference(EffectiveRC[R], EffectiveRC[Other]);
  }
  return Squeeze;
}

/// The max number of registers in DefRegs that a single register from
/// LiveRegs can block via aliasing.
unsigned MOSRegAlloc::maxRegInterference(const BitVector &DefRegs,
                                         const BitVector &LiveRegs) {
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
