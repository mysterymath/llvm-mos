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
#include "llvm/ADT/DenseMap.h"
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

using Cluster = iterator_range<MachineBasicBlock::iterator>;

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

  /// Cluster state: each cluster is a contiguous range in the MBB.
  SmallVector<Cluster, 0> Clusters;
  DenseMap<MachineInstr *, unsigned> InstrCluster;

  void validate(MachineBasicBlock &MBB);
  void initClusters(MachineBasicBlock &MBB);
  void mergeClusters(MachineBasicBlock &MBB);

  /// Active (live) vregs mapped to their register class.
  DenseMap<Register, const TargetRegisterClass *> Active;
  void checkColorability(MachineInstr *MI);

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
  assert(MBB.livein_empty() && "Block liveins not yet supported");

  validate(MBB);
  initClusters(MBB);
  mergeClusters(MBB);
  assignRegisters(MBB);
  MRI->clearVirtRegs();
  LLVM_DEBUG(dbgs() << "MOS RegAlloc: done\n");
  return true;
}

void MOSRegAlloc::validate(MachineBasicBlock &MBB) {
  for (MachineInstr &MI : MBB) {
    assert(!MI.isTerminator() && "Terminators not yet supported");
    assert(!MI.isCopy() && "COPY instructions not yet supported");
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
      if (MO.isUse()) {
        assert(!MO.isUndef() && "Undef uses not yet supported");
        assert(MO.isKill() && "Non-kill virtual reg uses not yet supported");
      }
      if (MO.isDef())
        assert(!MO.isDead() && "Dead virtual reg defs not yet supported");
    }
  }
}

/// Create one singleton cluster per instruction.
void MOSRegAlloc::initClusters(MachineBasicBlock &MBB) {
  Clusters.clear();
  InstrCluster.clear();
  for (auto It = MBB.begin(), E = MBB.end(); It != E; ++It) {
    unsigned Idx = Clusters.size();
    Clusters.push_back(make_range(It, std::next(It)));
    InstrCluster[&*It] = Idx;
  }
}

/// Merge clusters along vreg def-use edges until one remains.
/// Splices instructions in the MBB so it ends up in final schedule order.
void MOSRegAlloc::mergeClusters(MachineBasicBlock &MBB) {
  unsigned NumActive = Clusters.size();

  // For each vreg, merge the def cluster with the (single) use cluster.
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;

    // Collect uses; assert single use.
    MachineInstr *UseMI = nullptr;
    for (MachineInstr &MI : MRI->use_nodbg_instructions(R)) {
      assert(!UseMI && "Multiple uses not yet supported");
      UseMI = &MI;
    }
    assert(UseMI && "Vreg has no use");

    unsigned DefC = InstrCluster[MRI->getVRegDef(R)];
    unsigned UseC = InstrCluster[UseMI];

    if (DefC == UseC)
      continue; // Already in same cluster.

    LLVM_DEBUG(dbgs() << "  Merge cluster " << UseC << " into " << DefC
                      << " for " << printReg(R, TRI) << "\n");

    // Merge UseC into DefC right after DefC's last instruction
    Cluster &Def = Clusters[DefC];
    Cluster &Use = Clusters[UseC];
    if (Def.end() != Use.begin())
      MBB.splice(Def.end(), &MBB, Use.begin(), Use.end());
    for (MachineInstr &MI : Use)
      InstrCluster[&MI] = DefC;
    Def = make_range(Def.begin(), Use.end());
    Use = make_range(Use.end(), Use.end());
    NumActive--;

    // Verify colorability of the merged cluster instruction by instruction.
    Active.clear();
    for (MachineInstr &MI : Def)
      checkColorability(&MI);
  }

  assert(NumActive == 1 && "Not all clusters merged into one");

  // MBB is now in final schedule order.
  LLVM_DEBUG({
    dbgs() << "  Final schedule:\n";
    for (MachineInstr &MI : MBB)
      dbgs() << "    " << MI;
  });
}

/// Compute worst^1(N, C): the maximum number of registers in N that a single
/// register from C can block via aliasing.
/// (Smith, Ramsey, Holloway, PLDI'04, equation 3 with m=1.)
static unsigned worst1(const TargetRegisterClass *N,
                       const TargetRegisterClass *C,
                       const TargetRegisterInfo *TRI) {
  unsigned Max = 0;
  for (MCPhysReg CReg : *C) {
    unsigned Blocked = 0;
    for (MCPhysReg NReg : *N)
      if (TRI->regsOverlap(NReg, CReg))
        Blocked++;
    Max = std::max(Max, Blocked);
  }
  return Max;
}

/// Check that adding MI to the schedule preserves colorability
/// (squeeze < |class|). Updates Active: retires killed uses, adds new defs.
///
/// Uses the approximation worst^m(N,C) <= m * worst^1(N,C) (PLDI'04, eq. 11).
/// This may overcount when neighbor classes overlap (the full Z(n,R) with
/// saturation bounds would fix that).
void MOSRegAlloc::checkColorability(MachineInstr *MI) {
  // Retire killed uses.
  for (const MachineOperand &MO : MI->all_uses())
    Active.erase(MO.getReg());

  // Add new defs and check squeeze.
  for (const MachineOperand &MO : MI->all_defs()) {
    Register R = MO.getReg();
    const TargetRegisterClass *RC = MRI->getRegClass(R);

    // Group active vregs by class.
    DenseMap<const TargetRegisterClass *, unsigned> ClassCount;
    for (auto &[Other, OtherRC] : Active)
      ClassCount[OtherRC]++;

    // squeeze <= sum_C degree(C) * worst^1(RC, C)
    unsigned Squeeze = 0;
    for (auto &[OtherRC, Count] : ClassCount)
      Squeeze += Count * worst1(RC, OtherRC, TRI);

    assert(Squeeze < RC->getNumRegs() &&
           "Colorability check failed: squeeze >= class size");
    LLVM_DEBUG(dbgs() << "    Squeeze for " << printReg(R, TRI) << " ("
                      << TRI->getRegClassName(RC) << "): " << Squeeze
                      << " / " << RC->getNumRegs() << "\n");

    Active[R] = RC;
  }
}

/// Assign physical registers top-down greedily, replacing vregs in place.
void MOSRegAlloc::assignRegisters(MachineBasicBlock &MBB) {
  BitVector LiveUnits(TRI->getNumRegUnits());

  for (MachineInstr &MI : MBB) {
    // Free killed uses. Prior defs' replaceRegWith already made these physregs.
    for (const MachineOperand &MO : MI.all_uses()) {
      assert(MO.getReg().isPhysical() && "Use not yet replaced");
      for (MCRegUnit Unit : TRI->regunits(MO.getReg().asMCReg()))
        LiveUnits.reset(static_cast<unsigned>(Unit));
    }

    // Assign defs: pick first free physreg, replace all occurrences.
    for (const MachineOperand &MO : MI.all_defs()) {
      Register Reg = MO.getReg();
      assert(Reg.isVirtual() && "Def already replaced");
      const TargetRegisterClass *RC = MRI->getRegClass(Reg);

      MCPhysReg Assigned = 0;
      for (MCPhysReg PhysReg : RCI.getOrder(RC)) {
        bool Free = llvm::none_of(TRI->regunits(PhysReg), [&](MCRegUnit U) {
          return LiveUnits.test(static_cast<unsigned>(U));
        });
        if (Free) {
          Assigned = PhysReg;
          break;
        }
      }
      assert(Assigned && "No free register in class");
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
