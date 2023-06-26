//===-- MOSReglloc.cpp - MOS Register Allocation -------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocator.
//
//===----------------------------------------------------------------------===//
//
// The program is assumed on entry to be in SSA form, with all subregister
// definitions performed by INSERT_SUBREG operations, and with critical edges
// split. To begin, COPY operations between vregs are all forwarded, such
// that the only remaining copy operations are between pregs and vregs.

// This isn't technically legal, as the register classes of definitions are not
// properly constrained by their uses. This is intentional, as vregs are used as
// a global value numbering, not as real temporary variables.

// Tied operands are handles specially; the tied use is required to be killed.
// This is achieved by the earlier SSA pass. The new SSA value formed by the
// tied def is considered an extension of the use; they must be assigned to the
// same location around the instruction. This does not interfere with the SSA
// properties used to produce the initial allocation.

// We consider there to be two "program points" between every two instructions
// and between an instruction and the basic block begin or end. One program
// point immediately follows the instruction, and one immediately preceeds the
// next. This accomidates arbirtrary instruction insertion between the two
// points to transform from the preceeding allocation to the next.

// First, a Hack-style SSA allocation is performed to assign values to imaginary
// registers. These form "backups" for values that cannot be placed in
// architectural registers. Some slight modifications need to be made, since
// instructions may not have imaginary registers in their register classes. For
// such instructions that define values, the live range of the imaginary
// register is considered to begin at the second point after the instruction, to
// allow for the copy. Similarly, for instructions that kill values, the value
// is considered killed after the first of the points beefore the use. This
// again accomodates the copy.

// Now, a cost model is established for tree-width dynamic programming. Given a
// pair of program points, and allocations for each, there is an integral cost.
// Use and earlyclobber def register classes hard-constrain the preceeding
// program point, while def register classes and regmasks hard-constrain the
// following program point.

#include "MOSRegAlloc.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominanceFrontier.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterClassInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <optional>
#include <vector>

#define DEBUG_TYPE "mos-regalloc"

using namespace llvm;

namespace {

struct NextUseDist {
  unsigned NumLoopExits;
  unsigned NumInstrs;

  bool operator<(const NextUseDist &R) const {
    if (NumLoopExits < R.NumLoopExits)
      return true;
    if (NumLoopExits > R.NumLoopExits)
      return false;
    return NumInstrs < R.NumInstrs;
  }
};

// Tracks the worst-case imaginary register pressure as values are added and
// removed from the allocated set. This accounts for the potential internal
// fragmentation of the imaginary register file due to allocation and
// deallocation patterns.
class ImagPressure {
  unsigned NumImag8Pairs = 0;
  unsigned NumImag8Unpaired = 0;
  unsigned NumImag16 = 0;

  unsigned NumImag8PairsCSR = 0;
  unsigned NumImag8UnpairedCSR = 0;
  unsigned NumImag16CSR = 0;

public:
  bool numImag16Needed() const {
    return NumImag16 + NumImag16CSR + NumImag8Pairs + NumImag8PairsCSR +
           NumImag8Unpaired + NumImag8UnpairedCSR;
  }

  bool numImag16CSRNeeded() const {
    return NumImag16CSR + NumImag8PairsCSR + NumImag8UnpairedCSR;
  }

  void addImag8() {
    if (NumImag8Unpaired) {
      NumImag8Unpaired--;
      NumImag8Pairs++;
    } else {
      NumImag8Unpaired++;
    }
  }

  void addImag8CSR() {
    if (NumImag8UnpairedCSR) {
      NumImag8UnpairedCSR--;
      NumImag8PairsCSR++;
    } else {
      NumImag8UnpairedCSR++;
    }
  }

  void addImag16() { NumImag16++; }

  void addImag16CSR() { NumImag16CSR++; }

  void removeImag8() {
    // In the worst case, a paired register is the one removed.
    if (NumImag8Pairs) {
      NumImag8Pairs--;
      NumImag8Unpaired++;
    } else {
      NumImag8Unpaired--;
    }
  }

  void removeImag8CSR() {
    if (NumImag8PairsCSR) {
      NumImag8PairsCSR--;
      NumImag8UnpairedCSR++;
    } else {
      NumImag8UnpairedCSR--;
    }
  }

  void removeImag16() { NumImag16--; }

  void removeImag16CSR() { NumImag16CSR--; }
};

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    llvm::initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  MachineFunctionProperties getRequiredProperties() const override;
  void getAnalysisUsage(AnalysisUsage &AU) const override;
  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  LiveVariables *LV;
  MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineDominanceFrontier *MDF;
  const MachineDominatorTree *MDT;
  const MachineLoopInfo *MLI;
  MachineFunction *MF;
  RegisterClassInfo RCI;

  // Values live across calls that clobber imaginary registers.
  DenseSet<Register> CSRVals;

  // Map from value to imaginary register
  DenseMap<Register, Register> ImagAlloc;

  unsigned NumImag16Avail;
  unsigned NumImag16CSRAvail;

  DenseMap<std::pair<Register, const MachineBasicBlock *>, NextUseDist>
      NextUseDists;

  void countAvailImag16Regs();
  void findCSRVals(const MachineDomTreeNode &MDTN,
                   const SmallSet<Register, 8> &DomLiveOutVals = {});
  void allocateImagRegs();
  void assignImagRegs(const MachineDomTreeNode &MDTN,
                      const SmallSet<Register, 8> &DomLiveOutVals = {});

  bool nearerNextUse(Register Left, Register Right,
                     const MachineBasicBlock &MBB,
                     MachineBasicBlock::const_iterator Pos) const;
  std::optional<NextUseDist>
  succNextUseDist(Register R, const MachineBasicBlock &MBB,
                  const MachineBasicBlock &Succ) const;
  void recomputeNextUseDists(Register R);

  const TargetRegisterClass *getOperandRegClass(const MachineOperand &MO) const;
  bool isDeadMI(const MachineInstr &MI) const;
  void addDefPressure(Register R, ImagPressure *IP) const;
  void removeKillPressure(Register R, ImagPressure *IP) const;
  bool isImagPressureOver(const ImagPressure &IP) const;
};

} // namespace

MachineFunctionProperties MOSRegAlloc::getRequiredProperties() const {
  return MachineFunctionProperties().set(
      MachineFunctionProperties::Property::IsSSA);
}

void MOSRegAlloc::getAnalysisUsage(AnalysisUsage &AU) const {
  MachineFunctionPass::getAnalysisUsage(AU);
  AU.addRequired<LiveVariables>();
  AU.addRequired<MachineDominanceFrontier>();
  AU.addPreserved<MachineDominanceFrontier>();
  AU.addRequired<MachineDominatorTree>();
  AU.addPreserved<MachineDominatorTree>();
  AU.addRequired<MachineLoopInfo>();
  AU.addPreserved<MachineLoopInfo>();
}

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  // TODO: Subregister liveness.
  this->MF = &MF;

  LLVM_DEBUG(dbgs() << "\n# MOS Register Allocator: " << MF.getName()
                    << "\n\n");

  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MRI->getTargetRegisterInfo();
  MDF = &getAnalysis<MachineDominanceFrontier>();
  MDT = &getAnalysis<MachineDominatorTree>();
  MLI = &getAnalysis<MachineLoopInfo>();
  LV = &getAnalysis<LiveVariables>();

  MF.dump();

  LLVM_DEBUG(dbgs() << "## Coalesce away copies.\n");

  // Temporarily coalesce away all copies. This makes the register classes
  // invalid, but true copies will be inserted as allocation proceeds to
  // restore them.
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;
    MachineInstr *MI = MRI->getUniqueVRegDef(R);
    if (!MI->isCopy())
      continue;
    MachineOperand &Src = MI->getOperand(1);
    if (!Src.getReg().isVirtual())
      continue;
    LLVM_DEBUG(dbgs() << "Coalescing: " << *MI);
    for (MachineOperand &Use :
         make_early_inc_range(MRI->use_nodbg_operands(R))) {
      LLVM_DEBUG(dbgs() << "Use MI: " << *Use.getParent());
      unsigned SubRegIdx = 0;
      if (Src.getSubReg()) {
        SubRegIdx = Src.getSubReg();
        if (Use.getSubReg())
          SubRegIdx = TRI->composeSubRegIndices(SubRegIdx, Use.getSubReg());
      }
      Use.setReg(Src.getReg());
      Use.setSubReg(SubRegIdx);
    }
    MI->eraseFromParent();
  }

  MF.dump();

  MRI->freezeReservedRegs(MF);
  RCI.runOnMachineFunction(MF);

  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; I = I + 1) {
    Register R = Register::index2VirtReg(I);
    if (MRI->use_nodbg_empty(R))
      continue;
    recomputeNextUseDists(R);
  }

  findCSRVals(*MDT->getRootNode());
  LLVM_DEBUG({
    dbgs() << "Values live across calls:\n";
    for (Register R : CSRVals)
      dbgs() << '\t' << printReg(R) << '\n';
  });

  allocateImagRegs();
  ImagAlloc.clear();

  LLVM_DEBUG(dbgs() << "Assigning imaginary registers to each value:\n");
  assignImagRegs(*MDT->getRootNode());

  // Recompute liveness and kill dead instructions.
  for (MachineBasicBlock *MBB : post_order(&MF)) {
    bool RemovedAny;
    do {
      recomputeLivenessFlags(*MBB);
      RemovedAny = false;
      for (MachineInstr &MI : make_early_inc_range(*MBB)) {
        if (!isDeadMI(MI))
          continue;
        LLVM_DEBUG(dbgs() << "Remove dead MI: " << MI);
        MI.eraseFromParent();
        RemovedAny = true;
      }
    } while (RemovedAny);

    recomputeLiveIns(*MBB);
  }

  MRI->leaveSSA();
  MF.getProperties().set(MachineFunctionProperties::Property::NoPHIs);
  MRI->clearVirtRegs();

  return false;
}

void MOSRegAlloc::countAvailImag16Regs() {
  for (Register I = MOS::RS0, E = MOS::RS15 + 1; I != E; I = I + 1) {
    if (MRI->isReserved(I))
      continue;
    NumImag16Avail++;
    if (I >= MOS::RS10)
      NumImag16CSRAvail++;
  }
}

void MOSRegAlloc::findCSRVals(const MachineDomTreeNode &MDTN,
                              const SmallSet<Register, 8> &DomLiveOutVals) {

  const MachineBasicBlock &MBB = *MDTN.getBlock();

  SmallSet<Register, 8> LiveVals;
  for (Register R : DomLiveOutVals)
    if (LV->isLiveIn(R, *MDTN.getBlock()))
      LiveVals.insert(R);

  for (const MachineInstr &MI : MBB) {
    for (const MachineOperand &MO : MI.defs())
      if (MO.isEarlyClobber() && MO.getReg().isVirtual())
        LiveVals.insert(MO.getReg());
    for (const MachineOperand &MO : MI.uses())
      if (MO.isReg() && MO.isKill())
        LiveVals.erase(MO.getReg());

    for (const MachineOperand &MO : MI.operands())
      if (MO.isRegMask() && MO.clobbersPhysReg(MOS::RC2))
        for (Register R : LiveVals)
          CSRVals.insert(R);

    for (const MachineOperand &MO : MI.defs())
      if (!MO.isEarlyClobber() && MO.getReg().isVirtual())
        LiveVals.insert(MO.getReg());
    for (const MachineOperand &MO : MI.defs())
      if (MO.isDead())
        LiveVals.erase(MO.getReg());
  }

  for (const MachineDomTreeNode *Child : MDTN.children())
    findCSRVals(*Child, LiveVals);
}

void MOSRegAlloc::allocateImagRegs() {
  DenseMap<MachineBasicBlock *, SmallSet<Register, 8>> LiveOutVals;
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF->begin());

  for (MachineBasicBlock *MBB : RPOT) {
    LLVM_DEBUG(dbgs() << "Choosing allocated live in vals for MBB %bb."
                      << MBB->getNumber() << '\n');

    SmallSet<Register, 8> AllPredsLOV;
    SmallSet<Register, 8> SomePredLOV;
    bool IsFirst = true;
    for (MachineBasicBlock *Pred : MBB->predecessors()) {
      const auto &PredLOV = LiveOutVals[Pred];

      if (IsFirst) {
        for (Register R : PredLOV) {
          if (!LV->isLiveIn(R, *MBB))
            continue;
          AllPredsLOV.insert(R);
          SomePredLOV.insert(R);
        }
        IsFirst = false;
        continue;
      }

      SmallSet<Register, 8> ToRemove;
      for (Register R : AllPredsLOV)
        if (!PredLOV.contains(R))
          ToRemove.insert(R);
      for (Register R : ToRemove)
        AllPredsLOV.erase(R);

      for (Register R : PredLOV)
        if (LV->isLiveIn(R, *MBB))
          SomePredLOV.insert(R);
    }

    LLVM_DEBUG({
      dbgs() << "Candidates live out of all predecessors:\n";
      for (Register R : AllPredsLOV)
        dbgs() << printReg(R) << ' ';
      dbgs() << '\n';
    });

    SmallVector<Register> Candidates(SomePredLOV.begin(), SomePredLOV.end());
    llvm::sort(Candidates, [&](Register A, Register B) {
      // To avoid reloads, prefer to keep values allocated in all predecessors,
      // not just some.
      if (AllPredsLOV.contains(A) && !AllPredsLOV.contains(B))
        return true;
      if (!AllPredsLOV.contains(A) && AllPredsLOV.contains(B))
        return false;

      // Prefer values with nearer next uses.
      return nearerNextUse(A, B, *MBB, MBB->begin());
    });

    LLVM_DEBUG({
      dbgs() << "Ordered candidates:\n";
      for (Register R : Candidates)
        dbgs() << printReg(R) << ' ';
      dbgs() << '\n';
    });

    ImagPressure IP;
    Register *I = Candidates.begin();
    for (Register *E = Candidates.end(); I != E; ++I) {
      Register R = *I;
      ImagPressure NewIP = IP;
      addDefPressure(R, &NewIP);
      if (isImagPressureOver(NewIP))
        break;
      IP = NewIP;
    }

    Candidates.erase(I, Candidates.end());
    LLVM_DEBUG({
      dbgs() << "Chosen candidates:\n";
      for (Register R : Candidates)
        dbgs() << printReg(R) << ' ';
      dbgs() << '\n';
    });

    auto &LV = LiveOutVals[MBB];
    for (Register R : Candidates)
      LiveOutVals[MBB].insert(R);

    const auto Allocate = [&](Register R,
                              MachineBasicBlock::const_iterator Pos) {
      LLVM_DEBUG(dbgs() << "Allocating " << printReg(R) << '\n');
      LV.insert(R);
      ImagPressure NewIP = IP;
      addDefPressure(R, &NewIP);
      if (isImagPressureOver(NewIP)) {
        SmallVector<Register> Candidates(LV.begin(), LV.end());
        llvm::sort(Candidates, [&](Register A, Register B) {
          return nearerNextUse(A, B, *MBB, Pos);
        });
        do {
          Register Evicted = Candidates.back();
          Candidates.pop_back();
          LLVM_DEBUG(dbgs() << "Evicting " << printReg(Evicted) << '\n');
          report_fatal_error("Spill/reload not yet implemented");
          LV.erase(Evicted);
          removeKillPressure(Evicted, &IP);
          NewIP = IP;
          addDefPressure(R, &NewIP);
        } while (isImagPressureOver(NewIP));
      }
      IP = NewIP;
    };

    for (const MachineInstr &MI : *MBB) {
      LLVM_DEBUG(dbgs() << "Allocating " << MI);
      for (const MachineOperand &MO : MI.defs())
        if (MO.isEarlyClobber() && MO.getReg().isVirtual())
          Allocate(MO.getReg(), MI);
      for (const MachineOperand &MO : MI.uses()) {
        if (MO.isReg() && MO.isKill() && MO.getReg().isVirtual()) {
          LV.erase(MO.getReg());
          removeKillPressure(MO.getReg(), &IP);
        }
      }
      for (const MachineOperand &MO : MI.defs())
        if (!MO.isEarlyClobber() && MO.getReg().isVirtual())
          Allocate(MO.getReg(), MI);
      for (const MachineOperand &MO : MI.defs()) {
        if (MO.isDead() && MO.getReg().isVirtual()) {
          LV.erase(MO.getReg());
          removeKillPressure(MO.getReg(), &IP);
        }
      }
    }
  }
}

void MOSRegAlloc::assignImagRegs(const MachineDomTreeNode &MDTN,
                                 const SmallSet<Register, 8> &DomLiveOutVals) {
  const MachineBasicBlock &MBB = *MDTN.getBlock();

  SmallSet<Register, 8> LiveVals;
  for (Register R : DomLiveOutVals)
    if (LV->isLiveIn(R, *MDTN.getBlock()))
      LiveVals.insert(R);

  const auto Assign = [&](Register R) {
    const TargetRegisterClass *RC = MRI->getRegClass(R);
    bool IsCSR = CSRVals.contains(R);
    Register I, E;
    if (IsCSR)
      I = (RC == &MOS::Imag16RegClass) ? MOS::RS10 : MOS::RC20;
    else
      I = (RC == &MOS::Imag16RegClass) ? MOS::RS1 : MOS::RC2;
    E = (RC == &MOS::Imag16RegClass) ? MOS::RS15 + 1 : MOS::RC31 + 1;

    for (; I != E; I = I + 1) {
      if (MRI->isReserved(I))
        continue;
      if (llvm::none_of(LiveVals, [&](Register V) {
            return TRI->regsOverlap(I, ImagAlloc[V]);
          })) {
        LLVM_DEBUG(dbgs() << printReg(R) << " -> " << printReg(I, TRI) << '\n');
        ImagAlloc[R] = I;
        LiveVals.insert(R);
        return;
      }
    }
    report_fatal_error("ran out of callee-saved registers");
  };

  for (const MachineInstr &MI : MBB) {
    for (const MachineOperand &MO : MI.defs())
      if (MO.isEarlyClobber() && MO.getReg().isVirtual())
        Assign(MO.getReg());
    for (const MachineOperand &MO : MI.uses())
      if (MO.isReg() && MO.isKill())
        LiveVals.erase(MO.getReg());
    for (const MachineOperand &MO : MI.defs())
      if (!MO.isEarlyClobber() && MO.getReg().isVirtual())
        Assign(MO.getReg());
    for (const MachineOperand &MO : MI.defs())
      if (MO.isDead())
        LiveVals.erase(MO.getReg());
  }

  for (const MachineDomTreeNode *Child : MDTN.children())
    assignImagRegs(*Child, LiveVals);
}

bool MOSRegAlloc::nearerNextUse(Register Left, Register Right,
                                const MachineBasicBlock &MBB,
                                MachineBasicBlock::const_iterator Pos) const {
  for (MachineBasicBlock::const_iterator I = Pos, E = MBB.end(); I != E; ++I) {
    bool ReadsLeft = I->readsVirtualRegister(Left);
    bool ReadsRight = I->readsVirtualRegister(Right);
    if (ReadsLeft || ReadsRight)
      return ReadsLeft && !ReadsRight;
  }

  std::optional<NextUseDist> BestLeft;
  std::optional<NextUseDist> BestRight;
  for (MachineBasicBlock *Succ : MBB.successors()) {
    std::optional<NextUseDist> LeftDist = succNextUseDist(Left, MBB, *Succ);
    if (LeftDist && (!BestLeft || LeftDist < *BestLeft))
      BestLeft = LeftDist;
    std::optional<NextUseDist> RightDist = succNextUseDist(Right, MBB, *Succ);
    if (RightDist && (!BestRight || RightDist < *BestRight))
      BestRight = RightDist;
  }
  assert((BestLeft || BestRight) && "Must be live out.");
  if (!BestRight)
    return true;
  return BestLeft && *BestLeft < *BestRight;
}

std::optional<NextUseDist>
MOSRegAlloc::succNextUseDist(Register R, const MachineBasicBlock &MBB,
                             const MachineBasicBlock &Succ) const {
  unsigned NumLoopExits = 0;
  if (MLI->getLoopFor(&Succ) != MLI->getLoopFor(&MBB) &&
      MLI->getLoopDepth(&Succ) <= MLI->getLoopDepth(&MBB))
    NumLoopExits = 1;
  for (const MachineInstr &MI : Succ.phis())
    for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2)
      if (MI.getOperand(I).getReg() == R &&
          MI.getOperand(I + 1).getMBB() == &MBB)
        return NextUseDist{NumLoopExits, 0};
  auto It = NextUseDists.find({R, &Succ});
  if (It == NextUseDists.end())
    return std::nullopt;
  return NextUseDist{It->second.NumLoopExits + NumLoopExits,
                     It->second.NumInstrs};
}

void MOSRegAlloc::recomputeNextUseDists(Register R) {
  dbgs() << "Compute next uses: " << printReg(R) << '\n';

  for (const MachineBasicBlock &MBB : *MF)
    NextUseDists.erase({R, &MBB});

  DenseSet<const MachineBasicBlock *> UseBlocks;
  for (const MachineInstr &MI : MRI->use_nodbg_instructions(R))
    if (!MI.isPHI() && LV->isLiveIn(R, *MI.getParent()))
      UseBlocks.insert(MI.getParent());

  std::deque<const MachineBasicBlock *> Worklist;
  DenseSet<const MachineBasicBlock *> Active;
  for (const MachineBasicBlock *MBB : UseBlocks) {
    for (const auto &[I, MI] : llvm::enumerate(*MBB)) {
      if (MI.isPHI())
        continue;
      if (MI.readsVirtualRegister(R)) {
        auto Res = NextUseDists.try_emplace(
            {R, MBB},
            NextUseDist{/*NumLoopExits=*/0, /*NumInstrs=*/(unsigned)I});
        (void)Res;
        assert(Res.second && "Should have inserted.");

        for (const MachineBasicBlock *Pred : MBB->predecessors())
          if (LV->isLiveIn(R, *Pred) && Active.insert(Pred).second)
            Worklist.push_back(Pred);
        break;
      }
    }
  }

  while (!Worklist.empty()) {
    const MachineBasicBlock &MBB = *Worklist.front();
    Worklist.pop_front();
    Active.erase(&MBB);

    std::optional<NextUseDist> BestDist;
    auto It = NextUseDists.find({R, &MBB});
    if (It != NextUseDists.end())
      BestDist = It->second;

    bool Improved = false;
    for (const MachineBasicBlock *Succ : MBB.successors()) {
      std::optional<NextUseDist> SuccDist = succNextUseDist(R, MBB, *Succ);
      if (!SuccDist)
        continue;
      SuccDist->NumInstrs += MBB.size();
      if (!BestDist || *SuccDist < *BestDist) {
        Improved = true;
        BestDist = SuccDist;
      }
    }

    if (Improved) {
      NextUseDists[{R, &MBB}] = *BestDist;
      for (const MachineBasicBlock *Pred : MBB.predecessors())
        if (LV->isLiveIn(R, *Pred) && Active.insert(Pred).second)
          Worklist.push_back(Pred);
    }
  }

  for (const MachineBasicBlock &MBB : *MF) {
    auto It = NextUseDists.find({R, &MBB});
    if (It == NextUseDists.end())
      continue;
    dbgs() << "  MBB %bb." << MBB.getNumber() << ": {"
           << It->second.NumLoopExits << ',' << It->second.NumInstrs << "}\n";
  }
}

const TargetRegisterClass *
MOSRegAlloc::getOperandRegClass(const MachineOperand &MO) const {
  const TargetRegisterClass *RC =
      MO.getParent()->getRegClassConstraint(MO.getOperandNo(), TII, TRI);
  if (!RC)
    RC = TRI->getLargestLegalSuperClass(MRI->getRegClass(MO.getReg()),
                                        *MO.getParent()->getMF());
  return RC;
}

/// Returns whether an instruction is dead. Conservative, but depends on
/// liveness flags for good results.
bool MOSRegAlloc::isDeadMI(const MachineInstr &MI) const {
  if (MI.isCall() || MI.isBranch() || MI.isReturn() || MI.mayLoadOrStore())
    return false;

  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.isDef() && !MO.isDead())
      return false;
  }

  return true;
}

void MOSRegAlloc::addDefPressure(Register R, ImagPressure *IP) const {
  bool IsCSR = CSRVals.contains(R);
  if (MRI->getRegClass(R) == &MOS::Imag16RegClass) {
    if (IsCSR)
      IP->addImag16CSR();
    else
      IP->addImag16();
  } else {
    if (IsCSR)
      IP->addImag16CSR();
    else
      IP->addImag16();
  }
}

void MOSRegAlloc::removeKillPressure(Register R, ImagPressure *IP) const {
  bool IsCSR = CSRVals.contains(R);
  if (MRI->getRegClass(R) == &MOS::Imag16RegClass) {
    if (IsCSR)
      IP->removeImag16CSR();
    else
      IP->removeImag16();
  } else {
    if (IsCSR)
      IP->removeImag16CSR();
    else
      IP->removeImag16();
  }
}

bool MOSRegAlloc::isImagPressureOver(const ImagPressure &IP) const {
  return IP.numImag16Needed() > NumImag16Avail ||
         IP.numImag16CSRNeeded() > NumImag16CSRAvail;
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocator", false, false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
