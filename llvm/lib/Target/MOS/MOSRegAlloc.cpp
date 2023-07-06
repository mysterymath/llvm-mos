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
#include "llvm/CodeGen/ISDOpcodes.h"
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
#include "llvm/ExecutionEngine/Orc/Core.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <memory>
#include <optional>
#include <pthread.h>
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
  unsigned numImag16Needed() const {
    return NumImag16 + NumImag8Pairs + NumImag8Unpaired;
  }

  unsigned numImag16CSRNeeded() const {
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

  void dump() const {
    dbgs() << "Imag16 Pressure: CSR: " << numImag16CSRNeeded()
           << " Non-CSR: " << numImag16Needed() << '\n';
  }
};

struct Position {
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator Pos;

  // False if before any moves at this position, true if after any moves at this
  // position. If before moves, the position is constrained by the
  // post-conditions of the previous instruction. If after moves, the position
  // is constrained by the pre-conditions of the next instruction.
  bool AfterMoves;

  bool operator==(Position Other) const {
    return MBB == Other.MBB && Pos == Other.Pos &&
           AfterMoves == Other.AfterMoves;
  }
};

struct Node {
  SmallVector<Position> Positions;
  SmallVector<Node *> Children;
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

  // Values where storing in an imaginary register is always slower than remat.
  DenseSet<Register> NeverImagVals;

  // Map from value to imaginary register
  DenseMap<Register, Register> ImagAlloc;

  DenseMap<const MachineBasicBlock *, DenseSet<Register>> MBBInRegVals;
  DenseMap<const MachineBasicBlock *, DenseSet<Register>> MBBOutRegVals;

  unsigned NumImag16Avail;
  unsigned NumImag16CSRAvail;

  DenseMap<std::pair<Register, const MachineBasicBlock *>, NextUseDist>
      NextUseDists;

  // <MaxImag16, MaxImag16CSR> for each loop
  DenseMap<const MachineLoop *, std::pair<unsigned, unsigned>>
      MaxLoopImagPressure;

  // Values used anywhere within each loop
  DenseMap<const MachineLoop *, DenseSet<Register>> LoopUsedVals;

  DenseMap<Position, SmallSet<Register, 8>> PositionLiveVals;
  DenseMap<Position, unsigned> PositionIndices;

  std::vector<Node> Tree;

  void countAvailImag16Regs();
  void findCSRVals(const MachineDomTreeNode &MDTN,
                   const SmallSet<Register, 8> &DomLiveOutVals = {});
  void findNeverImagVals();
  void scanLoops();
  void assignImagRegs();
  void scanPositions();
  void decomposeToTree();

  bool nearerNextUse(Register Left, Register Right,
                     const MachineBasicBlock &MBB,
                     MachineBasicBlock::const_iterator Pos) const;

  bool betterToAssign(Register Left, Register Right,
                      const MachineBasicBlock &MBB,
                      MachineBasicBlock::const_iterator Pos);

  std::optional<NextUseDist>
  succNextUseDist(Register R, const MachineBasicBlock &MBB,
                  const MachineBasicBlock &Succ) const;
  void recomputeNextUseDists(Register R);

  const TargetRegisterClass *getOperandRegClass(const MachineOperand &MO) const;
  bool isDeadMI(const MachineInstr &MI) const;
  void addDefPressure(Register R, ImagPressure *IP) const;
  void removeKillPressure(Register R, ImagPressure *IP) const;
  bool isImagPressureOver(const ImagPressure &IP) const;
  void rewriteVReg(Register From, Register To, MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator Pos);

  void dumpTree(Node *Root = nullptr, unsigned Indent = 0);
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

  // Recompute liveness information
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;
    LV->recomputeForSingleDefVirtReg(R);
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
      dbgs() << printReg(R) << ' ';
    dbgs() << '\n';
  });

  findNeverImagVals();
  LLVM_DEBUG({
    dbgs() << "Values that should always be rematerialized:\n";
    for (Register R : NeverImagVals)
      dbgs() << printReg(R) << ' ';
    dbgs() << '\n';
  });

  countAvailImag16Regs();
  scanLoops();
  ImagAlloc.clear();

  assignImagRegs();

  // TODO: Recompute liveness flags and kill dead instructions. Still in SSA.

  scanPositions();
  decomposeToTree();

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
  NumImag16Avail = 0;
  NumImag16CSRAvail = 0;
  for (Register I = MOS::RS0, E = MOS::RS15 + 1; I != E; I = I + 1) {
    if (MRI->isReserved(I))
      continue;
    if (I >= MOS::RS10)
      NumImag16CSRAvail++;
    else
      NumImag16Avail++;
  }
  LLVM_DEBUG(dbgs() << "Number of non-CSR Imag16 avail: " << NumImag16Avail
                    << '\n');
  LLVM_DEBUG(dbgs() << "Number of CSR Imag16 avail: " << NumImag16CSRAvail
                    << '\n');
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

void MOSRegAlloc::findNeverImagVals() {
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (MRI->reg_nodbg_empty(R))
      continue;

    MachineInstr *MI = MRI->getUniqueVRegDef(R);
    if (!TII->isTriviallyReMaterializable(*MI))
      continue;

    bool PotentialImagUse = false;
    for (MachineOperand &MO : MRI->use_nodbg_operands(R)) {
      if (MO.getParent()->isPHI()) {
        PotentialImagUse = true;
        break;
      }

      const TargetRegisterClass *RC =
          MI->getRegClassConstraint(MO.getOperandNo(), TII, TRI);
      if (!RC)
        continue;
      if (RC == &MOS::Imag16RegClass || RC == &MOS::Anyi1RegClass ||
          RC->hasSubClassEq(&MOS::Imag8RegClass)) {
        PotentialImagUse = true;
        break;
      }
    }
    if (!PotentialImagUse)
      NeverImagVals.insert(R);
  }
}

// Find the maximum register pressure and used variables for each loop. This
// information is necessary for evaluating the priorities of registers on loop
// entry: those used within the loop have first pick, and other registers are
// only included if they have a chance of surviving the loop.
void MOSRegAlloc::scanLoops() {
  for (const MachineBasicBlock &MBB : *MF) {
    MachineLoop *ML = MLI->getLoopFor(&MBB);
    if (!ML)
      continue;

    ImagPressure IP;

    const auto UpdateMaxImag16 = [&]() {
      for (const MachineLoop *CurML = ML; CurML;
           CurML = CurML->getParentLoop()) {
        auto Res = MaxLoopImagPressure.try_emplace(CurML, std::make_pair(0, 0));
        auto &[MaxImag16, MaxImag16CSR] = Res.first->second;
        MaxImag16 = std::max(MaxImag16, IP.numImag16Needed());
        MaxImag16CSR = std::max(MaxImag16CSR, IP.numImag16CSRNeeded());
      }
    };

    const auto InsertUsedVal = [&](Register R, const MachineLoop *ML) {
      for (; ML; ML = ML->getParentLoop())
        LoopUsedVals[ML].insert(R);
    };

    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
      Register R = Register::index2VirtReg(I);
      if (MRI->use_nodbg_empty(R) || !LV->isLiveIn(R, MBB))
        continue;
      addDefPressure(R, &IP);
    }

    UpdateMaxImag16();

    for (const MachineInstr &MI : MBB) {
      for (const MachineOperand &MO : MI.defs())
        if (MO.isEarlyClobber() && MO.getReg().isVirtual())
          addDefPressure(MO.getReg(), &IP);

      UpdateMaxImag16();

      if (MI.isPHI()) {
        for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
          const MachineOperand &ValMO = MI.getOperand(I);
          const MachineOperand &MBBMO = MI.getOperand(I + 1);
          if (!ValMO.isReg() || !ValMO.getReg().isVirtual())
            continue;
          InsertUsedVal(ValMO.getReg(), MLI->getLoopFor(MBBMO.getMBB()));
        }
      } else {
        for (const MachineOperand &MO : MI.uses()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          if (MO.isKill())
            removeKillPressure(MO.getReg(), &IP);
          InsertUsedVal(MO.getReg(), ML);
        }
      }

      for (const MachineOperand &MO : MI.defs())
        if (!MO.isEarlyClobber() && MO.getReg().isVirtual())
          addDefPressure(MO.getReg(), &IP);

      UpdateMaxImag16();

      for (const MachineOperand &MO : MI.defs())
        if (MO.isDead() && MO.getReg().isVirtual())
          removeKillPressure(MO.getReg(), &IP);
    }
  }

  LLVM_DEBUG({
    dbgs() << "Scanned loops:\n";
    for (const auto &[ML, MLIP] : MaxLoopImagPressure) {
      dbgs() << *ML;
      dbgs() << "Used vals:\n";
      for (Register R : LoopUsedVals[ML])
        dbgs() << printReg(R) << ' ';
      dbgs() << "\nMax pressure: Imag16: " << MLIP.first
             << " Imag16CSR: " << MLIP.second << "\n\n";
    }
  });
}

void MOSRegAlloc::assignImagRegs() {
  LLVM_DEBUG(dbgs() << "Assigning imaginary registers to each value.\n");
  ReversePostOrderTraversal<MachineBasicBlock *> RPOT(&*MF->begin());

  for (MachineBasicBlock *MBB : RPOT) {
    LLVM_DEBUG(dbgs() << "Choosing allocated live in vals for MBB %bb."
                      << MBB->getNumber() << '\n');

    // Values currently in imaginary registers
    auto &RegVals = MBBOutRegVals[MBB];

    const auto TryAssignFixed = [&](Register R, Register Phys) -> bool {
      if (MRI->isReserved(Phys))
        return false;
      if (llvm::any_of(RegVals, [&](Register V) {
            return TRI->regsOverlap(Phys, ImagAlloc[V]);
          }))
        return false;
      if (!ImagAlloc.contains(R))
        LLVM_DEBUG(dbgs() << printReg(R) << " -> " << printReg(Phys, TRI)
                          << '\n');
      LLVM_DEBUG(dbgs() << "Chose val: " << printReg(R) << '\n');
      ImagAlloc[R] = Phys;
      RegVals.insert(R);
      return true;
    };

    const auto TryAssign = [&](Register R) -> bool {
      // If the register is already assigned, just not currently in a register,
      // then there's only one choice for the register.
      auto It = ImagAlloc.find(R);
      if (It != ImagAlloc.end())
        return TryAssignFixed(R, It->second);

      const TargetRegisterClass *RC = MRI->getRegClass(R);
      bool IsCSR = CSRVals.contains(R);
      Register I, E;
      if (IsCSR)
        I = (RC == &MOS::Imag16RegClass) ? MOS::RS10 : MOS::RC20;
      else
        I = (RC == &MOS::Imag16RegClass) ? MOS::RS1 : MOS::RC2;
      E = (RC == &MOS::Imag16RegClass) ? MOS::RS15 + 1 : MOS::RC31 + 1;

      for (; I != E; I = I + 1)
        if (TryAssignFixed(R, I))
          return true;
      return false;
    };

    if (MLI->isLoopHeader(MBB)) {
      LLVM_DEBUG(dbgs() << "Loop header\n");

      SmallVector<Register> Candidates;
      const DenseSet<Register> &LUV =
          LoopUsedVals.find(MLI->getLoopFor(MBB))->second;
      for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
        Register R = Register::index2VirtReg(I);
        if (!MRI->use_nodbg_empty(R) && LV->isLiveIn(R, *MBB) &&
            !NeverImagVals.contains(R))
          Candidates.push_back(R);
      }
      llvm::sort(Candidates, [&](Register A, Register B) {
        bool AUsed = LUV.contains(A);
        bool BUsed = LUV.contains(B);
        if (AUsed && !BUsed)
          return true;
        if (!AUsed && BUsed)
          return false;
        return betterToAssign(A, B, *MBB, MBB->begin());
      });

      LLVM_DEBUG({
        dbgs() << "Candidates:\n";
        for (Register R : Candidates)
          dbgs() << printReg(R) << ' ';
        dbgs() << '\n';
      });

      LLVM_DEBUG(dbgs() << "Chosen candidates:\n");
      for (Register R : Candidates)
        TryAssign(R);
      LLVM_DEBUG(dbgs() << '\n');
    } else {
      SmallSet<Register, 8> AllPredsORV;
      SmallSet<Register, 8> SomePredORV;
      bool IsFirst = true;
      for (MachineBasicBlock *Pred : MBB->predecessors()) {
        const auto &PredORV = MBBOutRegVals[Pred];

        if (IsFirst) {
          for (Register R : PredORV) {
            if (!LV->isLiveIn(R, *MBB) || NeverImagVals.contains(R))
              continue;
            AllPredsORV.insert(R);
            SomePredORV.insert(R);
          }
          IsFirst = false;
          continue;
        }

        SmallSet<Register, 8> ToRemove;
        for (Register R : AllPredsORV)
          if (!PredORV.contains(R))
            ToRemove.insert(R);
        for (Register R : ToRemove)
          AllPredsORV.erase(R);

        for (Register R : PredORV)
          if (LV->isLiveIn(R, *MBB) && !NeverImagVals.contains(R))
            SomePredORV.insert(R);
      }

      LLVM_DEBUG({
        dbgs() << "Candidates in imag regs out of all predecessors:\n";
        for (Register R : AllPredsORV)
          dbgs() << printReg(R) << ' ';
        dbgs() << '\n';
      });

      SmallVector<Register> Candidates(SomePredORV.begin(), SomePredORV.end());
      llvm::sort(Candidates, [&](Register A, Register B) {
        // To avoid reloads, prefer to keep values allocated in all
        // predecessors, not just some.
        if (AllPredsORV.contains(A) && !AllPredsORV.contains(B))
          return true;
        if (!AllPredsORV.contains(A) && AllPredsORV.contains(B))
          return false;

        return betterToAssign(A, B, *MBB, MBB->begin());
      });

      LLVM_DEBUG({
        dbgs() << "Ordered candidates:\n";
        for (Register R : Candidates)
          dbgs() << printReg(R) << ' ';
        dbgs() << '\n';
      });

      LLVM_DEBUG(dbgs() << "Chosen candidates:\n");
      for (Register R : Candidates)
        TryAssign(R);
      LLVM_DEBUG(dbgs() << '\n');
    }

    MBBInRegVals[MBB] = RegVals;

    const auto Assign = [&](Register R, MachineBasicBlock::iterator Pos) {
      if (NeverImagVals.contains(R))
        return;
      if (TryAssign(R))
        return;

      SmallVector<Register> EvictCandidates(RegVals.begin(), RegVals.end());
      llvm::sort(EvictCandidates, [&](Register A, Register B) {
        return betterToAssign(A, B, *MBB, Pos);
      });
      LLVM_DEBUG({
        dbgs() << "Eviction candidates:\n";
        for (Register R : EvictCandidates)
          dbgs() << printReg(R) << ' ';
        dbgs() << '\n';
      });

      while (!EvictCandidates.empty()) {
        Register Evicted = EvictCandidates.back();
        EvictCandidates.pop_back();
        LLVM_DEBUG(dbgs() << "Evicting " << printReg(Evicted) << '\n');
        RegVals.erase(Evicted);
        if (TryAssign(R))
          return;
      }

      report_fatal_error("ran out of registers during register allocation");
    };

    for (MachineInstr &MI : *MBB) {
      LLVM_DEBUG(dbgs() << "Allocating " << MI);

      for (const MachineOperand &MO : MI.defs())
        if (MO.isEarlyClobber() && MO.getReg().isVirtual())
          Assign(MO.getReg(), MI);
      if (!MI.isPHI()) {
        for (const MachineOperand &MO : MI.uses()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          Register R = MO.getReg();
          if (NeverImagVals.contains(R))
            continue;
          if (!RegVals.contains(R)) {
            MachineInstr *Def = MRI->getUniqueVRegDef(R);
            if (TII->isTriviallyReMaterializable(*Def)) {
              dbgs() << "Rematerializing " << printReg(R) << '\n';
              TII->reMaterialize(*MBB, MI, R, /*SubRegIdx=*/0, *Def, *TRI);
              MachineInstr &Remat = *MI.getPrevNode();
              LLVM_DEBUG(dbgs() << Remat);
              Register NewReg = MRI->cloneVirtualRegister(R);
              rewriteVReg(R, NewReg, *MBB, MI);
              Assign(NewReg, Remat);
            } else {
              dbgs() << "Reloading " << printReg(R) << '\n';
              report_fatal_error("Not yet implemented");
            }
          }
          if (MO.isKill())
            RegVals.erase(R);
        }
      }
      MachineBasicBlock::iterator NextIter = MI;
      ++NextIter;
      for (const MachineOperand &MO : MI.defs())
        if (!MO.isEarlyClobber() && MO.getReg().isVirtual())
          Assign(MO.getReg(), NextIter);
      for (const MachineOperand &MO : MI.defs())
        if (MO.isDead())
          RegVals.erase(MO.getReg());

      LLVM_DEBUG({
        dbgs() << "Vals in regs:\n";
        for (Register R : RegVals) {
          dbgs() << printReg(R) << ' ';
        }
        dbgs() << '\n';
      });
    }
  }

  for (MachineBasicBlock &MBB : *MF) {
    const DenseSet<Register> &OutRegVals = MBBOutRegVals[&MBB];
    for (MachineBasicBlock *Succ : MBB.successors()) {
      for (Register R : MBBInRegVals[Succ]) {
        assert(OutRegVals.contains(R) &&
               "MBB value reload not yet implemented.");
      }
    }
  }
}

namespace {
// Thorup Algorithm E.
void findMaximalChains(DenseMap<unsigned, unsigned> &MaxChainsByEnd,
                       DenseMap<unsigned, unsigned> &MaxJump, uint64_t Size) {
  SmallVector<std::pair<unsigned, unsigned>> Stack = {{0, Size}};
  for (unsigned I = 0; I < Size; ++I) {
    const auto It = MaxJump.find(I);
    if (It == MaxJump.end())
      continue;
    unsigned J = It->second;

    while (Stack.back().second <= I) {
      MaxChainsByEnd[Stack.back().second] = Stack.back().first;
      Stack.pop_back();
    }
    unsigned K = I;
    while (J >= Stack.back().second && Stack.back().second > K) {
      K = Stack.back().first;
      Stack.pop_back();
    }
    Stack.push_back({K, J});
  }
}

} // namespace

template <> struct DenseMapInfo<Position> {
  static inline Position getEmptyKey() {
    return Position{
        DenseMapInfo<MachineBasicBlock *>::getEmptyKey(), {}, false};
  }
  static inline Position getTombstoneKey() {
    return Position{
        DenseMapInfo<MachineBasicBlock *>::getTombstoneKey(), {}, false};
  }
  static unsigned getHashValue(const Position &Val) {
    MachineInstr *PosMI = Val.Pos == Val.MBB->end() ? nullptr : &*Val.Pos;
    auto T = std::make_tuple(Val.MBB, PosMI, (char)Val.AfterMoves);
    return DenseMapInfo<decltype(T)>::getHashValue(T);
  }
  static bool isEqual(const Position &LHS, const Position &RHS) {
    return LHS.MBB == RHS.MBB && LHS.Pos == RHS.Pos &&
           LHS.AfterMoves == RHS.AfterMoves;
  }
};

namespace {

SmallVector<Position> positionSuccessors(Position Pos) {
  if (!Pos.AfterMoves)
    return {{Pos.MBB, Pos.Pos, true}};
  if (Pos.Pos != Pos.MBB->getFirstTerminator())
    return {{Pos.MBB, std::next(Pos.Pos), false}};
  SmallVector<Position> Successors;
  for (MachineBasicBlock *Succ : Pos.MBB->successors())
    Successors.push_back({Succ, Succ->getFirstNonPHI(), false});
  return Successors;
}

SmallVector<Position> positionPredecessors(Position Pos) {
  if (Pos.AfterMoves)
    return {{Pos.MBB, Pos.Pos, false}};
  if (Pos.Pos != Pos.MBB->getFirstNonPHI())
    return {{Pos.MBB, std::prev(Pos.Pos), true}};
  SmallVector<Position> Predecessors;
  for (MachineBasicBlock *Pred : Pos.MBB->predecessors())
    Predecessors.push_back({Pred, Pred->getFirstTerminator(), true});
  return Predecessors;
}

} // namespace

void MOSRegAlloc::scanPositions() {
  LLVM_DEBUG(dbgs() << "Collecting live vals for each position.\n");

  PositionLiveVals.clear();
  LLVM_DEBUG(PositionIndices.clear());

  SmallSet<Register, 8> LiveVals;
  unsigned Idx = 0;
  (void)Idx;
  for (MachineBasicBlock &MBB : *MF) {
    for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
      Register R = Register::index2VirtReg(I);
      if (!MRI->use_nodbg_empty(R) && LV->isLiveIn(R, MBB))
        LiveVals.insert(R);
    }

    for (MachineInstr &MI : MBB.phis()) {
      if (!MI.getOperand(0).isDead())
        LiveVals.insert(MI.getOperand(0).getReg());
    }

    const auto DumpPos = [&](Position Pos) {
      dbgs() << PositionIndices[Pos] << ":";
      if (Pos.AfterMoves)
        dbgs() << " after moves";
      else
        dbgs() << " before moves";
      dbgs() << " %bb." << Pos.MBB->getNumber() << ':';
      if (Pos.Pos == Pos.MBB->end())
        dbgs() << " end\n";
      else
        dbgs() << ' ' << *Pos.Pos;
      dbgs() << "Live vals:";
      for (Register R : LiveVals)
        dbgs() << ' ' << printReg(R);
      dbgs() << '\n';
    };
    (void)DumpPos;

    const auto RecordPos = [&](Position Pos) {
      PositionLiveVals[Pos] = LiveVals;
      LLVM_DEBUG({
        PositionIndices[Pos] = Idx++;
        DumpPos(Pos);
      });
    };

    MachineBasicBlock::iterator E = MBB.getFirstTerminator();
    RecordPos({&MBB, MBB.getFirstNonPHI(), false});
    for (MachineBasicBlock::iterator I = MBB.getFirstNonPHI(); I != E; ++I) {
      const MachineInstr &MI = *I;

      for (const MachineOperand &MO : MI.defs())
        if (MO.isEarlyClobber() && MO.getReg().isVirtual())
          LiveVals.insert(MO.getReg());

      RecordPos({&MBB, I, true});

      for (const MachineOperand &MO : MI.uses())
        if (MO.isReg() && MO.isKill())
          LiveVals.erase(MO.getReg());
      for (const MachineOperand &MO : MI.defs())
        if (!MO.isEarlyClobber() && MO.getReg().isVirtual())
          LiveVals.insert(MO.getReg());

      RecordPos({&MBB, std::next(I), false});

      for (const MachineOperand &MO : MI.defs())
        if (MO.isDead())
          LiveVals.erase(MO.getReg());
    }
    RecordPos({&MBB, E, true});
  }
}

void MOSRegAlloc::decomposeToTree() {
  LLVM_DEBUG(dbgs() << "Producing tree decomposition of basic block graph.\n");

  SmallVector<Position> Positions;
  DenseMap<Position, unsigned> PositionIndices;
  for (MachineBasicBlock &MBB : *MF) {
    MachineBasicBlock::iterator E = MBB.getFirstTerminator();
    for (MachineBasicBlock::iterator I = MBB.getFirstNonPHI(); I != E; ++I) {
      Positions.push_back({&MBB, I, false});
      PositionIndices[Positions.back()] = Positions.size() - 1;
      Positions.push_back({&MBB, I, true});
      PositionIndices[Positions.back()] = Positions.size() - 1;
    }
    Positions.push_back({&MBB, E, false});
    PositionIndices[Positions.back()] = Positions.size() - 1;
    Positions.push_back({&MBB, E, true});
    PositionIndices[Positions.back()] = Positions.size() - 1;
  }

  DenseMap<unsigned, unsigned> MaxJJump;
  DenseMap<unsigned, unsigned> MaxSJump;
  for (MachineBasicBlock &MBB : *MF) {
    Position From = {&MBB, MBB.getFirstTerminator(), true};
    unsigned I = PositionIndices[From];

    for (MachineBasicBlock *Succ : MBB.successors()) {
      Position To = {Succ, Succ->getFirstNonPHI(), false};
      unsigned J = PositionIndices[To];
      assert(I != J);
      if (I < J) {
        const auto Res = MaxJJump.try_emplace(I, J);
        if (!Res.second && J > Res.first->second)
          Res.first->second = J;
        const auto Res2 = MaxSJump.try_emplace(I, J);
        if (!Res2.second && J > Res2.first->second)
          Res2.first->second = J;
      } else {
        const auto Res = MaxSJump.try_emplace(J, I);
        if (!Res.second && I > Res.first->second)
          Res.first->second = I;
      }
    }
  }
  dbgs() << "MaxJJump\n";
  for (const auto &[I, J] : MaxJJump)
    dbgs() << formatv("({0}, {1})\n", I, J);
  dbgs() << "MaxSJump\n";
  for (const auto &[I, J] : MaxSJump)
    dbgs() << formatv("({0}, {1})\n", I, J);

  DenseMap<unsigned, unsigned> MaximalJChainsByEnd;
  findMaximalChains(MaximalJChainsByEnd, MaxJJump, Positions.size());
  dbgs() << "MaxJChains\n";
  for (const auto &[I, J] : MaximalJChainsByEnd)
    dbgs() << formatv("({0}, {1})\n", I, J);

  DenseMap<unsigned, unsigned> MaximalSChainsByEnd;
  findMaximalChains(MaximalSChainsByEnd, MaxSJump, Positions.size());
  dbgs() << "MaxSChains\n";
  for (const auto &[I, J] : MaximalSChainsByEnd)
    dbgs() << formatv("({0}, {1})\n", I, J);

  // Algorithm D given by Thorup, for finding a good listing.
  std::vector<int> Listing(Positions.size(), -1);
  unsigned I = 0;
  for (int J = Positions.size() - 1; J >= 0; --J) {
    if (Listing[J] < 0)
      Listing[J] = I++;

    auto It = MaximalSChainsByEnd.find(J);
    if (It != MaximalSChainsByEnd.end() && Listing[It->second] < 0)
      Listing[It->second] = I++;

    It = MaximalJChainsByEnd.find(J);
    if (It != MaximalJChainsByEnd.end() && Listing[It->second] < 0)
      Listing[It->second] = I++;
  }

  // Permute blocks into Thorup listing order.
  {
    SmallVector<Position> OrderedPositions(Positions.size());
    for (const auto &[I, L] : llvm::enumerate(Listing))
      OrderedPositions[L] = Positions[I];

    Positions.swap(OrderedPositions);
    for (const auto &[I, Pos] : llvm::enumerate(Positions))
      PositionIndices[Pos] = I;
  }

  // Compute the minimum separators for each block. (Thorup Algorithm A).
  SmallVector<SmallSet<unsigned, 5>> Separators(Positions.size());
  SmallVector<SmallSet<unsigned, 5>> InvSeparators(Positions.size());
  DenseSet<unsigned> DSet;
  for (int I = Positions.size() - 1; I >= 0; --I) {
    Position Pos = Positions[I];
    for (Position Succ : positionSuccessors(Pos)) {
      unsigned H = PositionIndices[Succ];
      if (H >= (unsigned)I)
        continue;
      Separators[I].insert(H);
      InvSeparators[H].insert(I);
    }
    // Note that the graph is considered undirected here.
    for (Position Pred : positionPredecessors(Pos)) {
      unsigned H = PositionIndices[Pred];
      if (H >= (unsigned)I)
        continue;
      Separators[I].insert(H);
      InvSeparators[H].insert(I);
    }
    for (unsigned W : InvSeparators[I]) {
      if (!DSet.insert(W).second)
        continue;
      for (unsigned H : Separators[W]) {
        if (H >= (unsigned)I)
          continue;
        Separators[I].insert(H);
        InvSeparators[H].insert(I);
      }
    }
  }

  dbgs() << "Separators:\n";
  for (unsigned I = 0; I < Positions.size(); ++I) {
    dbgs() << I << ": ";
    for (unsigned J : Separators[I]) {
      dbgs() << J << ' ';
    }
    dbgs() << '\n';
  }

  // Thorup, Lemma 12.
  SmallVector<SmallSet<unsigned, 5>> NodePositions(Positions.size());
  SmallVector<SmallSet<unsigned, 5>> NodeChildren(Positions.size());
  NodePositions[0].insert(0);
  for (unsigned I = 1; I < Positions.size(); ++I) {
    unsigned H = 0;
    for (unsigned S : Separators[I])
      H = std::max(H, S);
    NodeChildren[H].insert(I);
    NodePositions[I] = Separators[I];
    NodePositions[I].insert(I);
  }

  // Produce a "nice" tree decomposition, where the position set differs by at
  // most one node between parents and children, and nodes with multiple
  // children have the same position set as their children.
  std::function<void(unsigned)> MakeSubTreeNice = [&](unsigned Root) {
    if (NodeChildren[Root].size() > 1) {
      SmallSet<unsigned, 5> JoinChildren;
      while (!NodeChildren[Root].empty()) {
        unsigned Child = *NodeChildren[Root].begin();
        NodeChildren[Root].erase(Child);
        if (NodePositions[Root] != NodePositions[Child]) {
          unsigned NewChild = NodeChildren.size();
          NodePositions.emplace_back();
          NodePositions[NewChild] = NodePositions[Root];
          NodeChildren.emplace_back();
          NodeChildren[NewChild].insert(Child);
          Child = NewChild;
        }
        JoinChildren.insert(Child);
      }
      NodeChildren[Root] = std::move(JoinChildren);
      for (unsigned C : NodeChildren[Root])
        MakeSubTreeNice(C);
      return;
    }

    SmallSet<unsigned, 5> ChildPositions;
    if (NodeChildren[Root].size() == 1) {
      unsigned Child = *NodeChildren[Root].begin();
      ChildPositions = NodePositions[Child];
    }
    unsigned NumRemoved = 0;
    unsigned ARemoved;
    for (unsigned P : NodePositions[Root]) {
      if (!ChildPositions.contains(P)) {
        NumRemoved++;
        ARemoved = P;
      }
    }
    unsigned NumInserted = 0;
    unsigned AnInserted;
    for (unsigned P : ChildPositions) {
      if (!NodePositions[Root].contains(P)) {
        NumInserted++;
        AnInserted = P;
      }
    }

    if (NumRemoved > 1 || (NumRemoved && NumInserted)) {
      unsigned NewChild = NodeChildren.size();
      NodePositions.emplace_back();
      NodePositions[NewChild] = NodePositions[Root];
      NodePositions[NewChild].erase(ARemoved);
      NodeChildren.emplace_back();
      if (NodeChildren[Root].size() == 1)
        NodeChildren[NewChild].insert(*NodeChildren[Root].begin());
      NodeChildren[Root].clear();
      NodeChildren[Root].insert(NewChild);
      MakeSubTreeNice(NewChild);
      return;
    }

    if (NumInserted > 1) {
      unsigned NewChild = NodeChildren.size();
      NodePositions.emplace_back();
      NodePositions[NewChild] = NodePositions[Root];
      NodePositions[NewChild].insert(AnInserted);
      NodeChildren.emplace_back();
      if (NodeChildren[Root].size() == 1)
        NodeChildren[NewChild].insert(*NodeChildren[Root].begin());
      NodeChildren[Root].clear();
      NodeChildren[Root].insert(NewChild);
      MakeSubTreeNice(NewChild);
      return;
    }

    for (unsigned C : NodeChildren[Root])
      MakeSubTreeNice(C);
  };
  MakeSubTreeNice(0);

  Tree.clear();
  Tree.resize(NodePositions.size());
  for (unsigned I = 0, E = NodePositions.size(); I != E; ++I) {
    for (unsigned P : NodePositions[I])
      Tree[I].Positions.push_back(Positions[P]);
    for (unsigned C : NodeChildren[I])
      Tree[I].Children.push_back(&Tree[C]);
  }

  dumpTree();
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

bool MOSRegAlloc::betterToAssign(Register Left, Register Right,
                                 const MachineBasicBlock &MBB,
                                 MachineBasicBlock::const_iterator Pos) {
  bool CanRematLeft =
      TII->isTriviallyReMaterializable(*MRI->getUniqueVRegDef(Left));
  bool CanRematRight =
      TII->isTriviallyReMaterializable(*MRI->getUniqueVRegDef(Right));
  if (!CanRematLeft && CanRematRight)
    return true;
  if (CanRematLeft && !CanRematRight)
    return false;
  return nearerNextUse(Left, Right, MBB, Pos);
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
      IP->addImag8CSR();
    else
      IP->addImag8();
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
      IP->removeImag8CSR();
    else
      IP->removeImag8();
  }
}

bool MOSRegAlloc::isImagPressureOver(const ImagPressure &IP) const {
  return IP.numImag16Needed() > NumImag16Avail ||
         IP.numImag16CSRNeeded() > NumImag16CSRAvail;
}

void MOSRegAlloc::rewriteVReg(Register From, Register To,
                              MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator Pos) {
  for (auto I = Pos, E = MBB.end(); I != E; ++I) {
    if (I->readsVirtualRegister(From) || I->modifiesRegister(From)) {
      LLVM_DEBUG(dbgs() << "Rewriting " << *I);
      I->substituteRegister(From, To, /*SubIdx=*/0, *TRI);
      LLVM_DEBUG(dbgs() << "To " << *I);
    }
  }

  DenseMap<const MachineDomTreeNode *, Register> MDTNReg;
  MDTNReg[MDT->getNode(&MBB)] = To;
  const std::set<MachineBasicBlock *> &DFSet = MDF->find(&MBB)->second;

  SmallVector<MachineOperand *> Uses;
  for (MachineOperand &MO : MRI->use_nodbg_operands(From))
    Uses.push_back(&MO);

  SmallVector<Register> AffectedRegs = {From, To};

  while (!Uses.empty()) {
    MachineOperand *MO = Uses.back();
    Uses.pop_back();
    MachineInstr &MI = *MO->getParent();

    const MachineDomTreeNode *MDTN;
    for (MDTN = MDT->getNode(
             MI.isPHI() ? MI.getOperand(MO->getOperandNo() + 1).getMBB()
                        : MI.getParent());
         MDTN; MDTN = MDTN->getIDom()) {

      auto It = MDTNReg.find(MDTN);
      if (It != MDTNReg.end()) {
        LLVM_DEBUG(dbgs() << "Rewriting: " << MI);
        MO->setReg(It->second);
        LLVM_DEBUG(dbgs() << "To: " << MI);
        break;
      }

      MachineBasicBlock *CurMBB = MDTN->getBlock();

      if (!llvm::is_contained(DFSet, CurMBB))
        continue;

      MachineIRBuilder Builder(*CurMBB, CurMBB->begin());
      Register NewReg = MRI->cloneVirtualRegister(To);
      AffectedRegs.push_back(NewReg);
      auto Phi = Builder.buildInstr(MOS::PHI, {NewReg}, {});
      for (MachineBasicBlock *Pred : CurMBB->predecessors()) {
        Phi.addUse(From);
        Phi.addMBB(Pred);
      }
      for (unsigned I = 1, E = Phi->getNumOperands(); I != E; I += 2)
        Uses.push_back(&Phi->getOperand(I));
      LLVM_DEBUG(dbgs() << "Inserted PHI to MBB %bb." << CurMBB->getNumber()
                        << '\n');
      LLVM_DEBUG(dbgs() << *Phi);
      MDTNReg[MDTN] = NewReg;
      break;
    }
    assert(MDTN && "Could not find definition");
  }

  for (Register R : AffectedRegs)
    LV->recomputeForSingleDefVirtReg(R);
  for (Register R : AffectedRegs)
    recomputeNextUseDists(R);
}

void MOSRegAlloc::dumpTree(Node *Root, unsigned Indent) {
  if (!Root)
    Root = &Tree[0];
  for (unsigned I = 0; I < Indent; ++I)
    dbgs() << ' ';
  dbgs() << Root - &Tree[0] << ": ";
  for (Position P : Root->Positions) {
    dbgs() << PositionIndices[P] << ' ';
  }
  dbgs() << '\n';
  for (Node *C : Root->Children)
    dumpTree(C, Indent + 2);
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocator", false, false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
