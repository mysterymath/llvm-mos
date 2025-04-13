//===-- MOSRegAlloc.cpp - MOS Register Allocation -------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the MOS register allocation pass.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"

#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"

#define DEBUG_TYPE "mos-reg-alloc"

using namespace llvm;

namespace {

// An allocation of values for each tracked architectural register.
struct Alloc {
  static constexpr std::array<Register, 5> Regs = {MOS::A, MOS::X, MOS::Y,
                                                   MOS::C, MOS::V};

  Register A, X, Y, C, V;

  static bool isTracked(Register R) {
    assert(R.isPhysical());
    switch (R) {
    case MOS::A:
      return true;
    case MOS::X:
      return true;
    case MOS::Y:
      return true;
    case MOS::C:
      return true;
    case MOS::V:
      return true;
    default:
      return false;
    }
  }

  bool operator==(const Alloc &Other) const {
    return A == Other.A && X == Other.X && Y == Other.Y && C == Other.C &&
           V == Other.V;
  };

  Register &operator[](Register R) {
    assert(isTracked(R));
    switch (R) {
    case MOS::A:
      return A;
    case MOS::X:
      return X;
    case MOS::Y:
      return Y;
    case MOS::C:
      return C;
    case MOS::V:
      return V;
    default:
      llvm_unreachable("Unexpected register");
    }
  }

  Register &operator[](Register R) const {
    return (*const_cast<Alloc *>(this))[R];
  }

  Register getReg(Register Val) const {
    if (A == Val)
      return MOS::A;
    if (X == Val)
      return MOS::X;
    if (Y == Val)
      return MOS::Y;
    if (C == Val)
      return MOS::C;
    if (C == Val)
      return MOS::V;
    return {};
  };

  void print(raw_ostream &OS, const TargetRegisterInfo *TRI) const {
    if (A)
      OS << "A: " << printReg(A, TRI) << '\n';
    if (X)
      OS << "X: " << printReg(X, TRI) << '\n';
    if (Y)
      OS << "Y: " << printReg(Y, TRI) << '\n';
    if (C)
      OS << "C: " << printReg(C, TRI) << '\n';
    if (V)
      OS << "V: " << printReg(V, TRI) << '\n';
  }
};

} // namespace

template <> struct DenseMapInfo<Alloc> {
  static inline Alloc getEmptyKey() {
    return {MOS::NUM_TARGET_REGS, 0, 0, 0, 0};
  }
  static inline Alloc getTombstoneKey() {
    return {MOS::NUM_TARGET_REGS + 1, 0, 0, 0, 0};
  }

  static unsigned getHashValue(const Alloc &Val) {
    auto Tuple = std::make_tuple(Val.A, Val.X, Val.Y, Val.C, Val.V);
    return DenseMapInfo<decltype(Tuple)>::getHashValue(Tuple);
  }

  static bool isEqual(const Alloc &LHS, const Alloc &RHS) { return LHS == RHS; }
};

namespace {

// A chain of allocations that can be followed from the current allocation.
struct AllocImpl {
  // Cost of using this impl.
  unsigned Cost;

  // True if the next alloc is in this alloc point rather than the next.
  bool IsShuffle;

  Alloc Next;
};

bool operator<(const std::pair<Alloc, AllocImpl> &L,
               const std::pair<Alloc, AllocImpl> &R) {
  return L.second.Cost < R.second.Cost;
}

// A point within the program that can have a tracked allocation.
struct AllocPoint {
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator I;

  DenseSet<Register> LiveValues;

  AllocPoint(MachineBasicBlock *MBB, MachineBasicBlock::iterator I)
      : MBB(MBB), I(I) {}

#if 0
  void dump(const TargetRegisterInfo *TRI) const {
    dbgs() << "Number of allocations: " << AllocImpls.size() << '\n';
    if (!AllocImpls.empty()) {
      const auto Min = min_element(AllocImpls);
      dbgs() << "Min Cost: " << Min->second.Cost << '\n';
      dbgs() << "Allocation:\n";
      Min->first.print(dbgs(), TRI);
    }
  }

  void dumpFull(const TargetRegisterInfo *TRI) const {
    dbgs() << "Allocations:\n";
    for (const auto &[A, AI] : AllocImpls) {
      dbgs() << "Alloc:\n";
      A.print(dbgs(), TRI);
      dbgs() << "Cost: " << AI.Cost << ", IsShuffle: " << AI.IsShuffle << '\n';
      dbgs() << "Next:\n";
      AI.Next.print(dbgs(), TRI);
    }
  }
#endif

  bool canInsert() const {
    if (I == MBB->begin())
      return true;
    return !std::prev(I)->isTerminator();
  }
};

struct TreeNode {
  enum class Type { Intro, Forget, Join } Type;
  struct APEntry {
    unsigned APIdx;
    DenseMap<Alloc, AllocImpl> AllocImpls;
  };
  SmallVector<APEntry> APEntries;
  SmallVector<unsigned> Children;
};

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::IsSSA);
  }

  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoPHIs);
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  MachineFunction *MF;
  MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;

  SmallVector<Register, 0> RewrittenVReg;

  // Allocation points for each instruction. These are ordered such that defs
  // always appear before uses. Block predecessors appear before block
  // successors, except for back edges.
  SmallVector<AllocPoint, 0> AllocPoints;

  DenseMap<const MachineBasicBlock *, unsigned> MBBStartIdx;
  DenseSet<Register> LiveValues;

  void rewriteSSAValues();
  Register rewriteSSAValue(Register R);
  LLT findRegType(Register R);

  void initAllocPoints();

  void decomposeToTree();
  void dumpPositions();
  void dumpTree(TreeNode *Root = nullptr, unsigned Indent = 0);

  void allocatePhysRegs();
  void allocateMBBEnd(unsigned APIdx);
  void allocateMI(unsigned APIdx);
  void allocateMO(AllocPoint &AP, const MachineOperand &MO);
  void freeDef(AllocPoint &AP, const MachineOperand &MO);
  void shuffleAllocs(AllocPoint &AP);
  void dumpLiveValues() const;

  void applyBestAlloc();
  void eliminateTrivialCopies();
  void computeLiveIns();
};

} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MF.dump();
  rewriteSSAValues();
  MF.dump();
  initAllocPoints();
  allocatePhysRegs();
  applyBestAlloc();
  MF.dump();
  // TODO: Return to SSA form for duplicate imagreg defs.
  eliminateTrivialCopies();
  computeLiveIns();
  MF.dump();
  return false;
}

// Strip out register classes and copies from virtual regs to establish the
// invariant that each SSA value has exactly one SSA variable.
void MOSRegAlloc::rewriteSSAValues() {
  dbgs() << "Rewriting SSA Values.\n";
  RewrittenVReg.clear();
  RewrittenVReg.resize(MRI->getNumVirtRegs());
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R))
      rewriteSSAValue(R);
  }
}

Register MOSRegAlloc::rewriteSSAValue(Register R) {
  unsigned Idx = Register::virtReg2Index(R);
  if (RewrittenVReg[Idx])
    return RewrittenVReg[Idx];

  MachineInstr *Def = MRI->getUniqueVRegDef(R);
  Register New;
  if (Def->isCopy()) {
    Register Src = Def->getOperand(1).getReg();
    if (Src.isVirtual()) {
      New = rewriteSSAValue(Src);
      Def->eraseFromParent();
    }
  }
  if (!New) {
    New = MRI->createGenericVirtualRegister(findRegType(R));
    RewrittenVReg.emplace_back();
  }

  RewrittenVReg[Idx] = Register::virtReg2Index(New);
  MRI->replaceRegWith(R, New);
  return New;
}

LLT MOSRegAlloc::findRegType(Register R) {
  LLT Ty = MRI->getType(R);
  if (Ty.isValid())
    return Ty;
  return LLT::scalar(TRI->getRegSizeInBits(R, *MRI));
}

// Create an allocation point for each machine instruction in the order of
// allocation.
void MOSRegAlloc::initAllocPoints() {
  AllocPoints.clear();
  for (MachineBasicBlock &MBB : *MF) {
    MBBStartIdx[&MBB] = AllocPoints.size();
    for (MachineInstr &MI : MBB)
      if (!MI.isPHI())
        AllocPoints.emplace_back(&MBB, MI);
    AllocPoints.emplace_back(MBB, MBB.end());
  }
}

void MOSRegAlloc::decomposeToTree() {
  SmallVector<Position> Positions;
  PositionIndices.clear();
  for (MachineBasicBlock &MBB : *MF) {
    for (MachineBasicBlock::iterator I = MBB.getFirstNonPHI(), E = MBB.end();;
         ++I) {
      Positions.push_back({&MBB, I});
      PositionIndices[Positions.back()] = Positions.size() - 1;
      if (I == E)
        break;
    }
  }

  DenseMap<unsigned, unsigned> MaxJJump;
  DenseMap<unsigned, unsigned> MaxSJump;
  for (MachineBasicBlock &MBB : *MF) {
    Position From = {&MBB, MBB.end()};
    unsigned I = PositionIndices[From];

    for (MachineBasicBlock *Succ : MBB.successors()) {
      Position To = {Succ, Succ->getFirstNonPHI()};
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

  // Algorithm D given by Thorup, for finding a good listing. A listing is the
  // permuted index for each position.
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

  // Compute the inverse listing (the original index for each permuted index).
  SmallVector<unsigned> InvListing(Positions.size());
  for (const auto &[I, L] : llvm::enumerate(Listing))
    InvListing[L] = I;

  // Compute the minimum separators for each block. (Thorup Algorithm A).
  SmallVector<SmallSet<unsigned, 5>> Separators(Positions.size());
  SmallVector<SmallSet<unsigned, 5>> InvSeparators(Positions.size());
  DenseSet<unsigned> DSet;
  for (int I = Positions.size() - 1; I >= 0; --I) {
    Position Pos = Positions[InvListing[I]];
    for (Position Succ : positionSuccessors(Pos)) {
      unsigned H = Listing[PositionIndices[Succ]];
      if (H >= (unsigned)I)
        continue;
      Separators[I].insert(H);
      InvSeparators[H].insert(I);
    }
    // Note that the graph is considered undirected here.
    for (Position Pred : positionPredecessors(Pos)) {
      unsigned H = Listing[PositionIndices[Pred]];
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
    for (unsigned J : Separators[Listing[I]])
      dbgs() << InvListing[J] << ' ';
    dbgs() << '\n';
  }

  // Thorup, Lemma 12.
  SmallVector<SmallSet<unsigned, 5>> NodePositions(Positions.size());
  SmallVector<SmallSet<unsigned, 5>> NodeChildren(Positions.size());
  NodePositions[0].insert(InvListing[0]);
  for (unsigned I = 1; I < Positions.size(); ++I) {
    unsigned H = 0;
    for (unsigned S : Separators[I])
      H = std::max(H, S);
    NodeChildren[H].insert(I);
    for (unsigned S : Separators[I])
      NodePositions[I].insert(InvListing[S]);
    NodePositions[I].insert(InvListing[I]);
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
      unsigned NewChild = NodePositions.size();
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
      unsigned NewChild = NodePositions.size();
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
  // Make the root node have no positions
  unsigned RootCopy = NodePositions.size();
  NodePositions.push_back(NodePositions[0]);
  NodeChildren.push_back(NodeChildren[0]);
  NodePositions[0].clear();
  NodeChildren[0].clear();
  NodeChildren[0].insert(RootCopy);

  Tree.clear();
  Tree.resize(NodePositions.size());
  for (unsigned I = 0, E = NodePositions.size(); I != E; ++I) {
    for (unsigned P : NodePositions[I])
      Tree[I].Positions.push_back(Positions[P]);
    for (unsigned C : NodeChildren[I])
      Tree[I].Children.push_back(&Tree[C]);
  }
}

void MOSRegAlloc::allocatePhysRegs() {
#if 0
  for (intptr_t I = AllocPoints.size() - 1; I >= 0; --I) {
    AllocPoint &AP = AllocPoints[I];
    MachineBasicBlock &MBB = *AP.MBB;
    dumpLiveValues();
    if (AP.I == MBB.end())
      allocateMBBEnd(I);
    else
      allocateMI(I);
    if (AP.AllocImpls.empty())
      report_fatal_error("physical register allocation failed");
    AP.dump(TRI);
    if (AP.canInsert()) {
      dbgs() << "Shuffling allocations.\n";
      shuffleAllocs(AP);
      AP.dump(TRI);
    }
  }
#endif
}

void MOSRegAlloc::allocateMBBEnd(unsigned APIdx) {
#if 0
  AllocPoint &AP = AllocPoints[APIdx];
  MachineBasicBlock &MBB = *AP.MBB;
  assert(AP.I == MBB.end());
  dbgs() << "\nAllocating MBB:\n" << MBB << '\n';

  if (APIdx != AllocPoints.size() - 1)
    AllocPoints[APIdx + 1].LiveValues = std::move(LiveValues);
  LiveValues.clear();
  if (MBB.succ_empty()) {
    AP.AllocImpls.try_emplace(Alloc{}, AllocImpl{0, false, {}});
  } else {
    for (const MachineBasicBlock *Succ : MBB.successors()) {
      const AllocPoint &SuccStartAP = AllocPoints[MBBStartIdx[Succ]];

      assert((Succ->empty() || !Succ->begin()->isPHI()) && "TODO: PHIs");
      for (Register R : SuccStartAP.LiveValues)
        LiveValues.insert(R);

      assert((AP.AllocImpls.empty() || SuccStartAP.AllocImpls.empty()) &&
             "TODO");
      for (const auto &[A, AI] : SuccStartAP.AllocImpls)
        AP.AllocImpls.try_emplace(A, AllocImpl{AI.Cost, false, A});
    }
  }
#endif
}

void MOSRegAlloc::allocateMI(unsigned APIdx) {
#if 0
  AllocPoint &AP = AllocPoints[APIdx];
  assert(AP.I != AP.MBB->end());
  dbgs() << "Allocating MI: " << *AP.I;
  AllocPoint &NextAP = AllocPoints[APIdx + 1];

  AP.AllocImpls = NextAP.AllocImpls;
  for (auto &[A, AI] : AP.AllocImpls) {
    AI.IsShuffle = false;

    // We do not track allocations within an instruction (although at various
    // 'points' within an instruciton they may differ). Accordingly, the next
    // allocation is always that of the next allocation point.
    AI.Next = A;
  }

  for (const MachineOperand &MO : AP.I->operands())
    if (MO.isReg() && MO.isDef())
      allocateMO(AP, MO);
  for (const MachineOperand &MO : AP.I->operands())
    if (MO.isReg() && MO.isDef())
      freeDef(AP, MO);
  for (const MachineOperand &MO : AP.I->operands())
    if (MO.isReg() && MO.isUse())
      allocateMO(AP, MO);
#endif
}

void MOSRegAlloc::allocateMO(AllocPoint &AP, const MachineOperand &MO) {
#if 0
  const MachineInstr &MI = *AP.I;
  Register Val = MO.getReg();

  // Determine the possible set of physical registers that Val could be / could
  // have been stored in.
  SmallVector<Register, 5> PhysRegs;
  if (Val.isPhysical()) {
    // Physical registers are considered to be some unique unknown value,
    // constrained to that physical register.
    Register R = Val;
    PhysRegs.push_back(R);
  } else {
    assert(Val.isVirtual());
    if (MI.getOpcode() == MOS::COPY) {
      Register R =
          MO.isDef() ? MI.getOperand(1).getReg() : MI.getOperand(0).getReg();
      assert(R.isPhysical() && "vreg-vreg COPY not allowed");
      PhysRegs.push_back(R);
    } else {
      const TargetRegisterClass *RC =
          TII->getRegClass(MI.getDesc(), MO.getOperandNo(), TRI, *MF);
      assert(RC && "TODO");
      for (Register R : *RC)
        PhysRegs.push_back(R);
    }
  }
  // If none of the possible physical registers are tracked, then this operand
  // has no impact on the effective allocation.
  llvm::erase_if(PhysRegs, [](Register R) { return !Alloc::isTracked(R); });
  if (PhysRegs.empty())
    return;

  DenseMap<Alloc, AllocImpl> NewAIs;
  for (const auto &[A, AI] : AP.AllocImpls) {
    for (Register R : PhysRegs) {
      Alloc NewA = A;
      if (MO.isDef()) {
        // Live defs require that the register hold the correct value.
        // Dead defs require only that the register is free.
        if (LiveValues.contains(Val) ? NewA[R] != Val : !!NewA[R])
          continue;

        // An operand defines at most one register.
        bool CanDef = true;
        for (Register Other : Alloc::Regs) {
          if (Other != R && NewA[Other] == Val) {
            CanDef = false;
            break;
          }
        }
        if (!CanDef)
          continue;

        if (CanDef)
          NewAIs.try_emplace(NewA, AI);
      } else {
        assert(MO.isUse());
        if (!NewA[R]) {
          NewA[R] = Val;
          NewAIs.try_emplace(NewA, AI);
        }
      }
    }
  }
  AP.AllocImpls = std::move(NewAIs);

  if (MO.isUse())
    LiveValues.insert(Val);
#endif
}

static bool recordAI(DenseMap<Alloc, AllocImpl> &AIs, const Alloc &A,
                     const AllocImpl &AI) {
  auto Result = AIs.try_emplace(A, AI);
  if (Result.second)
    return true;
  if (AI.Cost < Result.first->second.Cost) {
    Result.first->second = AI;
    return true;
  }
  return false;
}

// After all defs have been allocated, they must be freed before the uses are
// handled.
void MOSRegAlloc::freeDef(AllocPoint &AP, const MachineOperand &MO) {
#if 0
  DenseMap<Alloc, AllocImpl> NewAIs;
  Register Val = MO.getReg();
  for (const auto &[A, AI] : AP.AllocImpls) {
    Alloc NewA = A;
    for (Register R : Alloc::Regs)
      if (NewA[R] == Val)
        NewA[R] = {};
    recordAI(NewAIs, NewA, AI);
  }
  AP.AllocImpls = std::move(NewAIs);
  LiveValues.erase(Val);
#endif
}

// Find all transitively reachable allocs by spilling, restoring, and copying
// registers.
void MOSRegAlloc::shuffleAllocs(AllocPoint &AP) {
#if 0
  assert(AP.canInsert());
  SetVector<Alloc> WorkList;
  for (const auto &[A, _] : AP.AllocImpls)
    WorkList.insert(A);

  // Record an implementation for an allocation if new or superior to the
  // current implementation.
  const auto Expand = [&](Alloc A, AllocImpl AI) {
    if (recordAI(AP.AllocImpls, A, AI))
      WorkList.insert(A);
  };

  // TODO: Realistic costs
  // TODO: Realistic constraints
  while (!WorkList.empty()) {
    Alloc A = WorkList.pop_back_val();
    AllocImpl AI = AP.AllocImpls[A];
    for (Register R : Alloc::Regs) {
      if (!A[R]) {
        for (Register V : LiveValues) {
          if (MRI->getType(V).getScalarSizeInBits() !=
              TRI->getRegSizeInBits(R, *MRI))
            continue;

          // R may have previously held V, but V was spilled to an imaginary
          // register or simply forgotten.
          Alloc NewA = A;
          NewA[R] = V;
          Expand(NewA,
                 {AI.Cost + (A.getReg(V) ? 0 : 3), /*IsShuffle=*/true, A});
        }
        continue;
      }

      Register Val = A[R];

      // Don't copy/reload pinned physregs.
      if (Val.isPhysical())
        continue;

      const auto CanCopy = [&](Alloc &A, Register R, Register V) {
        for (Register Other : Alloc::Regs)
          if (Other != R && A[Other] == V)
            return true;
        return false;
      };

      // TODO: Allow reloading values that are already present. This
      // will require keeping track of the 2^5=32 cardinality spill statuses
      // for values presently in registers.
      //
      // R may now hold Val because it was copied from another register or
      // reloaded from an imaginary register.
      Alloc NewA = A;
      NewA[R] = {};
      Expand(NewA,
             {AI.Cost + (CanCopy(A, R, Val) ? 2 : 3), /*IsShuffle=*/true, A});
    }
  }
#endif
}

void MOSRegAlloc::dumpLiveValues() const {
  dbgs() << "Live values: ";
  for (Register R : LiveValues)
    dbgs() << printReg(R, TRI) << ' ';
  dbgs() << '\n';
}

// Apply the best found allocation implementation.
void MOSRegAlloc::applyBestAlloc() {
#if 0
  const auto Best = min_element(AllocPoints.front().AllocImpls);
  std::pair<Alloc, AllocImpl> Cur = *Best;
  const Alloc &A = Cur.first;
  const AllocImpl &AI = Cur.second;
  unsigned I = 0, E = AllocPoints.size();
  dbgs() << "Applying best alloc.\n";
  while (true) {
    const AllocPoint &AP = AllocPoints[I];
    MachineIRBuilder Builder(*AP.MBB, AP.I);
    if (AI.IsShuffle) {
      for (Register R : Alloc::Regs) {
        // Spill
        if (!AI.Next.getReg(A[R]) && A[R]) {
          MRI->setRegClass(A[R], &MOS::Imag8RegClass);
          Builder.buildCopy(A[R], R);
        }

        if (AI.Next[R] && !A[R]) {
          // Copy
          bool FoundCopy = false;
          for (Register Other : Alloc::Regs) {
            if (Other == R)
              continue;
            if (A[Other] == AI.Next[R]) {
              Builder.buildCopy(R, Other);
              FoundCopy = true;
              break;
            }
          }

          // Reload
          if (!FoundCopy)
            Builder.buildCopy(R, AI.Next[R]);
        }
      }
    } else {
      if (AP.I != AP.MBB->end()) {
        MachineInstr &MI = *AP.I;
        for (MachineOperand &MO : MI.operands()) {
          if (!MO.isReg() || !MO.getReg().isVirtual())
            continue;
          const Alloc *EffectiveAlloc = MO.isDef() ? &AI.Next : &A;
          Register R = EffectiveAlloc->getReg(MO.getReg());
          if (R)
            MO.setReg(R);
        }
      }
      if (++I == E)
        break;
    }
    Cur = {AI.Next, AllocPoints[I].AllocImpls.at(AI.Next)};
  }
#endif
}

void MOSRegAlloc::eliminateTrivialCopies() {
  for (MachineBasicBlock &MBB : *MF)
    for (MachineInstr &MI : make_early_inc_range(MBB))
      if (MI.isCopy() && MI.getOperand(0).getReg() == MI.getOperand(1).getReg())
        MI.eraseFromParent();
}

void MOSRegAlloc::computeLiveIns() {
  SmallVector<MachineBasicBlock *, 0> MBBs;
  for (MachineBasicBlock &MBB : *MF)
    MBBs.push_back(&MBB);
  fullyRecomputeLiveIns(MBBs);
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
