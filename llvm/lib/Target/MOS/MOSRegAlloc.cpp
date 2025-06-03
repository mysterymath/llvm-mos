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
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include <limits>

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

template <> struct DenseMapInfo<SmallVector<Alloc>> {
  static inline SmallVector<Alloc> getEmptyKey() {
    return {DenseMapInfo<Alloc>::getEmptyKey()};
  }
  static inline SmallVector<Alloc> getTombstoneKey() {
    return {DenseMapInfo<Alloc>::getTombstoneKey()};
  }

  static unsigned getHashValue(const SmallVector<Alloc> &Val) {
    unsigned Hash = 0;
    for (Alloc A : Val)
      Hash =
          detail::combineHashValue(Hash, DenseMapInfo<Alloc>::getHashValue(A));
    return Hash;
  }

  static bool isEqual(const SmallVector<Alloc> &LHS,
                      const SmallVector<Alloc> &RHS) {
    return LHS == RHS;
  }
};

namespace {

// A chain of allocations that can be followed backwards from the current
// allocation.
struct AllocImpl {
  // Cost of using this impl.
  unsigned Cost;

  // True if the prev alloc is at this MI rather than the previous.
  bool IsCopy;

  Alloc PrevAlloc;
};

#if 0
bool operator<(const std::pair<Alloc, AllocImpl> &L,
               const std::pair<Alloc, AllocImpl> &R) {
  return L.second.Cost < R.second.Cost;
}
#endif

#if 0
// A point within the program that can have a tracked allocation.
struct AllocPoint {
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator I;

  AllocPoint(MachineBasicBlock *MBB, MachineBasicBlock::iterator I)
      : MBB(MBB), I(I) {}

  bool canInsert() const {
    if (I == MBB->begin())
      return true;
    return !std::prev(I)->isTerminator();
  }
};
#endif

class MOSRegAlloc;

#if 0
struct TreeNode {
  enum class Type { Intro, Forget, Join };

  SmallVector<unsigned> AllocPoints;
  SmallVector<unsigned> Children;

  //DenseMap<SmallVector<Alloc>, AllocImpl> AllocImpls;

  Type getType(const MOSRegAlloc &Ctx) const;
  unsigned getIntroducedIdx(MOSRegAlloc &Ctx) const;
};
#endif

struct MBBAlloc {
  // For each machine instruction (plus the end), a map from start alloc then
  // end alloc to the best implementation.
  SmallVector<DenseMap<Alloc, DenseMap<Alloc, AllocImpl>>> MIAllocs;

  // For each end alloc, the start alloc with the best implementation.
  DenseMap<Alloc, Alloc> BestStartAlloc;
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

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    MachineFunctionPass::getAnalysisUsage(AU);
    AU.addRequired<MachineDominatorTreeWrapperPass>();
    AU.addPreserved<MachineDominatorTreeWrapperPass>();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

private:
  friend struct TreeNode;

  MachineFunction *MF;
  MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineDominatorTree *MDT;

  SmallVector<Register, 0> RewrittenVReg;

  // Allocation points for each instruction. These are ordered such that defs
  // always appear before uses. Block predecessors appear before block
  // successors, except for back edges.
  // SmallVector<AllocPoint, 0> AllocPoints;

  // DenseMap<const MachineBasicBlock *, unsigned> MBBStartIdx;
  // DenseMap<const MachineBasicBlock *, unsigned> MBBEndIdx;

  mutable std::optional<LiveVariables> LV;

  IndexedMap<MBBAlloc, MBB2NumberFunctor> MBBAllocs;

  // Lists of live registers ordered by next use distance
  IndexedMap<SmallVector<Register>, MBB2NumberFunctor> MBBEndNextUsed;
  SmallVector<Register> RegCandValues;

  // SmallVector<TreeNode, 0> Tree;

  void rewriteSSAValues();
  Register rewriteSSAValue(Register R);
  LLT findRegType(Register R);
  void initMBBEndNextUsed();
  void dumpMBBEndNextUsed();

#if 0
  void initAllocPoints();
  SmallVector<unsigned> allocPointSuccessors(unsigned APIdx) const;
  SmallVector<unsigned> allocPointPredecessors(unsigned APIdx) const;

  void decomposeToTree();
  void dumpAllocPoints() const;
  void dumpTree(unsigned RootIdx = 0, unsigned Indent = 0) const;

  void allocatePhysRegs(unsigned SubTreeIdx = 0);
#endif

  void allocateMBB(const MachineBasicBlock &MBB);

  void allocateMBBStart(const MachineBasicBlock &MBB);
  void selectRegCandValues(const MachineBasicBlock &MBB,
                           MachineBasicBlock::const_iterator I);
  void allocateMI(const MachineInstr &MI);
  void allocateMO(const MachineOperand &MO);
  void freeUse(const MachineOperand &MO);
  void shuffleAllocs(const MachineBasicBlock &MBB);
  void dumpNumAllocs(const MachineBasicBlock &MBB) const;

  void applyBestAlloc();
  void eliminateTrivialCopies();
  void computeLiveIns();
};

#if 0
TreeNode::Type TreeNode::getType(const MOSRegAlloc &Ctx) const {
  if (Children.empty())
    return Type::Intro;
  if (Children.size() > 1)
    return Type::Join;
  const TreeNode &Child = Ctx.Tree[Children[0]];
  assert(Child.AllocPoints.size() != AllocPoints.size() &&
         "Node must be either introduce or forget.");
  return Child.AllocPoints.size() > AllocPoints.size() ? Type::Forget
                                                       : Type::Intro;
}

unsigned TreeNode::getIntroducedIdx(MOSRegAlloc &Ctx) const {
  SmallSet<unsigned, 6> ChildAllocPoints;
  if (!Children.empty())
    for (unsigned I : Ctx.Tree[Children.back()].AllocPoints)
      ChildAllocPoints.insert(I);
  for (unsigned I : AllocPoints)
    if (!ChildAllocPoints.contains(I))
      return I;
  llvm_unreachable("no alloc point was introduced");
}
#endif

} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();

  MF.RenumberBlocks();

  rewriteSSAValues();
  LV.emplace(MF);
  MF.dump();

#if 0
  initAllocPoints();

  decomposeToTree();
  dumpAllocPoints();
  dumpTree();

  allocatePhysRegs();
#endif

  initMBBEndNextUsed();
  dumpMBBEndNextUsed();

  MBBAllocs.resize(MF.getNumBlockIDs());
  for (MachineBasicBlock &MBB : MF)
    allocateMBB(MBB);

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
  // TODO: Use IndexedMap to simplify this?
  RewrittenVReg.clear();
  RewrittenVReg.resize(MRI->getNumVirtRegs());
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!MRI->reg_nodbg_empty(R))
      rewriteSSAValue(R);
  }
}

Register MOSRegAlloc::rewriteSSAValue(Register R) {
  unsigned Idx = R.virtRegIndex();
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

  RewrittenVReg[Idx] = New.virtRegIndex();
  MRI->replaceRegWith(R, New);
  return New;
}

LLT MOSRegAlloc::findRegType(Register R) {
  LLT Ty = MRI->getType(R);
  if (Ty.isValid())
    return Ty;
  return LLT::scalar(TRI->getRegSizeInBits(R, *MRI));
}

#if 0
void MOSRegAlloc::initAllocPoints() {
  AllocPoints.clear();
  for (MachineBasicBlock &MBB : *MF) {
    MBBStartIdx[&MBB] = AllocPoints.size();
    for (MachineBasicBlock::iterator I = MBB.getFirstNonPHI(), E = MBB.end();;
         ++I) {
      AllocPoints.emplace_back(&MBB, I);
      if (I == E)
        break;
    }
    MBBEndIdx[&MBB] = AllocPoints.size() - 1;
  }
}

SmallVector<unsigned> MOSRegAlloc::allocPointSuccessors(unsigned APIdx) const {
  const AllocPoint &AP = AllocPoints[APIdx];
  SmallVector<unsigned> Successors;
  if (AP.I == AP.MBB->end())
    for (MachineBasicBlock *Succ : AP.MBB->successors())
      Successors.push_back(MBBStartIdx.at(Succ));
  else
    Successors.push_back(APIdx + 1);
  return Successors;
}

SmallVector<unsigned>
MOSRegAlloc::allocPointPredecessors(unsigned APIdx) const {
  const AllocPoint &AP = AllocPoints[APIdx];
  SmallVector<unsigned> Predecessors;
  if (AP.I == AP.MBB->getFirstNonPHI())
    for (MachineBasicBlock *Pred : AP.MBB->predecessors())
      Predecessors.push_back(MBBEndIdx.at(Pred));
  else
    Predecessors.push_back(APIdx - 1);
  return Predecessors;
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

void MOSRegAlloc::decomposeToTree() {
  DenseMap<unsigned, unsigned> MaxJJump;
  DenseMap<unsigned, unsigned> MaxSJump;
  for (MachineBasicBlock &MBB : *MF) {
    unsigned I = MBBEndIdx[&MBB];
    for (MachineBasicBlock *Succ : MBB.successors()) {
      unsigned J = MBBStartIdx[Succ];
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
  findMaximalChains(MaximalJChainsByEnd, MaxJJump, AllocPoints.size());
  dbgs() << "MaxJChains\n";
  for (const auto &[I, J] : MaximalJChainsByEnd)
    dbgs() << formatv("({0}, {1})\n", I, J);

  DenseMap<unsigned, unsigned> MaximalSChainsByEnd;
  findMaximalChains(MaximalSChainsByEnd, MaxSJump, AllocPoints.size());
  dbgs() << "MaxSChains\n";
  for (const auto &[I, J] : MaximalSChainsByEnd)
    dbgs() << formatv("({0}, {1})\n", I, J);

  // Algorithm D given by Thorup, for finding a good listing. A listing is the
  // permuted index for each position.
  SmallVector<int> Listing(AllocPoints.size(), -1);
  unsigned I = 0;
  for (int J = AllocPoints.size() - 1; J >= 0; --J) {
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
  SmallVector<unsigned> InvListing(AllocPoints.size());
  for (const auto &[I, L] : llvm::enumerate(Listing))
    InvListing[L] = I;

  // Compute the minimum separators for each block. (Thorup Algorithm A).
  SmallVector<SmallSet<unsigned, 5>> Separators(AllocPoints.size());
  SmallVector<SmallSet<unsigned, 5>> InvSeparators(AllocPoints.size());
  DenseSet<unsigned> DSet;
  for (int I = AllocPoints.size() - 1; I >= 0; --I) {
    unsigned P = InvListing[I];
    for (unsigned Succ : allocPointSuccessors(P)) {
      unsigned H = Listing[Succ];
      if (H >= (unsigned)I)
        continue;
      Separators[I].insert(H);
      InvSeparators[H].insert(I);
    }
    // Note that the graph is considered undirected here.
    for (unsigned Pred : allocPointPredecessors(P)) {
      unsigned H = Listing[Pred];
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
  for (unsigned I = 0; I < AllocPoints.size(); ++I) {
    dbgs() << I << ": ";
    for (unsigned J : Separators[Listing[I]])
      dbgs() << InvListing[J] << ' ';
    dbgs() << '\n';
  }

  // Thorup, Lemma 12.
  SmallVector<SmallSet<unsigned, 5>> NodeAllocPoints(AllocPoints.size());
  SmallVector<SmallSet<unsigned, 5>> NodeChildren(AllocPoints.size());
  NodeAllocPoints[0].insert(InvListing[0]);
  for (unsigned I = 1; I < AllocPoints.size(); ++I) {
    unsigned H = 0;
    for (unsigned S : Separators[I])
      H = std::max(H, S);
    NodeChildren[H].insert(I);
    for (unsigned S : Separators[I])
      NodeAllocPoints[I].insert(InvListing[S]);
    NodeAllocPoints[I].insert(InvListing[I]);
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
        if (NodeAllocPoints[Root] != NodeAllocPoints[Child]) {
          unsigned NewChild = NodeChildren.size();
          NodeAllocPoints.emplace_back();
          NodeAllocPoints[NewChild] = NodeAllocPoints[Root];
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

    SmallSet<unsigned, 5> ChildAllocPoints;
    if (NodeChildren[Root].size() == 1) {
      unsigned Child = *NodeChildren[Root].begin();
      ChildAllocPoints = NodeAllocPoints[Child];
    }
    unsigned NumRemoved = 0;
    unsigned ARemoved;
    for (unsigned P : NodeAllocPoints[Root]) {
      if (!ChildAllocPoints.contains(P)) {
        NumRemoved++;
        ARemoved = P;
      }
    }
    unsigned NumInserted = 0;
    unsigned AnInserted;
    for (unsigned P : ChildAllocPoints) {
      if (!NodeAllocPoints[Root].contains(P)) {
        NumInserted++;
        AnInserted = P;
      }
    }

    if (NumRemoved > 1 || (NumRemoved && NumInserted)) {
      unsigned NewChild = NodeAllocPoints.size();
      NodeAllocPoints.emplace_back();
      NodeAllocPoints[NewChild] = NodeAllocPoints[Root];
      NodeAllocPoints[NewChild].erase(ARemoved);
      NodeChildren.emplace_back();
      if (NodeChildren[Root].size() == 1)
        NodeChildren[NewChild].insert(*NodeChildren[Root].begin());
      NodeChildren[Root].clear();
      NodeChildren[Root].insert(NewChild);
      MakeSubTreeNice(NewChild);
      return;
    }

    if (NumInserted > 1) {
      unsigned NewChild = NodeAllocPoints.size();
      NodeAllocPoints.emplace_back();
      NodeAllocPoints[NewChild] = NodeAllocPoints[Root];
      NodeAllocPoints[NewChild].insert(AnInserted);
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
  unsigned RootCopy = NodeAllocPoints.size();
  NodeAllocPoints.push_back(NodeAllocPoints[0]);
  NodeChildren.push_back(NodeChildren[0]);
  NodeAllocPoints[0].clear();
  NodeChildren[0].clear();
  NodeChildren[0].insert(RootCopy);

  Tree.clear();
  Tree.resize(NodeAllocPoints.size());
  for (unsigned I = 0, E = NodeAllocPoints.size(); I != E; ++I) {
    for (unsigned P : NodeAllocPoints[I])
      Tree[I].AllocPoints.push_back(P);
    llvm::sort(Tree[I].AllocPoints);
    for (unsigned C : NodeChildren[I])
      Tree[I].Children.push_back(C);
    llvm::sort(Tree[I].Children);
  }
}

void MOSRegAlloc::dumpAllocPoints() const {
  for (MachineBasicBlock &MBB : *MF) {
    dbgs() << printMBBReference(MBB)
           << ": "
           //<< MBFI->getBlockFreq(&MBB).getFrequency()
           << '\n';
    for (unsigned I = MBBStartIdx.at(&MBB), E = MBBEndIdx.at(&MBB);; ++I) {
      dbgs() << I << ": ";
      if (I == E) {
        dbgs() << "<end>\n";
        break;
      }
      dbgs() << *AllocPoints[I].I;
    }
    dbgs() << '\n';
  }
}

void MOSRegAlloc::dumpTree(unsigned RootIdx, unsigned Indent) const {
  for (unsigned I = 0; I < Indent; ++I)
    dbgs() << ' ';
  dbgs() << RootIdx;
  const TreeNode &Root = Tree[RootIdx];
  switch (Root.getType(*this)) {
  case TreeNode::Type::Forget:
    dbgs() << 'F';
    break;
  case TreeNode::Type::Intro:
    dbgs() << 'I';
    break;
  case TreeNode::Type::Join:
    dbgs() << 'J';
    break;
  }
  dbgs() << ": ";
  for (unsigned P : Root.AllocPoints)
    dbgs() << P << ' ';
  dbgs() << '\n';
  for (unsigned C : Root.Children)
    dumpTree(C, Indent + 1);
}

void MOSRegAlloc::allocatePhysRegs(unsigned SubTreeIdx) {
  TreeNode &SubTree = Tree[SubTreeIdx];
  for (unsigned ChildIdx : SubTree.Children)
    allocatePhysRegs(ChildIdx);
  dbgs() << "Allocating node: " << SubTreeIdx << ':';
  for (unsigned I : SubTree.AllocPoints)
    dbgs() << ' ' << I;
  dbgs() << '\n';
  switch (SubTree.getType(*this)) {
  case TreeNode::Type::Intro: {
    unsigned IntroducedIdx = SubTree.getIntroducedIdx(*this);
    SmallVector<Alloc> IntroducedAllocs =
        collectAllAllocs(AllocPoints[IntroducedIdx]);
    assert(SubTree.Children.size() <= 1);
    TreeNode *Child =
        !SubTree.Children.empty() ? &Tree[SubTree.Children.back()] : nullptr;
    DenseMap<SmallVector<Alloc>, AllocImpl> EmptyAllocImpls;
    DenseMap<SmallVector<Alloc>, AllocImpl> *ChildAllocImpls;
    if (Child) {
      ChildAllocImpls = &Child->AllocImpls;
    } else {
      ChildAllocImpls = &EmptyAllocImpls;
      EmptyAllocImpls[{}] = {/*Cost=*/0, /*IsCopy=*/false, /*PrevAllocs=*/{}};
    }
    for (const auto &[ChildAllocs, ChildImpl] : *ChildAllocImpls) {
      for (const Alloc &IntroducedAlloc : IntroducedAllocs) {
        SmallVector<Alloc> NewAllocs;
        unsigned J = 0;
        for (unsigned I : SubTree.AllocPoints)
          if (I == IntroducedIdx)
            NewAllocs.push_back(IntroducedAlloc);
          else {
            NewAllocs.push_back(ChildAllocs[J]);
            J++;
          }
        SubTree.AllocImpls[NewAllocs] = {/*Cost=*/0, /*IsCopy=*/false,
                                         ChildAllocs};
      }
    }
    break;
  }
  default:
    llvm_unreachable("TODO: Non-intro");
  }
  dbgs() << "Num allocations: " << SubTree.AllocImpls.size() << '\n';
}

#endif

void MOSRegAlloc::initMBBEndNextUsed() {
  dbgs() << "Computing next used values at MBB ends.\n";

  IndexedMap<DenseMap<Register, unsigned>, MBB2NumberFunctor>
      MBBStartNextUseDists;
  IndexedMap<DenseMap<Register, unsigned>, MBB2NumberFunctor>
      MBBEndNextUseDists;
  IndexedMap<size_t, MBB2NumberFunctor> MBBEndDist;
  MBBStartNextUseDists.resize(MF->getNumBlockIDs());
  MBBEndNextUseDists.resize(MF->getNumBlockIDs());
  MBBEndDist.resize(MF->getNumBlockIDs());

  const auto UpdateDist = [](DenseMap<Register, unsigned> &Dists, Register V,
                             unsigned D) {
    auto Res = Dists.try_emplace(V, D);
    if (Res.second)
      return true;
    if (D < Res.first->second) {
      Res.first->second = D;
      return true;
    }
    return false;
  };

  for (const MachineBasicBlock &MBB : *MF) {
    MBBEndDist[&MBB] = std::distance(MBB.getFirstNonPHI(), MBB.end());
    for (const auto &[I, MI] :
         enumerate(make_range(MBB.getFirstNonPHI(), MBB.end()))) {
      for (const MachineOperand &MO : MI.uses()) {
        if (!MO.isReg() || !MO.isUse() || MO.getReg().isPhysical())
          continue;
        UpdateDist(MBBStartNextUseDists[&MBB], MO.getReg(), I);
      }
    }
  }

  // Propagate until fixed point.
  SetVector<const MachineBasicBlock *> WorkList;
  for (const MachineBasicBlock &MBB : *MF)
    WorkList.insert(&MBB);
  while (!WorkList.empty()) {
    const MachineBasicBlock &MBB = *WorkList.pop_back_val();

    // TODO: Hard-prioritize loop backedges.
    // TODO: Soft-prioritize more likely successors.
    bool Dirty = false;
    for (const MachineBasicBlock *Succ : MBB.successors()) {
      for (const MachineInstr &MI : Succ->phis()) {
        for (unsigned I = 1, E = MI.getNumOperands(); I != E; I += 2) {
          if (MI.getOperand(I + 1).getMBB() != &MBB)
            continue;
          Dirty |= UpdateDist(MBBEndNextUseDists[&MBB],
                              MI.getOperand(I).getReg(), 0);
        }
      }

      // A PHI block counts as one instruction, since they are implemented as a
      // single shuffle.
      for (const auto &[R, D] : MBBStartNextUseDists[Succ])
        Dirty |= UpdateDist(MBBEndNextUseDists[&MBB], R, D + 1);
    }
    if (!Dirty)
      continue;

    Dirty = false;
    unsigned EndDist = MBBEndDist[&MBB];
    for (const auto [R, D] : MBBEndNextUseDists[&MBB])
      Dirty |= UpdateDist(MBBStartNextUseDists[&MBB], R, EndDist + D);

    if (Dirty)
      for (const MachineBasicBlock *Pred : MBB.predecessors())
        WorkList.insert(Pred);
  }

  MBBEndNextUsed.resize(MF->getNumBlockIDs());
  for (const MachineBasicBlock &MBB : *MF) {
    const auto &Dists = MBBEndNextUseDists[&MBB];
    SmallVector<std::pair<Register, unsigned>> NextUsed(Dists.begin(),
                                                        Dists.end());
    sort(NextUsed,
         [](const std::pair<Register, unsigned> &L,
            std::pair<Register, unsigned> &R) { return L.second < R.second; });
    sort(NextUsed,
         [](const std::pair<Register, unsigned> &L,
            std::pair<Register, unsigned> &R) { return L.second < R.second; });
    for (const auto [R, _] : NextUsed)
      MBBEndNextUsed[&MBB].push_back(R);
  }
}

void MOSRegAlloc::dumpMBBEndNextUsed() {
  for (const MachineBasicBlock &MBB : *MF) {
    dbgs() << "bb." << MBB.getNumber() << '.' << MBB.getName()
           << " end next used values:";
    for (Register V : MBBEndNextUsed[&MBB])
      dbgs() << ' ' << printReg(V, TRI);
    dbgs() << '\n';
  }
}

void MOSRegAlloc::allocateMBB(const MachineBasicBlock &MBB) {
  allocateMBBStart(MBB);
  for (MachineBasicBlock::const_iterator I = MBB.getFirstNonPHI(),
                                         E = MBB.end();
       ; ++I) {
    if (I == E)
      break;
    allocateMI(*I);
  }

  if (!MBB.empty() && !std::prev(MBB.end())->isTerminator()) {
    shuffleAllocs(MBB);
    dumpNumAllocs(MBB);
  }

  // TODO: Compute BestStartAlloc
}

void MOSRegAlloc::allocateMBBStart(const MachineBasicBlock &MBB) {
  selectRegCandValues(MBB, MBB.getFirstNonPHI());
  SmallVector<Alloc> Allocs = {{}};
  SmallVector<Alloc> NewAllocs;
  // For each register, add all possible assignments of values to that register
  // to the existing set of possible allocations.
  for (Register P : Alloc::Regs) {
    NewAllocs.clear();
    for (Alloc A : Allocs) {
      // No value for register P.
      NewAllocs.push_back(A);

      for (Register V : RegCandValues) {
        LLT Ty = MRI->getType(V);
        if (TRI->getRegSizeInBits(P, *MRI) != Ty.getSizeInBits())
          continue;
        Alloc New = A;
        New[P] = V;
        NewAllocs.push_back(New);
      }
    }
    Allocs.swap(NewAllocs);
  }

  MBBAllocs[&MBB].MIAllocs.emplace_back();
  auto &MIAllocs = MBBAllocs[&MBB].MIAllocs.back();
  for (Alloc A : Allocs)
    MIAllocs[A][A] = AllocImpl{/*Cost=*/0, /*IsCopy=*/false, /*PrevAlloc=*/{}};
}

void MOSRegAlloc::selectRegCandValues(const MachineBasicBlock &MBB,
                                      MachineBasicBlock::const_iterator I) {
  RegCandValues.clear();
  constexpr size_t MaxNumCands = 5;
  MachineBasicBlock::const_iterator E = MBB.end();
  for (; I != E; ++I) {
    for (const MachineOperand &MO : I->uses()) {
      if (!MO.isReg() || MO.isUndef() || MO.getReg().isPhysical())
        continue;
      RegCandValues.push_back(MO.getReg());
      if (RegCandValues.size() == MaxNumCands)
        return;
    }
  }
  for (Register R : MBBEndNextUsed[&MBB]) {
    RegCandValues.push_back(R);
    if (RegCandValues.size() == MaxNumCands)
      return;
  }
}

void MOSRegAlloc::allocateMI(const MachineInstr &MI) {
  const MachineBasicBlock &MBB = *MI.getParent();

  dbgs() << "Allocating MI: " << MI;

  dumpNumAllocs(MBB);

  if (!MI.isTerminator()) {
    shuffleAllocs(MBB);
    dumpNumAllocs(MBB);
  }

  for (const MachineOperand &MO : MI.uses())
    if (MO.isReg())
      allocateMO(MO);
  for (const MachineOperand &MO : MI.uses())
    if (MO.isReg() && MO.isUse() && MO.isKill())
      freeUse(MO);
  for (const MachineOperand &MO : MI.defs())
    allocateMO(MO);
}

void MOSRegAlloc::allocateMO(const MachineOperand &MO) {
  Register V = MO.getReg();

  const TargetRegisterClass *RC =
      TII->getRegClass(MO.getParent()->getDesc(), MO.getOperandNo(), TRI, *MF);

  // Some operands effectively have a single register class. In such cases, RC
  // is null.
  Register RegRC;
  if (Val.isVirtual() && MO.getParent()->getOpcode() == MOS::COPY) {
    RegRC = MO.isDef() ? MO.getParent()->getOperand(1).getReg()
                       : MO.getParent()->getOperand(0).getReg();
    assert(RegRC.isPhysical() && "vreg-vreg COPY not allowed");
  }

  // If none of the possible physical registers are tracked, then this operand
  // has no impact on the effective allocation.
  if (RC) {
    if (none_of(*RC, [](Register R) { return Alloc::isTracked(R); }))
      return;
  } else if (!Alloc::isTracked(RegRC)) {
    return;
  }

  auto &Allocs = MBBAllocs[MO.getParent()->getParent()].MIAllocs.back();
  DenseMap<Alloc, DenseMap<Alloc, AllocImpl>> NewAllocs;

  for (auto &[StartA, EndAllocs] : Allocs) {
    DenseMap<Alloc, AllocImpl> &NewEndAllocs = NewAllocs[StartA];
    for (auto &[EndA, AI] : EndAllocs) {
      if (MO.isUse()) {
        if (RC) {
          const auto &EndARef = EndA;
          if (none_of(Alloc::Regs, [&](Register R) {
                return EndARef[R] == V && RC->contains(R);
              }))
            continue;
        } else {
          assert(RegRC && "expected either RC or single register constraint");
          if (EndA[RegRC] != V)
            continue;
        }
        NewEndAllocs[EndA] = AI;
      } else {
        assert(MO.isDef());
      }

#if 0
        if (MO.isDef()) {
          // Live defs require that the register hold the correct value.
          // Dead defs require only that the register is free.
          if (LV->isLiveOut(Val, *MO.getParent()->getParent())
                  ? NewEndA[R] != Val
                  : !!NewEndA[R])
            continue;

          // An operand defines at most one register.
          bool CanDef = true;
          for (Register Other : Alloc::Regs) {
            if (Other != R && NewEndA[Other] == Val) {
              CanDef = false;
              break;
            }
          }
          if (!CanDef)
            continue;

          NewEndA[R] = Val;
          NewAllocs[StartA][NewEndA] = AI;
        } else {
          assert(MO.isUse());
          if (!NewEndA[R]) {
            NewEndA[R] = Val;
            NewAllocs[StartA][NewEndA] = AI;
          }
        }
#endif
    }
  }
  Allocs = std::move(NewAllocs);
}

void MOSRegAlloc::freeUse(const MachineOperand &MO) {
  Register Val = MO.getReg();
  auto &Allocs = MBBAllocs[MO.getParent()->getParent()].MIAllocs.back();
  DenseMap<Alloc, DenseMap<Alloc, AllocImpl>> NewAllocs;

  for (auto &[StartA, EndAllocs] : Allocs) {
    for (auto &[EndA, AI] : EndAllocs) {
      Alloc NewEndA = EndA;
      for (Register R : Alloc::Regs)
        if (NewEndA[R] == Val)
          NewEndA[R] = {};
      NewAllocs[StartA][NewEndA] = AI;
    }
  }
  Allocs = std::move(NewAllocs);
}

// Find all transitively reachable allocs by spilling, restoring, and copying
// registers.
void MOSRegAlloc::shuffleAllocs(const MachineBasicBlock &MBB) {
  dbgs() << "Shuffling allocs\n";
  for (auto &[StartA, AllocImpls] : MBBAllocs[&MBB].MIAllocs.back()) {
    SmallVector<Alloc> StartAllocs;
    for (const auto &[A, _] : AllocImpls)
      StartAllocs.push_back(A);

    // Worklist sorted by cost
    std::map<unsigned, SmallVector<std::pair<Alloc, AllocImpl>>> WorkList;

    // Run Dijkstra's to find the shortest-cost path to each alloc reachable
    // from a StartAlloc by some shuffle. AllocImpls contains the closed allocs.
    // TODO: Realistic costs
    // TODO: Realistic constraints
    while (!StartAllocs.empty() || !WorkList.empty()) {
      // Find a lowest-cost alloc in the worklist. Due to the Dijkstra's
      // invariant, the current path to this alloc will be a shortest path.
      // Handle each starting alloc first, since the paths to each such alloc
      // are necessarily the shortest.
      Alloc A;
      AllocImpl AI;
      if (!StartAllocs.empty()) {
        A = StartAllocs.pop_back_val();
        AI = AllocImpls[A];
        // The start allocs are present in AllocImpls, but they are not closed
        // until StartAllocs empties.
      } else {
        // Find a worklist entry of lowest cost.
        while (!WorkList.empty() && WorkList.begin()->second.empty())
          WorkList.erase(WorkList.begin());
        if (WorkList.empty())
          break;
        std::tie(A, AI) = WorkList.begin()->second.pop_back_val();

        if (AllocImpls.contains(A))
          continue;
        // AI is a shortest path to A.
        AllocImpls[A] = AI;
      }
      for (Register R : Alloc::Regs) {
        if (!A[R]) {
          const auto CanCopy = [&](Alloc &A, Register R, Register V) {
            for (Register Other : Alloc::Regs)
              if (Other != R && A[Other] == V)
                return true;
            return false;
          };

          for (Register V : RegCandValues) {
            if (MRI->getType(V).getScalarSizeInBits() !=
                TRI->getRegSizeInBits(R, *MRI))
              continue;

            // Copy or reload V to R.
            Alloc NewA = A;
            NewA[R] = V;
            if (AllocImpls.contains(NewA))
              continue;
            AllocImpl NewAI{AI.Cost + (CanCopy(A, R, V) ? 2 : 3),
                            /*IsShuffle=*/true, A};
            WorkList[NewAI.Cost].emplace_back(NewA, NewAI);
          }
        } else {
          Register V = A[R];

          // Don't copy/reload pinned physregs.
          if (V.isPhysical())
            continue;

          // Forget V or spill it to an imaginary register.
          Alloc NewA = A;
          NewA[R] = {};
          if (AllocImpls.contains(NewA))
            continue;
          AllocImpl NewAI{AI.Cost + (NewA.getReg(V) ? 0 : 3),
                          /*IsShuffle=*/true, A};
          WorkList[NewAI.Cost].emplace_back(NewA, NewAI);
        }
      }
    }
  }
}

void MOSRegAlloc::dumpNumAllocs(const MachineBasicBlock &MBB) const {
  unsigned NumAllocs = 0;
  for (const auto &[_, KV] : MBBAllocs[&MBB].MIAllocs.back())
    for (const auto &_ : KV)
      NumAllocs++;
  dbgs() << "Num allocs: " << NumAllocs << '\n';
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
