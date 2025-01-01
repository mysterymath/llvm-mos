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

#include "MOS.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#define DEBUG_TYPE "mos-reg-alloc"

using namespace llvm;

namespace {

struct Position {
  MachineBasicBlock *MBB;
  MachineBasicBlock::iterator Pos;

  bool operator==(Position Other) const {
    return MBB == Other.MBB && Pos == Other.Pos;
  }
  bool operator!=(Position Other) const { return !(*this == Other); }
};

} // namespace

template <> struct DenseMapInfo<Position> {
  static Position getEmptyKey() {
    return Position{DenseMapInfo<MachineBasicBlock *>::getEmptyKey(), {}};
  }
  static Position getTombstoneKey() {
    return Position{DenseMapInfo<MachineBasicBlock *>::getTombstoneKey(), {}};
  }
  static unsigned getHashValue(const Position &Val) {
    MachineInstr *PosMI = Val.Pos == Val.MBB->end() ? nullptr : &*Val.Pos;
    auto T = std::make_pair(Val.MBB, PosMI);
    return DenseMapInfo<decltype(T)>::getHashValue(T);
  }
  static bool isEqual(const Position &LHS, const Position &RHS) {
    return LHS == RHS;
  }
};

namespace {

// Generalized register class
struct GenRC {};

typedef std::optional<DenseMap<Register, GenRC>> PosAlloc;

struct Alloc {
  SmallVector<PosAlloc> PosAllocs;
  unsigned Cost;
};

struct Node {
  enum class Type { Intro, Forget, Join };

  SmallVector<Position> Positions;
  SmallVector<Node *> Children;

  SmallVector<Alloc> Allocs;

  Type getType() const {
    if (Children.empty())
      return Type::Intro;
    if (Children.size() > 1)
      return Type::Join;
    Node *Child = Children[0];
    assert(Child->Positions.size() != Positions.size() &&
           "Node must be either introduce or forget.");
    return Child->Positions.size() > Positions.size() ? Type::Forget
                                                      : Type::Intro;
  }
};

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;

  MOSRegAlloc() : MachineFunctionPass(ID) {
    llvm::initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
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
    AU.addRequired<LiveVariablesWrapperPass>();
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  void decomposeToTree();
  void dumpPositions();
  void dumpTree(Node *Root = nullptr, unsigned Indent = 0);
  void solveTree(Node *Root = nullptr);

  SmallVector<Register> getLiveRegs(Position P);

private:
  MachineFunction *MF;
  LiveVariables *LV;

  DenseMap<Position, unsigned> PositionIndices;
  SmallVector<Node, 0> Tree;
};

} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  LV = &getAnalysis<LiveVariablesWrapperPass>().getLV();

  LLVM_DEBUG(dbgs() << "Producing tree decomposition of basic block graph.\n");
  decomposeToTree();
  dumpPositions();
  dumpTree();
  solveTree();

  return false;
}

namespace {

SmallVector<Position> positionSuccessors(Position Pos) {
  if (Pos.Pos != Pos.MBB->end())
    return {{Pos.MBB, std::next(Pos.Pos)}};
  SmallVector<Position> Successors;
  for (MachineBasicBlock *Succ : Pos.MBB->successors())
    Successors.push_back({Succ, Succ->getFirstNonPHI()});
  return Successors;
}

SmallVector<Position> positionPredecessors(Position Pos) {
  if (Pos.Pos != Pos.MBB->getFirstNonPHI())
    return {{Pos.MBB, std::prev(Pos.Pos)}};
  SmallVector<Position> Predecessors;
  for (MachineBasicBlock *Pred : Pos.MBB->predecessors())
    Predecessors.push_back({Pred, Pred->end()});
  return Predecessors;
}

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

void MOSRegAlloc::dumpPositions() {
  for (MachineBasicBlock &MBB : *MF) {
    dbgs() << printMBBReference(MBB) << ":\n";
    for (MachineBasicBlock::iterator I = MBB.getFirstNonPHI(), E = MBB.end();;
         ++I) {
      dbgs() << PositionIndices[{&MBB, I}] << ": ";
      if (I == E) {
        dbgs() << "<end>\n";
        break;
      }
      dbgs() << *I;
    }
    dbgs() << '\n';
  }
}

void MOSRegAlloc::dumpTree(Node *Root, unsigned Indent) {
  if (!Root)
    Root = &Tree[0];
  for (unsigned I = 0; I < Indent; ++I)
    dbgs() << ' ';
  dbgs() << Root - &Tree[0];
  switch (Root->getType()) {
  case Node::Type::Forget:
    dbgs() << 'F';
    break;
  case Node::Type::Intro:
    dbgs() << 'I';
    break;
  case Node::Type::Join:
    dbgs() << 'J';
    break;
  }
  dbgs() << ": ";
  for (Position P : Root->Positions)
    dbgs() << PositionIndices[P] << ' ';
  dbgs() << '\n';
  for (Node *C : Root->Children)
    dumpTree(C, Indent + 1);
}

void MOSRegAlloc::solveTree(Node *Root) {
  if (!Root)
    Root = &Tree[0];
  switch (Root->getType()) {
  case Node::Type::Forget: {
    Node *Child = Root->Children.front();
    solveTree(Child);
    Position Forgotten;
    for (unsigned I = 0; I < Child->Positions.size(); ++I)
      if (I >= Root->Positions.size() ||
          Root->Positions[I] != Child->Positions[I])
        Forgotten = Child->Positions[I];
    dbgs() << "Forget " << PositionIndices[Forgotten] << '\n';

    SmallVector<Position> Preds;
    for (Position P : positionPredecessors(Forgotten))
      if (llvm::is_contained(Root->Positions, P))
        Preds.push_back(P);
    SmallVector<Position> Succs;
    for (Position P : positionSuccessors(Forgotten))
      if (llvm::is_contained(Root->Positions, P))
        Succs.push_back(P);

    for (Position P : Preds)
      dbgs() << "Predecessor " << PositionIndices[P] << '\n';
    for (Position P : Succs)
      dbgs() << "Successor " << PositionIndices[P] << '\n';

    SmallVector<Register> LiveRegs = getLiveRegs(Forgotten);
    for (Register R : LiveRegs)
      dbgs() << "Live reg " << printReg(R) << '\n';

    llvm_unreachable("TODO: Forget");
    break;
  }
  case Node::Type::Intro:
    if (Root->Children.empty()) {
      assert(Root->Positions.size() == 1 &&
             "leaves must have only one position");
      Root->Allocs.push_back({{std::nullopt}, 0});
    } else {
      Node *Child = Root->Children.front();
      solveTree(Child);
      for (const Alloc &ChildAlloc : Child->Allocs) {
        Root->Allocs.emplace_back();
        Alloc &A = Root->Allocs.back();
        A.Cost = ChildAlloc.Cost;
        for (unsigned I = 0, J = 0; I < Root->Positions.size(); ++I) {
          if (J < Child->Positions.size() &&
              Root->Positions[I] == Child->Positions[J]) {
            A.PosAllocs.push_back(ChildAlloc.PosAllocs[J++]);
          } else {
            A.PosAllocs.push_back(std::nullopt);
          }
        }
      }
    }
    break;
  case Node::Type::Join:
    for (Node *Child : Root->Children)
      solveTree(Child);
    llvm_unreachable("TODO: Join");
    break;
  }
  dbgs() << Root - &Tree[0];
  dbgs() << ":\n";
  for (Alloc &A : Root->Allocs) {
    dbgs() << A.Cost << "{\n";
    for (const auto &[I, PA] : llvm::enumerate(A.PosAllocs)) {
      if (PA)
        llvm_unreachable("TODO: Print present posalloc entry");
      else
        llvm::dbgs() << "  " << PositionIndices[Root->Positions[I]]
                     << " { * } ";
    }
    dbgs() << "\n}\n";
  }
}

SmallVector<Register> MOSRegAlloc::getLiveRegs(Position P) {
  SmallVector<Register> LiveRegs;
  for (unsigned I = 0, E = MF->getRegInfo().getNumVirtRegs(); I != E; ++I) {
    Register R = Register::index2VirtReg(I);
    if (!LV->isLiveIn(R, *P.MBB))
      continue;
    LiveRegs.push_back(R);
  }
  return LiveRegs;
}

char MOSRegAlloc::ID = 0;

INITIALIZE_PASS(MOSRegAlloc, DEBUG_TYPE, "MOS Register Allocation", false,
                false)

MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc(); }
