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
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

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

bool operator<(const std::pair<Alloc, AllocImpl> &L,
               const std::pair<Alloc, AllocImpl> &R) {
  return L.second.Cost < R.second.Cost;
}

class MOSRegAlloc;

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
  MachineFunction *MF;
  MachineRegisterInfo *MRI;
  const TargetInstrInfo *TII;
  const TargetRegisterInfo *TRI;
  const MachineDominatorTree *MDT;

  SmallVector<Register, 0> RewrittenVReg;

  std::optional<LiveVariables> LV;

  // For each machine instruction (plus the end), a map from end alloc to the
  // best implementation.
  SmallVector<DenseMap<Alloc, AllocImpl>> MIAllocs;
  DenseMap<Alloc, AllocImpl> NextAllocs;

  SmallSet<Register, 16> RegCandValues;

  void rewriteSSAValues();
  Register rewriteSSAValue(Register R);
  LLT findRegType(Register R);
  void allocateMBB(const MachineBasicBlock &MBB);
  void allocateMI(const MachineInstr &MI);
  void allocateMO(const MachineOperand &MO);
  void freeUse(const MachineOperand &MO);

  std::optional<Register> selectBestOperandReg(Alloc A,
                                               const MachineOperand &MO);
  void shuffleAllocs(const MachineBasicBlock &MBB);
  void dumpNumAllocs(const MachineBasicBlock &MBB) const;

  void recordAllocImpl(Alloc A, AllocImpl AI);

  void applyBestAllocImpl(MachineBasicBlock &MBB);
  void eliminateTrivialCopies();
  void computeLiveIns();
};

} // namespace

bool MOSRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  this->MF = &MF;
  MRI = &MF.getRegInfo();
  TII = MF.getSubtarget().getInstrInfo();
  TRI = MF.getSubtarget().getRegisterInfo();
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  rewriteSSAValues();
  LV.emplace(MF);
  MF.dump();
  allocateMBB(*MF.begin());
  applyBestAllocImpl(*MF.begin());
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

void MOSRegAlloc::allocateMBB(const MachineBasicBlock &MBB) {
  MIAllocs.clear();
  MIAllocs.emplace_back();
  DenseMap<Alloc, AllocImpl> &StartAllocs = MIAllocs.back();
  assert(MBB.isEntryBlock() && "TODO");

  RegCandValues.clear();
  for (unsigned I = 0, E = MRI->getNumVirtRegs(); I != E; ++I) {
    Register V = Register::index2VirtReg(I);
    if (LV->isLiveIn(V, MBB))
      RegCandValues.insert(V);
  }

  Alloc Entry;
  for (auto LiveIn : MBB.liveins()) {
    assert(LiveIn.LaneMask.all() && "TODO");
    if (Alloc::isTracked(LiveIn.PhysReg)) {
      Entry[LiveIn.PhysReg] = LiveIn.PhysReg;
      RegCandValues.insert(LiveIn.PhysReg);
    }
  }

  StartAllocs[Entry] =
      AllocImpl{/*Cost=*/0, /*IsCopy=*/false, /*PrevAlloc=*/{}};

  for (MachineBasicBlock::const_iterator I = MBB.getFirstNonPHI(),
                                         E = MBB.end();
       ; ++I) {
    if (I == E)
      break;
    if (!I->isTerminator()) {
      shuffleAllocs(MBB);
      dumpNumAllocs(MBB);
    }
    allocateMI(*I);
    dumpNumAllocs(MBB);
  }

  if (!MBB.empty() && !std::prev(MBB.end())->isTerminator()) {
    shuffleAllocs(MBB);
    dumpNumAllocs(MBB);
  }
}

void MOSRegAlloc::allocateMI(const MachineInstr &MI) {
  dbgs() << "Allocating MI: " << MI;

  // Instantiate new allocs for MI.
  MIAllocs.emplace_back(MIAllocs.back());
  for (auto &[A, AI] : MIAllocs.back()) {
    AI.IsCopy = false;
    AI.PrevAlloc = A;
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

  SmallVector<Register> Regs;
  if (V.isVirtual()) {
    if (MO.getParent()->isCopy()) {
      const MachineInstr &MI = *MO.getParent();
      Regs.push_back(MI.getOperand(MI.getOperandNo(&MO) == 0 ? 1 : 0).getReg());
    } else {
      append_range(Regs, *TII->getRegClass(MO.getParent()->getDesc(),
                                           MO.getOperandNo(), TRI, *MF));
    }
  } else {
    Regs.push_back(V);
  }
  if (llvm::none_of(Regs, Alloc::isTracked))
    return;

  NextAllocs.clear();
  for (auto &[A, AI] : MIAllocs.back()) {
    if (MO.isDef()) {
      for (Register R : Regs) {
        if (A[R])
          continue;
        Alloc NewA = A;
        if (!MO.isDead())
          NewA[R] = V;
        recordAllocImpl(NewA, AI);
      }
    } else {
      // TODO: Test for undef use
      assert(MO.isUse());
      Alloc &ARef = A;
      if (llvm::any_of(Regs, [&](Register R) {
            return Alloc::isTracked(R) && ARef[R] == V;
          }))
        recordAllocImpl(A, AI);
    }
  }
  MIAllocs.back().swap(NextAllocs);

  if (MO.isDef() && !MO.isDead())
    RegCandValues.insert(V);
}

void MOSRegAlloc::freeUse(const MachineOperand &MO) {
  NextAllocs.clear();
  Register V = MO.getReg();
  for (auto &[A, AI] : MIAllocs.back()) {
    Alloc NewA = A;
    for (Register R : Alloc::Regs)
      if (A[R] == V)
        NewA[R] = {};
    recordAllocImpl(NewA, AI);
  }
  MIAllocs.back().swap(NextAllocs);

  RegCandValues.erase(V);
}

std::optional<Register>
MOSRegAlloc::selectBestOperandReg(Alloc A, const MachineOperand &MO) {
  // TODO: Pick an register that meets RC and tie constraints. Merge this
  // logic with the code in allocateMO.
  return A.getReg(MO.getReg());
}

// Find all transitively reachable allocs by spilling, restoring, and copying
// registers.
void MOSRegAlloc::shuffleAllocs(const MachineBasicBlock &MBB) {
  dbgs() << "Shuffling allocs\n";
  NextAllocs.clear();

  dbgs() << "Register candidate values: ";
  for (Register R : RegCandValues)
    dbgs() << " " << printReg(R, TRI);
  dbgs() << '\n';

  // Worklist sorted by cost
  std::map<unsigned, SmallVector<std::pair<Alloc, AllocImpl>>> WorkList;
  for (const auto &KV : MIAllocs.back())
    WorkList[KV.second.Cost].push_back(KV);

  // Run Dijkstra's to find the shortest-cost path to each alloc reachable
  // from a StartAlloc by some shuffle. NextAllocs contains the closed allocs.
  // TODO: Realistic costs
  // TODO: Realistic constraints
  while (true) {
    // Find a lowest-cost alloc in the worklist. Due to the Dijkstra's
    // invariant, the current path to this alloc will be a shortest path.
    while (!WorkList.empty() && WorkList.begin()->second.empty())
      WorkList.erase(WorkList.begin());
    if (WorkList.empty())
      break;
    Alloc A;
    AllocImpl AI;
    std::tie(A, AI) = WorkList.begin()->second.pop_back_val();

    // If we we have already closed A, then AI was not the shortest path.
    // Otherwise, it is.
    if (NextAllocs.contains(A))
      continue;
    NextAllocs[A] = AI;

    for (Register R : Alloc::Regs) {
      if (!A[R]) {
        for (Register V : RegCandValues) {
          // Ensure that R is a suitable register to hold V.
          if (V.isPhysical() && R != V)
            continue;
          if (MRI->getType(V).getScalarSizeInBits() !=
              TRI->getRegSizeInBits(R, *MRI))
            continue;

          // Copy or reload V to R.
          Alloc NewA = A;
          NewA[R] = V;
          if (NextAllocs.contains(NewA))
            continue;
          bool CanCopy = [&]() {
            for (Register Other : Alloc::Regs)
              if (Other != R && A[Other] == V)
                return true;
            return false;
          }();
          AllocImpl NewAI{AI.Cost + (CanCopy ? 2 : 3),
                          /*IsShuffle=*/true, A};
          WorkList[NewAI.Cost].emplace_back(NewA, NewAI);
        }
      } else {
        Register V = A[R];

        // Forget V or spill it to an imaginary register.
        Alloc NewA = A;
        NewA[R] = {};
        if (NextAllocs.contains(NewA))
          continue;
        AllocImpl NewAI{AI.Cost + (NewA.getReg(V) ? 0 : 3),
                        /*IsShuffle=*/true, A};
        WorkList[NewAI.Cost].emplace_back(NewA, NewAI);
      }
    }
  }

  MIAllocs.back().swap(NextAllocs);
}

void MOSRegAlloc::dumpNumAllocs(const MachineBasicBlock &MBB) const {
  dbgs() << "Num allocs: " << MIAllocs.back().size() << '\n';
}

// For the next set of allocs, if the given alloc impl is better than the
// current best, or if there is no curent impl, then record it as the new impl.
void MOSRegAlloc::recordAllocImpl(Alloc A, AllocImpl AI) {
  auto Res = NextAllocs.try_emplace(A, AI);
  if (!Res.second && AI.Cost < Res.first->second.Cost)
    Res.first->second = AI;
}

// Apply the best found allocation implementation.
void MOSRegAlloc::applyBestAllocImpl(MachineBasicBlock &MBB) {
  dbgs() << "Applying best alloc impl.\n";
  unsigned I = MIAllocs.size() - 1;
  Alloc A;
  AllocImpl AI = MIAllocs[I].at(A);
  for (MachineInstr &MI :
       llvm::reverse(llvm::make_range(MBB.getFirstNonPHI(), MBB.end()))) {
    assert(!AI.IsCopy && "TODO");
    for (MachineOperand &MO : MI.defs()) {
      if (!MO.getReg().isVirtual())
        continue;
      MO.setReg(*selectBestOperandReg(A, MO));
    }

    A = AI.PrevAlloc;
    AI = MIAllocs[--I].at(AI.PrevAlloc);

    for (MachineOperand &MO : MI.uses()) {
      if (!MO.isReg() || !MO.getReg().isVirtual())
        continue;
      MO.setReg(*selectBestOperandReg(A, MO));
    }
  }

#if 0
  while (true) {
    MachineIRBuilder Builder(MBB, MI);
    if (AI.IsCopy) {
      for (Register R : Alloc::Regs) {
        // Spill
        if (!AI.PrevAlloc.getReg(A[R]) && A[R]) {
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
