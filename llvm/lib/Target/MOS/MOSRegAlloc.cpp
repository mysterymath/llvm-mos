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

  void rewriteSSAValues();
  Register rewriteSSAValue(Register R);
  LLT findRegType(Register R);
  void allocateMBB(const MachineBasicBlock &MBB);
  void allocateMI(const MachineInstr &MI);
  void allocateMO(const MachineOperand &MO);
  void freeUse(const MachineOperand &MO);

  std::optional<Register> selectBestOperandReg(Alloc A,
                                               const MachineOperand &MO);

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
  MDT = &getAnalysis<MachineDominatorTreeWrapperPass>().getDomTree();
  rewriteSSAValues();
  LV.emplace(MF);
  MF.dump();
  allocateMBB(*MF.begin());
  applyBestAlloc();
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
  MIAllocs.emplace_back();
  DenseMap<Alloc, AllocImpl> &StartAllocs = MIAllocs.back();
  assert(MBB.isEntryBlock() && "TODO");
  Alloc Entry;
  for (auto LiveIn : MBB.liveins()) {
    assert(LiveIn.LaneMask.all() && "TODO");
    if (Alloc::isTracked(LiveIn.PhysReg))
      Entry[LiveIn.PhysReg] = LiveIn.PhysReg;
  }
  StartAllocs[Entry] =
      AllocImpl{/*Cost=*/0, /*IsCopy=*/false, /*PrevAlloc=*/{}};
  for (MachineBasicBlock::const_iterator I = MBB.getFirstNonPHI(),
                                         E = MBB.end();
       ; ++I) {
    if (I == E)
      break;
    allocateMI(*I);
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

  auto &Allocs = MIAllocs.back();
  NextAllocs.clear();
  for (auto &[A, AI] : Allocs) {
    if (MO.isDef()) {
      for (Register R : Regs) {
        if (A[R])
          continue;
        Alloc NewA = A;
        if (!MO.isDead())
          NewA[R] = V;
        NextAllocs[NewA] = AI;
      }
    } else {
      // TODO: Test for undef use
      assert(MO.isUse());
      Alloc &ARef = A;
      if (llvm::any_of(Regs, [&](Register R) {
            return Alloc::isTracked(R) && ARef[R] == V;
          }))
        NextAllocs[A] = AI;
    }
  }
  Allocs.swap(NextAllocs);
}

void MOSRegAlloc::freeUse(const MachineOperand &MO) {
  Register V = MO.getReg();
  auto &Allocs = MIAllocs.back();
  NextAllocs.clear();
  for (auto &[A, AI] : Allocs) {
    Alloc NewA = A;
    for (Register R : Alloc::Regs)
      if (A[R] == V)
        NewA[R] = {};
    NextAllocs[NewA] = AI;
  }
  Allocs.swap(NextAllocs);
}

std::optional<Register>
MOSRegAlloc::selectBestOperandReg(Alloc A, const MachineOperand &MO) {
  // TODO: Pick an register that meets RC and tie constraints. Merge this
  // logic with the code in allocateMO.
  return A.getReg(MO.getReg());
}

// Apply the best found allocation implementation.
void MOSRegAlloc::applyBestAlloc() {
  MachineBasicBlock &MBB = *MF->begin();
  dbgs() << "Applying best alloc.\n";
  unsigned I = MIAllocs.size() - 1;
  const auto Best = min_element(MIAllocs[I]);
  if (Best == MIAllocs[I].end())
    report_fatal_error("register allocation failed");
  Alloc A = Best->first;
  AllocImpl AI = Best->second;
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

    // TODO: Shuffles
  }
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
