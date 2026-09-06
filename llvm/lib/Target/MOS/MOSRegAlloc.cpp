//===-- MOSRegAlloc.cpp - Direct value placement --------------------------===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Experimental executable baseline for a direct constraint allocator. SSA
// values are distinct from their copies in hardware registers. Operand domains
// are searched together, including tied operands and scratch requirements.
//
// Deliberate first-stage restrictions: retain the input schedule; give each
// nonconstant byte/pair a colored zero-page backing location; flush live values
// at block boundaries. Only the backing locations are global decisions today.
// This is not yet a whole-function optimal allocator. No solver-specific model
// or synthetic issue slots are needed to express these restrictions.
//
//===----------------------------------------------------------------------===//

#include "MOSRegAlloc.h"
#include "MCTargetDesc/MOSMCTargetDesc.h"
#include "MOS.h"
#include "MOSRegisterInfo.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/LivePhysRegs.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"
#include <array>
#include <functional>

using namespace llvm;

static cl::opt<unsigned>
    PlacementLimit("mos-placement-limit", cl::Hidden, cl::init(100000),
                   cl::desc("Node budget for MOS backing-location search"));
static cl::opt<bool>
    PlacementStats("mos-placement-stats", cl::Hidden,
                   cl::desc("Print MOS value placement statistics"));

namespace {
constexpr std::array<MCPhysReg, 3> Hardware = {MOS::A, MOS::X, MOS::Y};

struct Value {
  Register Reg;
  unsigned Bits = 0;
  // Only operand-independent immediate materializations are recognized.
  MachineInstr *Immediate = nullptr;
  MCPhysReg Home = 0;
  BitVector Neighbors;
  SmallVector<MCPhysReg> Domain;
  SmallVector<unsigned> Preferences;
};
struct Block {
  MachineBasicBlock *MBB;
  SmallVector<MachineInstr *> Instructions;
  BitVector In, Out, Def, Use;
  unsigned FixedOut = 0;
  bool NZOut = false, CarryOut = false;
};
struct Point {
  BitVector Before, After;
  unsigned Fixed = 0;
  bool NZ = false, Carry = false;
};
struct Move {
  unsigned Opcode;
  MCPhysReg Dst, Src;
  unsigned Immediate = 0;
};
struct State {
  std::array<unsigned, 3> Contents = {};
  unsigned Carry = 0;
  BitVector Valid;
};
struct Plan {
  State S;
  SmallVector<Move> Moves;
  SmallVector<MCPhysReg> Operands;
  unsigned Cost = 0;
};
struct EdgeMove {
  MCPhysReg Dst, Src;
  unsigned Immediate = 0;
  EdgeMove(unsigned Dst, unsigned Src, unsigned Immediate = 0)
      : Dst(Dst), Src(Src), Immediate(Immediate) {}
};

class Placement {
  MachineFunction &MF;
  MachineRegisterInfo &MRI;
  const TargetInstrInfo &TII;
  const TargetRegisterInfo &TRI;
  SmallVector<Value> Values = {Value()}; // Zero denotes unknown contents.
  DenseMap<Register, unsigned> IDs;
  SmallVector<Block, 0> Blocks;
  DenseMap<MachineBasicBlock *, unsigned> BlockIDs;
  DenseMap<MachineInstr *, Point> Points;
  BitVector Reserved;
  unsigned Nodes = 0, Transfers = 0;

  [[noreturn]] void fail(const Twine &Reason, MachineInstr *MI = nullptr) {
    errs() << "MOS placement: " << Reason << " in " << MF.getName() << '\n';
    if (MI)
      errs() << *MI;
    report_fatal_error("unsupported MOS value placement problem");
  }
  unsigned id(Register R) {
    if (!R.isVirtual())
      return 0;
    auto It = IDs.find(R);
    if (It != IDs.end())
      return It->second;
    MachineInstr *D = MRI.getVRegDef(R);
    if (!D)
      fail("virtual register without SSA definition");
    if (D->isCopy() && D->getOperand(1).getReg().isVirtual() &&
        !D->getOperand(1).getSubReg())
      return IDs[R] = id(D->getOperand(1).getReg());
    unsigned Bits = TRI.getRegSizeInBits(*MRI.getRegClass(R)).getFixedValue();
    bool Imm = D->getOpcode() == MOS::LDImm || D->getOpcode() == MOS::LDImm1;
    if (Imm)
      for (unsigned I = 1; I < Values.size(); ++I)
        if (Values[I].Immediate && Values[I].Bits == Bits &&
            Values[I].Immediate->getOperand(1).isIdenticalTo(D->getOperand(1)))
          return IDs[R] = I;
    unsigned I = Values.size();
    Values.push_back(Value());
    Values[I].Reg = R;
    Values[I].Bits = Bits;
    Values[I].Immediate = Imm ? D : nullptr;
    return IDs[R] = I;
  }
  bool marker(const MachineInstr &MI) const {
    return MI.isPHI() || MI.getOpcode() == MOS::LDImm ||
           MI.getOpcode() == MOS::LDImm1 ||
           (MI.isCopy() && MI.getOperand(0).getReg().isVirtual() &&
            MI.getOperand(1).getReg().isVirtual() &&
            !MI.getOperand(1).getSubReg());
  }
  BitVector uses(MachineInstr &MI) {
    BitVector B(Values.size());
    if (!marker(MI))
      for (MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.isUse() && !MO.isUndef())
          if (unsigned V = id(MO.getReg()))
            B.set(V);
    return B;
  }
  BitVector defs(MachineInstr &MI) {
    BitVector B(Values.size());
    if (!marker(MI) || MI.isPHI())
      for (MachineOperand &MO : MI.operands())
        if (MO.isReg() && MO.isDef())
          if (unsigned V = id(MO.getReg()))
            B.set(V);
    return B;
  }
  void clique(const BitVector &Live) {
    for (int I : Live.set_bits()) {
      Values[I].Neighbors |= Live;
      Values[I].Neighbors.reset(I);
    }
  }
  void analyze();
  bool color();
  void allocateHomes();
  static int hardware(MCPhysReg R) {
    for (unsigned I = 0; I < Hardware.size(); ++I)
      if (Hardware[I] == R)
        return I;
    return -1;
  }
  bool overlap(MCPhysReg A, MCPhysReg B) const {
    return A && B && TRI.regsOverlap(A, B);
  }
  bool available(const State &S, const BitVector &Needed) const {
    for (int V : Needed.set_bits()) {
      if (Values[V].Immediate || S.Valid[V] ||
          llvm::is_contained(S.Contents, unsigned(V)) || S.Carry == unsigned(V))
        continue;
      return false;
    }
    return true;
  }
  bool preservesFlags(const Plan &P, bool NZ, bool Carry) const {
    for (const Move &M : P.Moves) {
      if (NZ && M.Opcode != MOS::STImag8 && M.Opcode != MOS::LDImm1)
        return false;
      if (Carry && M.Opcode == MOS::LDImm1)
        return false;
    }
    return true;
  }
  void writeHome(unsigned V, Plan &P) {
    for (unsigned I = 1; I < Values.size(); ++I)
      if (overlap(Values[I].Home, Values[V].Home))
        P.S.Valid.reset(I);
    P.S.Valid.set(V);
  }
  bool save(unsigned H, Plan &P, const BitVector &Needed) {
    unsigned V = P.S.Contents[H];
    if (!V || !Needed[V] || Values[V].Immediate || P.S.Valid[V])
      return true;
    if (!Values[V].Home)
      return false;
    P.Moves.push_back({MOS::STImag8, Values[V].Home, Hardware[H]});
    P.Cost += 2;
    writeHome(V, P);
    return true;
  }
  bool free(unsigned H, Plan &P, const BitVector &Needed, unsigned Fixed) {
    return !(Fixed & (1u << H)) && save(H, P, Needed);
  }
  bool ensure(unsigned V, MCPhysReg R, Plan &P, const BitVector &Needed,
              unsigned Fixed);
  bool copyByte(MCPhysReg Dst, MCPhysReg Src, unsigned Imm, Plan &P,
                const BitVector &Needed, unsigned Fixed);
  bool parallel(SmallVector<EdgeMove> Moves, Plan &P, const BitVector &Needed,
                unsigned Fixed);
  void emitMoves(MachineBasicBlock &MBB, MachineBasicBlock::iterator At,
                 const DebugLoc &DL, ArrayRef<Move> Moves);
  bool complete(MachineInstr &MI, Plan &P);
  Plan choose(MachineInstr &MI, const State &S);
  void emitBlock(Block &B);
  void emitEdges();

public:
  explicit Placement(MachineFunction &MF)
      : MF(MF), MRI(MF.getRegInfo()), TII(*MF.getSubtarget().getInstrInfo()),
        TRI(*MF.getSubtarget().getRegisterInfo()),
        Reserved(TRI.getReservedRegs(MF)) {}
  void run();
};

void Placement::analyze() {
  // Intern before allocating any bitsets. COPY names share semantic identity;
  // their original register classes do not constrain where a value can survive.
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      for (MachineOperand &MO : MI.operands())
        if (MO.isReg())
          id(MO.getReg());
  for (Value &V : Values)
    V.Neighbors.resize(Values.size());
  for (MachineBasicBlock &MBB : MF) {
    BlockIDs[&MBB] = Blocks.size();
    Block B{&MBB,
            {},
            BitVector(Values.size()),
            BitVector(Values.size()),
            BitVector(Values.size()),
            BitVector(Values.size())};
    for (MachineInstr &MI : MBB) {
      if (MI.isDebugInstr())
        continue;
      B.Instructions.push_back(&MI);
      BitVector U = uses(MI);
      U.reset(B.Def);
      B.Use |= U;
      B.Def |= defs(MI);
    }
    Blocks.push_back(std::move(B));
  }
  bool Changed;
  do {
    Changed = false;
    for (Block &B : reverse(Blocks)) {
      BitVector Out(Values.size());
      for (MachineBasicBlock *Succ : B.MBB->successors()) {
        Out |= Blocks[BlockIDs.lookup(Succ)].In;
        for (MachineInstr &Phi : Succ->phis())
          for (unsigned I = 1; I < Phi.getNumOperands(); I += 2)
            if (Phi.getOperand(I + 1).getMBB() == B.MBB)
              Out.set(id(Phi.getOperand(I).getReg()));
      }
      BitVector In = Out;
      In.reset(B.Def);
      In |= B.Use;
      Changed |= In != B.In || Out != B.Out;
      B.In = std::move(In);
      B.Out = std::move(Out);
    }
  } while (Changed);

  BitVector Used(Values.size());
  for (Block &B : Blocks) {
    BitVector Live = B.Out;
    Used |= Live;
    clique(Live);
    LivePhysRegs Phys(TRI);
    Phys.addLiveOutsNoPristines(*B.MBB);
    for (unsigned H = 0; H < Hardware.size(); ++H)
      if (Phys.contains(Hardware[H]))
        B.FixedOut |= 1u << H;
    B.NZOut = Phys.contains(MOS::N) || Phys.contains(MOS::Z);
    B.CarryOut = Phys.contains(MOS::C);
    for (MachineInstr *MI : reverse(B.Instructions)) {
      Point &P = Points[MI];
      P.After = Live;
      Live.reset(defs(*MI));
      Live |= uses(*MI);
      P.Before = Live;
      Used |= Live;
      clique(Live);
      // A result may reuse a dying input's home. Pair construction and PHI
      // edges use parallel copies, so permutations remain legal.
      BitVector During = P.After;
      During |= defs(*MI);
      clique(During);
      Phys.stepBackward(*MI);
      P.NZ = Phys.contains(MOS::N) || Phys.contains(MOS::Z);
      P.Carry = Phys.contains(MOS::C);
      for (unsigned H = 0; H < Hardware.size(); ++H)
        if (Phys.contains(Hardware[H]))
          P.Fixed |= 1u << H;
    }
  }

  // Exclude ABI-named ZP storage from the backing pool. Physical operands are
  // still emitted normally. The scavenger's reserved pair supplies a byte for
  // resolving cycles in parallel edge copies.
  for (Block &B : Blocks)
    for (MachineInstr *MI : B.Instructions)
      for (MachineOperand &MO : MI->operands())
        if (MO.isReg() && MO.getReg().isPhysical())
          for (MCPhysReg R : MOS::Imag8RegClass)
            if (overlap(R, MO.getReg()))
              Reserved.set(R);
  for (unsigned I = 1; I < Values.size(); ++I) {
    Value &V = Values[I];
    if (!Used[I] || V.Immediate || V.Bits == 1)
      continue;
    if (V.Bits != 8 && V.Bits != 16)
      fail("only byte and pair backing locations are implemented");
    const auto &RC = V.Bits == 8 ? MOS::Imag8RegClass : MOS::Imag16RegClass;
    for (MCPhysReg R : RC) {
      bool Allowed = !Reserved[R];
      if (V.Bits == 16)
        Allowed &= !Reserved[TRI.getSubReg(R, MOS::sublo)] &&
                   !Reserved[TRI.getSubReg(R, MOS::subhi)];
      for (Block &B : Blocks)
        for (MachineInstr *MI : B.Instructions)
          if (Points[MI].Before[I] && Points[MI].After[I])
            for (MachineOperand &MO : MI->operands())
              if (MO.isRegMask() && MO.clobbersPhysReg(R))
                Allowed = false;
      if (Allowed)
        V.Domain.push_back(R);
    }
    if (V.Domain.empty())
      fail("no legal backing location");
  }
  for (Block &B : Blocks)
    for (MachineInstr *MI : B.Instructions)
      for (MachineOperand &MO : MI->operands())
        if (MO.isReg() && MO.isDef() && MO.isTied() &&
            MO.getReg().isVirtual()) {
          Register Src =
              MI->getOperand(MI->findTiedOperandIdx(MO.getOperandNo()))
                  .getReg();
          if (Src.isVirtual()) {
            unsigned D = id(MO.getReg()), S = id(Src);
            Values[D].Preferences.push_back(S);
            Values[S].Preferences.push_back(D);
          }
        }
  for (Block &B : Blocks)
    for (MachineInstr &Phi : B.MBB->phis()) {
      unsigned D = id(Phi.getOperand(0).getReg());
      for (unsigned I = 1; I < Phi.getNumOperands(); I += 2) {
        unsigned S = id(Phi.getOperand(I).getReg());
        Values[D].Preferences.push_back(S);
        Values[S].Preferences.push_back(D);
      }
    }
}

bool Placement::color() {
  if (++Nodes > PlacementLimit)
    return false;
  unsigned Best = 0;
  SmallVector<MCPhysReg> Domain;
  for (unsigned I = 1; I < Values.size(); ++I) {
    Value &V = Values[I];
    if (V.Home || V.Domain.empty())
      continue;
    SmallVector<MCPhysReg> Legal;
    for (MCPhysReg R : V.Domain) {
      bool OK = true;
      for (int N : V.Neighbors.set_bits())
        if (overlap(R, Values[N].Home)) {
          OK = false;
          break;
        }
      if (OK)
        Legal.push_back(R);
    }
    if (Legal.empty())
      return false;
    if (!Best || Legal.size() < Domain.size() ||
        (Legal.size() == Domain.size() &&
         V.Neighbors.count() > Values[Best].Neighbors.count())) {
      Best = I;
      Domain = std::move(Legal);
    }
  }
  if (!Best)
    return true;
  // Prefer edge-copy elimination without making it a correctness constraint.
  auto Score = [&](MCPhysReg R) {
    unsigned S = 0;
    for (unsigned V : Values[Best].Preferences)
      S += Values[V].Home == R;
    return S;
  };
  llvm::stable_sort(
      Domain, [&](MCPhysReg A, MCPhysReg B) { return Score(A) > Score(B); });
  for (MCPhysReg R : Domain) {
    Values[Best].Home = R;
    if (color())
      return true;
    Values[Best].Home = 0;
    if (Nodes > PlacementLimit)
      break;
  }
  return false;
}

void Placement::allocateHomes() {
  if (!color())
    fail(Nodes > PlacementLimit ? "backing-location search budget exhausted"
                                : "backing locations exhausted");
}

bool Placement::ensure(unsigned V, MCPhysReg R, Plan &P,
                       const BitVector &Needed, unsigned Fixed) {
  if (!V)
    return false;
  Value &Val = Values[V];
  if (R == MOS::C) {
    if (P.S.Carry == V)
      return true;
    if (!Val.Immediate || Val.Bits != 1 ||
        (P.S.Carry && Needed[P.S.Carry] && !Values[P.S.Carry].Immediate))
      return false;
    P.Moves.push_back({MOS::LDImm1, R, 0, V});
    ++P.Cost;
    P.S.Carry = V;
    return true;
  }
  int H = hardware(R);
  if (H >= 0) {
    if (P.S.Contents[H] == V)
      return true;
    if (!free(H, P, Needed, Fixed))
      return false;
    if (Val.Immediate) {
      if (Val.Bits != 8)
        return false;
      P.Moves.push_back({MOS::LDImm, R, 0, V});
      P.Cost += 2;
    } else {
      int Source = -1;
      for (unsigned S = 0; S < Hardware.size(); ++S)
        if (P.S.Contents[S] == V && (S == 0 || H == 0))
          Source = S;
      if (Source >= 0) {
        P.Moves.push_back(
            {Source == 0 ? MOS::TA : MOS::T_A, R, Hardware[Source]});
        ++P.Cost;
      } else {
        if (!ensure(V, Val.Home, P, Needed, Fixed))
          return false;
        P.Moves.push_back({MOS::LDImag8, R, Val.Home});
        P.Cost += 2;
      }
    }
    P.S.Contents[H] = V;
    return true;
  }
  if (R && R == Val.Home) {
    if (P.S.Valid[V])
      return true;
    for (unsigned H = 0; H < Hardware.size(); ++H)
      if (P.S.Contents[H] == V) {
        P.Moves.push_back({MOS::STImag8, R, Hardware[H]});
        P.Cost += 2;
        writeHome(V, P);
        return true;
      }
  }
  return false;
}

bool Placement::copyByte(MCPhysReg Dst, MCPhysReg Src, unsigned Imm, Plan &P,
                         const BitVector &Needed, unsigned Fixed) {
  if (!Imm && Dst == Src)
    return true;
  Plan Best;
  Best.Cost = ~0u;
  for (unsigned H = 0; H < Hardware.size(); ++H) {
    Plan Trial = P;
    if (!free(H, Trial, Needed, Fixed))
      continue;
    Trial.Moves.push_back(
        {Imm ? MOS::LDImm : MOS::LDImag8, Hardware[H], Src, Imm});
    Trial.Moves.push_back({MOS::STImag8, Dst, Hardware[H]});
    Trial.Cost += 4;
    Trial.S.Contents[H] = Imm;
    if (Trial.Cost < Best.Cost)
      Best = std::move(Trial);
  }
  if (Best.Cost == ~0u)
    return false;
  P = std::move(Best);
  return true;
}

bool Placement::parallel(SmallVector<EdgeMove> Moves, Plan &P,
                         const BitVector &Needed, unsigned Fixed) {
  llvm::erase_if(
      Moves, [](const EdgeMove &M) { return !M.Immediate && M.Dst == M.Src; });
  while (!Moves.empty()) {
    bool Progress = false;
    for (unsigned I = 0; I < Moves.size(); ++I) {
      EdgeMove M = Moves[I];
      if (llvm::any_of(Moves, [&](const EdgeMove &Other) {
            return !Other.Immediate && Other.Src == M.Dst;
          }))
        continue;
      if (!copyByte(M.Dst, M.Src, M.Immediate, P, Needed, Fixed))
        return false;
      Moves.erase(Moves.begin() + I);
      Progress = true;
      break;
    }
    if (Progress)
      continue;
    MCPhysReg Src = Moves.front().Src;
    if (!copyByte(MOS::RC16, Src, 0, P, Needed, Fixed))
      return false;
    for (EdgeMove &M : Moves)
      if (!M.Immediate && M.Src == Src)
        M.Src = MOS::RC16;
  }
  return true;
}

void Placement::emitMoves(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator At, const DebugLoc &DL,
                          ArrayRef<Move> Moves) {
  for (const Move &M : Moves) {
    auto B = BuildMI(MBB, At, DL, TII.get(M.Opcode), M.Dst);
    if (M.Immediate)
      B.add(Values[M.Immediate].Immediate->getOperand(1));
    else
      B.addReg(M.Src);
    ++Transfers;
  }
}

// Evaluate a fully narrowed operand assignment by constructing its transfers.
// Failed scratch/preservation requirements reject the assignment before MIR is
// mutated. Hardware copies may coexist and any available copy can satisfy a
// use.
bool Placement::complete(MachineInstr &MI, Plan &P) {
  Point &Pt = Points[&MI];
  unsigned Fixed = Pt.Fixed;
  unsigned Inputs = Fixed;
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    const MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isUse() || MO.isUndef())
      continue;
    int H = hardware(P.Operands[I]);
    if (H >= 0)
      Inputs |= 1u << H;
  }
  // Make memory operands available first, then populate hardware operands.
  for (unsigned Pass = 0; Pass != 2; ++Pass)
    for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
      MachineOperand &MO = MI.getOperand(I);
      if (!MO.isReg() || !MO.isUse() || MO.isUndef() ||
          !MO.getReg().isVirtual())
        continue;
      MCPhysReg R = P.Operands[I];
      if ((hardware(R) >= 0) != (Pass == 1))
        continue;
      unsigned V = id(MO.getReg());
      if (R == Values[V].Home || hardware(R) >= 0 || R == MOS::C) {
        unsigned Protect = Pass == 1 ? Inputs & ~(1u << hardware(R)) : Fixed;
        if (!ensure(V, R, P, Pt.Before, Protect))
          return false;
      } else {
        // A tied memory operand is copied into the output's backing location.
        if (Values[V].Bits != 8 || !MOS::Imag8RegClass.contains(R))
          return false;
        if (Values[V].Immediate) {
          if (!copyByte(R, 0, V, P, Pt.Before, Fixed))
            return false;
        } else {
          if (!ensure(V, Values[V].Home, P, Pt.Before, Fixed) ||
              !copyByte(R, Values[V].Home, 0, P, Pt.Before, Fixed))
            return false;
        }
      }
    }
  // Preserve surviving hardware contents before the actual operation clobbers
  // them. Transfers used above must not destroy fixed physical operands.
  for (unsigned H = 0; H < Hardware.size(); ++H) {
    bool Clobber = false;
    for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
      MachineOperand &MO = MI.getOperand(I);
      Clobber |=
          MO.isReg() && MO.isDef() && overlap(P.Operands[I], Hardware[H]);
      Clobber |= MO.isRegMask() && MO.clobbersPhysReg(Hardware[H]);
    }
    if (Clobber) {
      if (!save(H, P, Pt.After))
        return false;
      P.S.Contents[H] = 0;
    }
  }
  bool ClobberCarry = false;
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    MachineOperand &MO = MI.getOperand(I);
    ClobberCarry |= MO.isReg() && MO.isDef() && overlap(P.Operands[I], MOS::C);
    ClobberCarry |= MO.isRegMask() && MO.clobbersPhysReg(MOS::C);
  }
  if (ClobberCarry) {
    unsigned Old = P.S.Carry;
    if (Old && Pt.After[Old] && !Values[Old].Immediate)
      return false;
    P.S.Carry = 0;
  }
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg() || !MO.isDef() || !MO.getReg().isVirtual())
      continue;
    unsigned V = id(MO.getReg());
    MCPhysReg R = P.Operands[I];
    if (int H = hardware(R); H >= 0)
      P.S.Contents[H] = V;
    else if (R == MOS::C)
      P.S.Carry = V;
    else if (R && R == Values[V].Home)
      writeHome(V, P);
  }
  return preservesFlags(P, Pt.NZ, Pt.Carry) && available(P.S, Pt.After);
}

Plan Placement::choose(MachineInstr &MI, const State &S) {
  SmallVector<SmallVector<MCPhysReg>> Domains(MI.getNumOperands());
  for (unsigned I = 0; I < MI.getNumOperands(); ++I) {
    MachineOperand &MO = MI.getOperand(I);
    if (!MO.isReg()) {
      Domains[I].push_back(0);
      continue;
    }
    if (!MO.getReg().isVirtual()) {
      Domains[I].push_back(MO.getReg());
      continue;
    }
    if (MO.getSubReg())
      fail("subregister operands are not implemented", &MI);
    unsigned V = id(MO.getReg());
    const TargetRegisterClass *RC = MI.getRegClassConstraint(I, &TII, &TRI);
    if (!RC)
      RC = MRI.getRegClass(MO.getReg());
    for (MCPhysReg R : Hardware)
      if (RC->contains(R))
        Domains[I].push_back(R);
    if (Values[V].Home && RC->contains(Values[V].Home))
      Domains[I].push_back(Values[V].Home);
    if (RC->contains(MOS::C))
      Domains[I].push_back(MOS::C);
    if (RC->contains(MOS::V))
      Domains[I].push_back(MOS::V);
    // The tied input may occupy its output's home even though it is a
    // different semantic value. Its transfer is explicit in complete().
    if (MO.isUse() && MO.isTied()) {
      unsigned D = MI.findTiedOperandIdx(I);
      MCPhysReg Home = Values[id(MI.getOperand(D).getReg())].Home;
      if (Home && RC->contains(Home) && !llvm::is_contained(Domains[I], Home))
        Domains[I].push_back(Home);
    }
    if (Domains[I].empty())
      fail("empty operand placement domain", &MI);
  }
  Plan Best;
  Best.Cost = ~0u;
  SmallVector<MCPhysReg> Assignment(MI.getNumOperands());
  std::function<void(unsigned)> Search = [&](unsigned I) {
    if (I == MI.getNumOperands()) {
      Plan P{S, {}, Assignment, 0};
      if (complete(MI, P) && P.Cost < Best.Cost)
        Best = std::move(P);
      return;
    }
    MachineOperand &MO = MI.getOperand(I);
    for (MCPhysReg R : Domains[I]) {
      bool OK = true;
      if (MO.isReg())
        for (unsigned J = 0; J < I; ++J) {
          MachineOperand &Other = MI.getOperand(J);
          if (!Other.isReg())
            continue;
          bool Tied = MO.isTied() && MI.findTiedOperandIdx(I) == J;
          if (Tied) {
            OK &= R == Assignment[J];
            continue;
          }
          if (!overlap(R, Assignment[J]) || MO.isUndef() || Other.isUndef())
            continue;
          if (MO.isDef() && Other.isDef())
            OK = false;
          if (MO.isUse() && Other.isUse() &&
              (!MO.getReg().isVirtual() || !Other.getReg().isVirtual() ||
               id(MO.getReg()) != id(Other.getReg())))
            OK = false;
          if ((MO.isEarlyClobber() && Other.isUse()) ||
              (Other.isEarlyClobber() && MO.isUse()))
            OK = false;
        }
      if (OK) {
        Assignment[I] = R;
        Search(I + 1);
      }
    }
  };
  Search(0);
  if (Best.Cost == ~0u)
    fail("operand and preservation constraints have no supported solution",
         &MI);
  return Best;
}

void Placement::emitBlock(Block &B) {
  State S;
  S.Valid = B.In;
  // PHIs are initialized by incoming edge copies.
  for (MachineInstr &Phi : B.MBB->phis())
    S.Valid.set(id(Phi.getOperand(0).getReg()));
  for (MachineInstr *MI : B.Instructions) {
    if (marker(*MI))
      continue;
    if (MI->isCopy()) {
      Register D = MI->getOperand(0).getReg(), Src = MI->getOperand(1).getReg();
      Plan P{S, {}, {}, 0};
      Point &Pt = Points[MI];
      if (D.isPhysical() && Src.isVirtual()) {
        unsigned V = id(Src);
        if (hardware(D) >= 0) {
          if (!ensure(V, D, P, Pt.Before, Pt.Fixed))
            fail("physical copy destination cannot be populated", MI);
        } else {
          unsigned Bits = Values[V].Bits;
          if (!Values[V].Immediate &&
              !ensure(V, Values[V].Home, P, Pt.Before, Pt.Fixed))
            fail("physical copy source unavailable", MI);
          if (Bits == 8) {
            if (!copyByte(D, Values[V].Home, Values[V].Immediate ? V : 0, P,
                          Pt.Before, Pt.Fixed))
              fail("physical byte copy needs scratch", MI);
          } else if (Bits == 16) {
            SmallVector<EdgeMove> Moves;
            for (unsigned Sub : {MOS::sublo, MOS::subhi})
              Moves.push_back(
                  {TRI.getSubReg(D, Sub), TRI.getSubReg(Values[V].Home, Sub)});
            if (!parallel(Moves, P, Pt.Before, Pt.Fixed))
              fail("physical pair copy needs scratch", MI);
          } else
            fail("physical flag copy unsupported", MI);
        }
      } else if (D.isVirtual() && Src.isPhysical()) {
        unsigned V = id(D);
        int H = hardware(Src);
        if (H >= 0) {
          if (!save(H, P, Pt.Before))
            fail("physical source preservation failed", MI);
          P.S.Contents[H] = V;
        } else if (Values[V].Bits == 16) {
          SmallVector<EdgeMove> Moves;
          for (unsigned Sub : {MOS::sublo, MOS::subhi})
            Moves.push_back(
                {TRI.getSubReg(Values[V].Home, Sub), TRI.getSubReg(Src, Sub)});
          if (!parallel(Moves, P, Pt.Before, Pt.Fixed))
            fail("pair capture needs scratch", MI);
          writeHome(V, P);
        } else
          fail("physical source copy unsupported", MI);
      } else
        fail("physical-to-physical COPY unsupported", MI);
      if (!preservesFlags(P, Pt.NZ, Pt.Carry) || !available(P.S, Pt.After))
        fail("copy would lose a live value or physical flag", MI);
      emitMoves(*B.MBB, MI->getIterator(), MI->getDebugLoc(), P.Moves);
      S = std::move(P.S);
      continue;
    }
    if (MI->getOpcode() == TargetOpcode::REG_SEQUENCE) {
      Plan P{S, {}, {}, 0};
      unsigned D = id(MI->getOperand(0).getReg());
      SmallVector<EdgeMove> Moves;
      for (unsigned I = 1; I < MI->getNumOperands(); I += 2) {
        unsigned V = id(MI->getOperand(I).getReg());
        if (!Values[V].Immediate &&
            !ensure(V, Values[V].Home, P, Points[MI].Before, Points[MI].Fixed))
          fail("pair source unavailable", MI);
        Moves.push_back(
            {TRI.getSubReg(Values[D].Home, MI->getOperand(I + 1).getImm()),
             Values[V].Home, Values[V].Immediate ? V : 0});
      }
      if (!parallel(Moves, P, Points[MI].Before, Points[MI].Fixed))
        fail("pair construction needs scratch", MI);
      writeHome(D, P);
      if (!preservesFlags(P, Points[MI].NZ, Points[MI].Carry) ||
          !available(P.S, Points[MI].After))
        fail("pair construction would lose a live value or physical flag", MI);
      emitMoves(*B.MBB, MI->getIterator(), MI->getDebugLoc(), P.Moves);
      S = std::move(P.S);
      continue;
    }
    Plan P = choose(*MI, S);
    emitMoves(*B.MBB, MI->getIterator(), MI->getDebugLoc(), P.Moves);
    for (unsigned I = 0; I < MI->getNumOperands(); ++I) {
      MachineOperand &MO = MI->getOperand(I);
      if (!MO.isReg())
        continue;
      if (MO.getReg().isVirtual()) {
        MO.setReg(P.Operands[I]);
        MO.setIsRenamable(false);
      }
      if (MO.isUse())
        MO.setIsKill(false);
      else
        MO.setIsDead(false);
    }
    S = std::move(P.S);
  }
  Plan Exit{S, {}, {}, 0};
  for (unsigned H = 0; H < Hardware.size(); ++H)
    if (!save(H, Exit, B.Out))
      fail("block output cannot be preserved");
  // STImag8 preserves hardware operands and flags of the terminator sequence.
  emitMoves(*B.MBB, B.MBB->getFirstTerminator(), DebugLoc(), Exit.Moves);
}

void Placement::emitEdges() {
  struct Edge {
    MachineBasicBlock *From, *To;
    SmallVector<EdgeMove> Moves;
  };
  SmallVector<Edge> Edges;
  for (Block &B : Blocks)
    for (MachineBasicBlock *Pred : B.MBB->predecessors()) {
      Edge E{Pred, B.MBB, {}};
      for (MachineInstr &Phi : B.MBB->phis()) {
        unsigned D = id(Phi.getOperand(0).getReg());
        if (!Values[D].Home)
          continue;
        for (unsigned I = 1; I < Phi.getNumOperands(); I += 2) {
          if (Phi.getOperand(I + 1).getMBB() != Pred)
            continue;
          unsigned V = id(Phi.getOperand(I).getReg());
          if (Values[D].Bits != 8)
            fail("only byte PHIs are implemented", &Phi);
          E.Moves.push_back(
              {Values[D].Home, Values[V].Home, Values[V].Immediate ? V : 0});
        }
      }
      llvm::erase_if(E.Moves, [](const EdgeMove &M) {
        return !M.Immediate && M.Dst == M.Src;
      });
      if (!E.Moves.empty())
        Edges.push_back(std::move(E));
    }
  for (Edge &E : Edges) {
    bool FallThrough = E.From->getFallThrough() == E.To;
    MachineBasicBlock *MBB = MF.CreateMachineBasicBlock();
    MF.push_back(MBB);
    E.From->ReplaceUsesOfBlockWith(E.To, MBB);
    if (FallThrough)
      BuildMI(*E.From, E.From->end(), DebugLoc(), TII.get(MOS::JMP))
          .addMBB(MBB);
    MBB->addSuccessor(E.To);
    Plan P;
    P.S.Valid.resize(Values.size());
    BitVector Needed(Values.size());
    const Block &From = Blocks[BlockIDs.lookup(E.From)];
    if (!parallel(E.Moves, P, Needed, From.FixedOut) ||
        !preservesFlags(P, From.NZOut, From.CarryOut))
      fail("parallel PHI copy failed");
    emitMoves(*MBB, MBB->end(), DebugLoc(), P.Moves);
    BuildMI(*MBB, MBB->end(), DebugLoc(), TII.get(MOS::JMP)).addMBB(E.To);
  }
}

void Placement::run() {
  if (PlacementStats)
    errs() << "MOS placement: analyzing " << MF.getName() << '\n';
  analyze();
  if (PlacementStats)
    errs() << "MOS placement: assigning backing locations\n";
  allocateHomes();
  if (PlacementStats)
    errs() << "MOS placement: emitting blocks\n";
  for (Block &B : Blocks)
    emitBlock(B);
  emitEdges();
  if (PlacementStats)
    errs() << "MOS placement: repairing physical liveness\n";
  // Keep immediate definitions alive until every rematerialization and edge
  // copy has been emitted. All analysis thereafter uses the frozen value graph.
  for (Block &B : Blocks)
    for (MachineInstr *MI : B.Instructions)
      if (marker(*MI) || MI->isCopy() ||
          MI->getOpcode() == TargetOpcode::REG_SEQUENCE)
        MI->eraseFromParent();
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      if (MI.isDebugInstr())
        for (MachineOperand &MO : MI.operands())
          if (MO.isReg() && MO.getReg().isVirtual())
            MO.setReg(0);
  MRI.clearVirtRegs();
  // Reconstruct physical liveness after edge insertion and instruction
  // realization; downstream copy propagation and scavenging consume it.
  for (MachineBasicBlock &MBB : MF)
    MBB.clearLiveIns();
  bool Changed;
  do {
    Changed = false;
    for (MachineBasicBlock &MBB : reverse(MF)) {
      SmallVector<MachineBasicBlock::RegisterMaskPair> Old(MBB.liveins());
      MBB.clearLiveIns();
      LivePhysRegs Live(TRI);
      computeAndAddLiveIns(Live, MBB);
      MBB.sortUniqueLiveIns();
      Changed |= !llvm::equal(Old, MBB.liveins());
    }
  } while (Changed);
  for (MachineBasicBlock &MBB : MF)
    recomputeLivenessFlags(MBB);
  if (PlacementStats)
    errs() << "MOS placement " << MF.getName() << ": " << Values.size() - 1
           << " values, " << Nodes << " backing search nodes, " << Transfers
           << " transfer/materialization instructions\n";
}

class MOSRegAlloc : public MachineFunctionPass {
public:
  static char ID;
  MOSRegAlloc() : MachineFunctionPass(ID) {
    initializeMOSRegAllocPass(*PassRegistry::getPassRegistry());
  }
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
  MachineFunctionProperties getSetProperties() const override {
    return MachineFunctionProperties().setNoVRegs().setNoPHIs();
  }
  MachineFunctionProperties getClearedProperties() const override {
    return MachineFunctionProperties().setIsSSA();
  }
  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!MF.getRegInfo().getNumVirtRegs())
      return false;
    Placement(MF).run();
    return true;
  }
};
} // namespace

char MOSRegAlloc::ID = 0;
INITIALIZE_PASS(MOSRegAlloc, "mos-regalloc", "MOS direct value placement",
                false, false)
MachineFunctionPass *llvm::createMOSRegAllocPass() { return new MOSRegAlloc; }
