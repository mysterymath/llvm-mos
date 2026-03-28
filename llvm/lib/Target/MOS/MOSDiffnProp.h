//===-- MOSDiffnProp.h - Non-overlapping rectangle propagator ---*- C++ -*-===//
//
// Part of LLVM-MOS, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Chuffed propagator enforcing that no two "present" rectangles overlap in
// 2D integer space (X × Y). Each rectangle i has:
//   - x[i]:       IntVar, horizontal position
//   - w[i]:       int, horizontal width (>= 1)
//   - y_start[i]: IntVar, vertical interval start (inclusive)
//   - y_end[i]:   IntVar, vertical interval end (exclusive)
//   - present[i]: BoolView, whether the rectangle participates
//
// Y intervals are half-open: rectangle i occupies [y_start, y_end).
// Two present rectangles overlap iff their X ranges intersect AND their
// Y ranges intersect.
//
// This is a general-purpose propagator with no LLVM-specific knowledge.
// In the MOS register allocator, X = register unit, Y = schedule slot,
// and w = number of contiguous register units occupied.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_MOS_MOSDIFFNPROP_H
#define LLVM_LIB_TARGET_MOS_MOSDIFFNPROP_H

#include "chuffed/core/engine.h"
#include "chuffed/core/propagator.h"
#include "chuffed/core/sat.h"
#include "chuffed/support/vec.h"
#include "chuffed/vars/bool-view.h"
#include "chuffed/vars/int-var.h"

#include <cassert>
#include <vector>

/// Non-overlapping rectangle propagator with LCG explanations.
class MOSDiffnProp : public Propagator {
  int N;
  int NumXVals;
  vec<IntVar *> X;
  vec<int> W;
  vec<IntVar *> YStart;
  vec<IntVar *> YEnd;
  vec<BoolView> Present;

  enum InfType { INF_REMVAL, INF_START_MIN, INF_END_MAX };

  struct Pinfo {
    InfType Type;
    int Rect;
    int Cause;
    int XVal;
  };

  vec<Pinfo> PInfo;
  bool TrailedPInfoSz{false};

  Reason makeReason(InfType Type, int Rect, int Cause, int XVal) {
    if (!TrailedPInfoSz) {
      engine.trail.push(
          TrailElem(reinterpret_cast<int *>(&PInfo._size()), 4));
      TrailedPInfoSz = true;
    }
    PInfo.push({Type, Rect, Cause, XVal});
    return {prop_id, static_cast<int>(PInfo.size() - 1)};
  }

  void buildConflict(int AI, int BI) {
    vec<Lit> Ps;
    Ps.push(Present[AI].getValLit());
    Ps.push(Present[BI].getValLit());
    Ps.push(X[AI]->getValLit());
    Ps.push(X[BI]->getValLit());
    Ps.push(YStart[AI]->getMaxLit());
    Ps.push(YEnd[AI]->getMinLit());
    Ps.push(YStart[BI]->getMaxLit());
    Ps.push(YEnd[BI]->getMinLit());
    Clause *C = Clause_new(Ps);
    C->temp_expl = 1;
    sat.rtrail.last().push(C);
    sat.confl = C;
  }

  bool coversX(int I, int R) const {
    for (int Start = std::max(X[I]->getMin(), R - W[I] + 1);
         Start <= std::min(X[I]->getMax(), R); ++Start)
      if (X[I]->indomain(Start))
        return true;
    return false;
  }

  bool definitelyCoversX(int I, int R) const {
    return X[I]->getMin() >= R - W[I] + 1 && X[I]->getMax() <= R;
  }

public:
  MOSDiffnProp(vec<IntVar *> &XVars, vec<int> &Widths,
               vec<IntVar *> &YStarts, vec<IntVar *> &YEnds,
               vec<BoolView> &PresentFlags, int NumXVals)
      : N(XVars.size()), NumXVals(NumXVals) {
    assert(Widths.size() == N && YStarts.size() == N && YEnds.size() == N &&
           PresentFlags.size() == N);
    priority = 3;

    for (int I = 0; I < N; I++) {
      X.push(XVars[I]);
      W.push(Widths[I]);
      YStart.push(YStarts[I]);
      YEnd.push(YEnds[I]);
      Present.push(PresentFlags[I]);
    }

    for (int I = 0; I < N; I++) {
      int Base = I * 4;
      X[I]->attach(this, Base + 0, EVENT_C);
      YStart[I]->attach(this, Base + 1, EVENT_LU);
      YEnd[I]->attach(this, Base + 2, EVENT_LU);
      Present[I].attach(this, Base + 3, EVENT_F);
    }
  }

  void wakeup(int, int) override { pushInQueue(); }
  void clearPropState() override { in_queue = false; }

  bool propagate() override {
    TrailedPInfoSz = false;

    struct Task {
      int Idx, Est, Lst, Eet, Let;
    };

    for (int R = 0; R < NumXVals; R++) {
      std::vector<Task> Definite, Possible;
      for (int I = 0; I < N; I++) {
        if (!Present[I].isFixed() || !Present[I].isTrue())
          continue;
        if (!coversX(I, R))
          continue;
        Task T = {I, YStart[I]->getMin(), YStart[I]->getMax(),
                  YEnd[I]->getMin(), YEnd[I]->getMax()};
        if (definitelyCoversX(I, R))
          Definite.push_back(T);
        else
          Possible.push_back(T);
      }

      if (Definite.empty())
        continue;

      // Phase 1: Conflict — overlapping mandatory parts [LST, EET).
      for (size_t I = 0; I < Definite.size(); I++) {
        auto &A = Definite[I];
        if (A.Lst >= A.Eet)
          continue;
        for (size_t J = I + 1; J < Definite.size(); J++) {
          auto &B = Definite[J];
          if (B.Lst >= B.Eet)
            continue;
          if (A.Lst < B.Eet && B.Lst < A.Eet) {
            buildConflict(A.Idx, B.Idx);
            return false;
          }
        }
      }

      // Phase 2: Prune.
      for (auto &P : Possible) {
        if (W[P.Idx] != 1)
          continue;
        if (!X[P.Idx]->indomain(R))
          continue;
        for (auto &D : Definite) {
          if (D.Lst >= D.Eet)
            continue;
          if (!(P.Eet <= D.Lst || P.Lst >= D.Eet)) {
            Reason Rsn = makeReason(INF_REMVAL, P.Idx, D.Idx, R);
            if (!X[P.Idx]->remVal(R, Rsn))
              return false;
            break;
          }
        }
      }

      // Phase 3: Tighten Y bounds.
      for (size_t I = 0; I < Definite.size(); I++) {
        auto &A = Definite[I];
        for (size_t J = I + 1; J < Definite.size(); J++) {
          auto &B = Definite[J];
          bool ABeforeB = (A.Eet <= B.Lst);
          bool BBeforeA = (B.Eet <= A.Lst);

          if (!ABeforeB && !BBeforeA) {
            buildConflict(A.Idx, B.Idx);
            return false;
          }

          if (ABeforeB && !BBeforeA) {
            if (YStart[B.Idx]->setMinNotR(A.Eet)) {
              Reason Rsn = makeReason(INF_START_MIN, B.Idx, A.Idx, R);
              if (!YStart[B.Idx]->setMin(A.Eet, Rsn))
                return false;
            }
            if (YEnd[A.Idx]->setMaxNotR(B.Lst)) {
              Reason Rsn = makeReason(INF_END_MAX, A.Idx, B.Idx, R);
              if (!YEnd[A.Idx]->setMax(B.Lst, Rsn))
                return false;
            }
          } else if (BBeforeA && !ABeforeB) {
            if (YStart[A.Idx]->setMinNotR(B.Eet)) {
              Reason Rsn = makeReason(INF_START_MIN, A.Idx, B.Idx, R);
              if (!YStart[A.Idx]->setMin(B.Eet, Rsn))
                return false;
            }
            if (YEnd[B.Idx]->setMaxNotR(A.Lst)) {
              Reason Rsn = makeReason(INF_END_MAX, B.Idx, A.Idx, R);
              if (!YEnd[B.Idx]->setMax(A.Lst, Rsn))
                return false;
            }
          }
        }
      }
    }

    return true;
  }

  Clause *explain(Lit, int InfID) override {
    auto &PI = PInfo[InfID];
    int I = PI.Rect;
    int J = PI.Cause;

    vec<Lit> Ps(1);
    Ps.push(Present[J].getValLit());
    Ps.push(X[J]->getValLit());

    switch (PI.Type) {
    case INF_REMVAL:
      Ps.push(Present[I].getValLit());
      Ps.push(YStart[J]->getMaxLit());
      Ps.push(YEnd[J]->getMinLit());
      Ps.push(YStart[I]->getMaxLit());
      Ps.push(YEnd[I]->getMinLit());
      break;
    case INF_START_MIN:
      Ps.push(Present[I].getValLit());
      Ps.push(X[I]->getValLit());
      Ps.push(YEnd[J]->getMinLit());
      Ps.push(YStart[I]->getMaxLit());
      break;
    case INF_END_MAX:
      Ps.push(Present[I].getValLit());
      Ps.push(X[I]->getValLit());
      Ps.push(YStart[J]->getMaxLit());
      Ps.push(YEnd[I]->getMinLit());
      break;
    }

    Clause *Expl = Clause_new(Ps);
    Expl->temp_expl = 1;
    sat.rtrail.last().push(Expl);
    return Expl;
  }
};

#endif // LLVM_LIB_TARGET_MOS_MOSDIFFNPROP_H
