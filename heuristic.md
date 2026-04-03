# Heuristic for Greedy Combined Scheduling + Register Allocation

## Overview

**Input**: A MIR basic block in SSA form, after instruction selection.
**Output**: A fully scheduled and register-allocated instruction sequence.

The algorithm builds on a key abstraction: **COPY-in instructions**
make each cluster a hermetic unit. Every value lives entirely within
one cluster (as an internal value) or crosses into a cluster through
an explicit COPY-in at the cluster's entry. These COPYs are SSA-form
renaming points; some become real machine instructions, others get
eliminated by Phase 4 coalescing.

The algorithm has one scheduling phase, not two: **Phase 2** iteratively
merges clusters via insertion until a single cluster remains. That
cluster's schedule is the final instruction sequence. Phases 3 and 4
are post-processing (effect summary and concrete register assignment).

## Definitions

**Value**: A virtual register in the MIR.

**Cluster**: A set of instructions with a fixed internal schedule.
Initially each instruction is its own singleton cluster. Merging
combines clusters; the final single cluster's schedule is the block's
instruction sequence.

**COPY-in**: A `COPY` instruction at a cluster's entry whose source is
a vreg in another cluster:

```
%v_local = COPY %v_external    ; %v_external defined elsewhere
```

COPY-ins enforce the cluster invariant (§2.1): every value used within
a cluster is defined within it. Phase 4 coalescing eliminates COPY-ins
whose source and destination land on the same physical register.

**Tied-def**: An instruction output that must occupy the same physical
register as one of its inputs. The instruction destructively overwrites
the input.

**Mobility** of a register class: Combines register availability with
copy cost.

    mobility(C) = |available_registers(C)| / copy_cost(C)

Costs come from `MOSRegisterInfo::copyCost()` which returns
`MOSInstrCost(Bytes, Cycles)`.

### Copy cost table (standard 6502)

| Source → Dest     | Mechanism                  | Bytes | Cycles | Clobbers |
|-------------------|----------------------------|-------|--------|----------|
| A → X/Y           | TAX / TAY                  | 1     | 2      | —        |
| X/Y → A           | TXA / TYA                  | 1     | 2      | —        |
| X ↔ Y             | TXA+TAY (+ maybe PHA/PLA)  | ~3    | ~7     | A        |
| A → Imag8         | STA zp                     | 2     | 3      | —        |
| Imag8 → A         | LDA zp                     | 2     | 3      | —        |
| X/Y → Imag8       | STX/STY zp                 | 2     | 3      | —        |
| Imag8 → X/Y       | LDX/LDY zp                 | 2     | 3      | —        |
| Imag8 → Imag8     | LDA+STA zp (+ maybe PHA/PLA)| ~5   | ~9     | A        |
| Imag8_lsb → C     | [LDA zp +] CMP #1          | 2–4   | 2–5    | A        |
| C → Imag8_lsb     | LDA #0; BCC +2; LDA #1     | ~6    | ~7     | A        |
| C ↔ V             | BIT setv + BR + CLV        | 6     | 9      | —        |
| Imag8_lsb → V     | [LDA +] BNE+BIT+JMP+CLV    | ~11   | ~15    | A        |

### Mobility ranking

| Class  | Registers | Cheapest save/restore | Cycles | Mobility  |
|--------|-----------|----------------------|--------|-----------|
| Vc     | 1 (V)     | BIT+BR+CLV           | ~9     | very low  |
| Cc     | 1 (C)     | CMP #1 / LDA+BCC+LDA | 5–7    | low       |
| Ac     | 1 (A)     | STA zp / LDA zp      | 3      | moderate  |
| XY     | 2 (X, Y)  | STX zp / LDX zp      | 3      | moderate  |
| Imag8  | 256 (32 practical) | LDA+STA zp    | ~9     | very high |
| Anyi8  | 259       | STA zp / LDA zp      | 3      | very high |

Cc and Vc copies are expensive but not impossible.

## Phase 1: Setup

There is no copy propagation or class widening. The MIR's existing
COPY instructions are kept; vregs retain their declared (narrow)
register classes. The algorithm operates on the input MIR as-is, with
the following setup steps.

### 1.1 Why we keep the COPYs

Aggressive copy propagation in earlier versions of this algorithm
created two problems:

1. **Physical-register inputs became "external" values without
   in-IR vregs.** A livein like `$x` (the X register) had no SSA name
   the algorithm could grab onto. When an instruction needed `$x` in a
   different class, there was no way to insert a copy "for $x" because
   $x wasn't a vreg the algorithm could manipulate.

2. **Class transitions had no anchor.** A statement like
   `%1:imag8 = COPY $x` was a real STX-to-zp instruction that performs
   a class transition from {X} to Imag8. Propagating it left I15
   reading `$x` directly, with a constraint requiring Imag8 — a
   class mismatch with no in-IR copy to handle it.

By keeping the COPYs, every value becomes an in-IR vreg with a
specific narrow class. Class transitions are explicit instructions.
The algorithm can manipulate vregs (insert save/restore copies, merge
them into clusters) using its normal mechanisms.

Many of the original COPYs are no-ops (e.g., `%0:ac = COPY $a` is just
A → A) and will be eliminated by Phase 4 coalescing. They cost nothing
in the final code; they just give the algorithm named SSA points to
work with.

### 1.2 Setup steps

The setup is minimal:

- **The livein cluster**: create exactly one **start cluster** named
  `K_livein` that publishes every block-livein physreg (`$a, $x,
  $rc2, $rc3, ...`) as a value pinned to that physreg with a singleton
  class. There are **no per-cluster livein anchors** — every livein
  lives in this single cluster, and any other cluster that wants a
  livein consumes it via a COPY-in (the cluster invariant treats
  livein consumption like any other cross-cluster reference).
  `K_livein` participates in priority-queue merging like any other
  cluster: when it merges with cluster X, all liveins are then
  considered to be defined in the merged cluster, and other clusters
  that still need them route their COPY-ins to that merged cluster.

- **Identify the terminator block**: the block-ending instruction(s)
  (RTS, branch, etc.) form a fixed terminal block. They are not
  scheduled or merged; they're appended to the final schedule
  unchanged.

- **Note any regmasks**: function calls and inline asm carry regmasks
  that describe clobbered physregs. These are constraints applied at
  the regmask instruction's position. (Calls are out of scope for the
  current single-block-only setting; this note is for completeness.)

That's it for Phase 1. The vreg classes, the data flow, and the COPY
instructions are all kept as-is from the input MIR.

### 1.3 Special instructions in detail

#### Physical register defs and uses

A **physreg use** is just an operand whose constraint class is a singleton
`{R}`. The slack-cascade check enforces this naturally: when the instruction
is scheduled, the value being consumed must have an effective class that
includes `R`, and the cascade narrows it to `{R}` if needed (or fails if
some other live value is also forced to `R`).

A **physreg def** produces a value whose declared class is a singleton.
The value enters the active set with `effective = {R}` and stays there
for its entire live range.

The block's livein physregs are published by the single `K_livein`
start cluster (per §1.2). Any cluster that needs a livein consumes it
via a COPY-in sourcing from `K_livein` (or from whichever cluster has
absorbed `K_livein` by then). They typically feed into COPY
instructions that materialize them as vregs in the desired class for
downstream use; those COPYs are themselves the COPY-ins.

#### Terminators

The block-ending terminator (RTS, branch, etc.) is treated as a **fixed
terminal block** — a sequence of one or more instructions that sits at
the end of the basic block and **cannot be touched** by any phase of the
algorithm. The cluster formation and allocation phases operate on
the body of the block (everything
before the terminator). The terminator is appended verbatim at the very
end of the final schedule.

The terminator can still impose **live-out constraints** on the body:
any physreg use in the terminator (e.g., a return-with-value pattern
where A holds the return value) becomes a constraint that the named
value must be in the required physreg at the end of the body schedule.
These live-out constraints are checked by the slack cascade at the
boundary between the body and the terminal block: each terminator-used
value must have an effective class that includes its required physreg
at that point.

If a value used by the terminator is multi-use (also consumed by some
body instruction), Phase 2 treats it as a normal value with the
additional constraint that it must reach the body-terminator boundary
in the right register class. Effectively, the terminator contributes
"phantom uses" at the very end of the body that the slack mechanism
must satisfy.

#### Register mask clobbers

Some instructions carry a **regmask** — a bitmask of physical registers
that are *not preserved* across the instruction. This is how LLVM
represents the clobber lists of:
- Function calls (caller-saved physregs are clobbered)
- Inline assembly (the asm clobber list)

A regmask is logically distinct from an instruction's normal operand
def/use list. The mask says "any value live across this instruction
must not occupy any register in the mask."

In the slack model, a regmask at instruction `I` is a constraint at
`I`'s position: for every value `v` live across `I`, the cascade must
narrow `v.effective` so that `v.effective ∩ regmask = ∅`. Concretely:

- For each live `v` whose effective overlaps the regmask: subtract the
  regmask from `v.effective`.
- If `v.effective` becomes empty, the schedule is infeasible at this
  point — the value's class was entirely contained in the clobbered
  set, so it can't survive across `I`.
- The cascade then propagates as usual: shrunk values may now overlap
  others in newly-tighter classes, requiring further narrowings.

Function calls are out of scope for this version of the algorithm
(we're focusing on single-block straight-line code without call
boundaries). Regmask handling is documented here for completeness and
because **inline assembly** uses the same mechanism even within a
single block.

#### Tied-def relaxation

Tied-defs are no longer hard constraints requiring the input and output
to share a physical register. The algorithm treats them with two options:

1. **0-copy merge (preferred)**: the input and output share a
   register, the tied chain is preserved, no copy needed. This is the
   default and what cheapest-first naturally tries first.

2. **Adaptive copy at the use site**: insert a `COPY` immediately before
   the tied use, reading the input from wherever it currently lives and
   writing a fresh value into the register the tied-def needs. The
   instruction with the tied-def then operates on the fresh value (which
   has its own slot), and the original input value is unaffected.

Option 2 is the same save-and-rewrite mechanism used for other
register conflicts. A failed tied-def merge falls back to inserting
a copy right before the tied use, breaking the chain at the cost of
one copy.

This means tied-def chains are *preferred* groupings (the priority
queue naturally tries them first since they're 0-cost merges) but not
*required* groupings. When a chain can't be preserved without a
register conflict, the relaxation gives the algorithm a way out.

## Phase 2: Cluster Formation (always-merge to a single cluster)

### 2.1 Cluster Formation by Priority-Queue Cluster-Set Merging

A **cluster** is a partially-scheduled subgraph of the data dependency
graph. It carries an internal instruction sequence (its schedule) and
the set of values flowing through it. Initially, each instruction is its
own cluster — a singleton schedule.

Cluster formation proceeds by **iteratively merging cluster sets** until
**a single cluster remains**. Each merge takes two or more clusters and
absorbs them into one larger cluster containing all of their instructions,
with every data-flow path that becomes internal to the merged set being
internalized simultaneously.

The unit of merging is a **cluster set**, not a single edge — and
not necessarily a single pair, either. A multi-use vreg with N consumers
spread across N different clusters wants an (N+1)-way merge to be fully
internalized; we don't have to do it as N separate binary merges.

**Always merge.** Every merge attempt is required to succeed. There is
no "leave as a boundary" option — the algorithm runs until exactly one
cluster remains, and the final cluster's internal schedule **is** the
basic block's complete instruction sequence. There is no separate
inter-cluster scheduling phase.

#### Cluster invariant

> **Every value used within a cluster is defined within that cluster.**

Each cluster has its own private vreg space. External values are brought
in via **COPY-in instructions** at the cluster's entry. A COPY-in is a
regular MIR `COPY` instruction whose source is a vreg defined in some
other cluster and whose destination is a fresh vreg in this cluster's
space:

```
%v_local = COPY %v_external    ; %v_external is some other cluster's vreg
```

When clusters are merged, the COPY-in becomes a regular intra-cluster
copy. At register-allocation time, if `%v_local` and `%v_external` end
up in the same physreg, the COPY is coalesced away; otherwise it
becomes a real instruction.

The cluster invariant means: a cluster's internal slack and scheduling
analysis is *self-contained*. It doesn't need to look at any other
cluster's state. The COPY-in placeholder is a normal instruction that
the cluster's slack check sees as "this consumes external value
`%v_external` into local `%v_local`."

#### Live-out and the `killed` flag

Every use operand in MIR carries a `killed` flag (existing semantics).
A `killed` use is the last use of its value within the function. For our
purposes, a value defined in cluster `C` is **live-out from C** unless
its `killed` use is also inside `C`. Equivalently:

- If `C` contains the value's `killed` use → value dies inside `C`,
  its register is free at C's exit.
- If the `killed` use is in some other cluster → value is live-out from
  `C`, must be in some register at C's exit.

The cluster's schedule must respect this. Concretely: at the cluster's
exit point, every live-out value must occupy a physical register
compatible with its declared class.

#### How merging works

When we merge cluster set `S` (typically two clusters, but possibly
more for multi-use values), the basic mechanism is **insertion**:

1. Pick a **base cluster** from `S`. The algorithm **tries both
   directions** for binary merges (each cluster as base) and picks the
   cheapest successful result. For N-way merges, a good heuristic is
   "the cluster with the shortest schedule is the base."
2. For each other cluster in `S`, find an **insertion position** in
   the base cluster's schedule and insert that cluster's instructions
   as a contiguous block at that position.
3. As part of the insertion, fill in the inserted cluster's COPY-in
   sources with the base cluster's defs (which are now internal
   references) and elide COPY-ins where source = destination class.
4. Verify with the slack-cascade colorability check.

There is **no separate "interleave" mode** that splits clusters apart
into individual instructions. Each cluster's internal schedule is
preserved as a contiguous block. The merge picks where in the base
cluster's schedule to insert each block.

There is also **no separate "concatenation" tier**. Insertion at the end
of the base cluster's schedule is just one of the valid insertion
positions, considered alongside all earlier positions.

#### Valid insertion positions and allowed saves

A merge coalescing values `%v1–%vn` may only insert copies for:

1. **The coalesced values** `%v1–%vn` themselves.
2. **Live-out values with no remaining intra-cluster uses** after the
   insertion point. These are values that are "done" inside the base
   cluster — they only need to be preserved for consumers in external
   clusters.

An insertion position is **invalid** if the inserted block would
clobber a register holding a value that still has uses in the base
cluster after that position (and the value is not one of `%v1–%vn`).
Intra-cluster def/use chains bind more tightly than cross-cluster
ones: the algorithm never breaks them.

When no interior position is valid, the inserted cluster goes **at
the end** of the base cluster's schedule. Any COPY-in sources that
were clobbered by the base cluster before the insertion point are
handled by **save-and-rewrite** (see below).

#### Earliest-position tiebreaker

Among valid positions with equal cost, the algorithm picks the
**earliest** (front-to-back scan, first minimum wins).

The justification: the coalesced values `%v1–%vn` have their live
range span from def (typically in the base cluster) to use (in the
inserted block). Earlier insertion puts the use closer to the def,
**shortening the coalesced values' live ranges**. Shorter ranges mean
fewer conflicts with future merges — less register pressure during
the critical span where `%v1–%vn` are alive.

#### Save and rewrite for live-out preservation

Save-and-rewrite handles two situations:

1. **The inserted block clobbers a base-cluster live-out value**
   that has no remaining intra-cluster uses after the insertion point.
   The consumers are in other clusters (not in this merge).
2. **The base cluster clobbers a value** that the inserted cluster
   consumes via a COPY-in (because the inserted cluster lands after
   the clobber point in the base cluster's schedule).

In both cases the mechanism is the same: insert a **save copy** before
the clobber, then **rewrite COPY-in sources** to read the saved value.

```
%v_save = COPY %v       ; %v_save is a fresh vreg in some wide cheap class
```

The save is placed at the **latest valid position** before the first
clobbering instruction. "Latest" minimizes the saved value's live range.

The save's destination class is the **cheapest, broadest** class
reachable from `%v`'s current class:

- For Ac → Imag8 (STA zp, ≈3 cycles, 32-wide).
- For XY → Imag8 (STX/STY zp, ≈3 cycles).
- For Cc → Imag8 (via the bit-materialization sequence, more expensive).
- For Imag8 → already wide; no save needed in practice.

**Cheapest, broadest** because:
- Cheap minimizes the cost charged to this merge.
- Broad keeps maximum flexibility for downstream merges that might
  need to materialize the value into some specific narrow class.

After inserting the save, the merge **rewrites COPY-in sources** that
originally read `%v` to now read `%v_save`. For case 1, these are
COPY-ins in *other* clusters not in this merge. For case 2, these are
COPY-ins in the *inserted* cluster itself — the COPY-in instruction
serves as the "reload," just with a renamed source; no separate reload
instruction is needed.

The original `%v` is then effectively "ended" at the save (its
register holds bits that nobody references anymore until the clobber
overwrites them).

Because intra-cluster uses are never clobbered (see above), there is
**no intra-cluster spill+reload**. Every save-and-rewrite targets a
COPY-in, either in an external cluster or in the inserted cluster.
The COPY-in is always the reload.

This save-and-rewrite, combined with the try-both-directions search
and the earliest-position tiebreaker, guarantees that the merge
always succeeds.

#### Cost: cheapest merge first

Since every merge will eventually be committed (always-merge), the
question is the *order*. We use the most direct measure available:

```
actual_cost(merge) = Σ over copies c emitted during the attempt: copyCost(c)
```

The priority queue orders merges by `actual_cost` **ascending** —
cheapest merges first. This is a direct measure of "what does this
merge cost right now," with no proxy or savings calculation.

Why cheapest first: cheap merges (especially 0-copy ones) make progress
without burning resources. They commit to easy register choices that
don't conflict with much else. Saving the expensive merges for last
means they happen against a state where most of the easy commitments
are already in place — the algorithm has the most information when it
tackles the hardest merges.

Tied edges naturally come first: a tied-def chain has 0-copy merges
throughout if the class slots are available. The insertion mechanism
will succeed at zero cost as long as the slack cascade can satisfy the
tied chain's register requirement. The first opportunity a tied chain
gets, it wins (cost 0).

**No `max_cost`, no `savings`**. The previous formulation tried to
compute "what would this cost if left as a boundary" as a proxy for
regret. With always-merge, that's not a real alternative — the merge
will happen one way or another. The order is the only decision, and
`actual_cost` is the right metric for it.

#### Tied-def chains as merges

A tied-def chain is a sequence of instructions linked by tied edges, all
forced to share a single physical register. In the cluster-set view, a
tied chain emerges naturally as the priority-queue algorithm absorbs
each chain edge in turn:

- Initially, each instruction is its own cluster, and the tied edges
  are 0-cost binary merge candidates between consecutive singletons.
- After a few merges, two adjacent chain segments become one cluster.
  The next merge inserts the next chain link.

We do **not** pre-build tied-def chains as initial clusters. Multi-tied
instructions like `ADCImm` have *several* tied chains (one through Ac,
one through Cc, possibly one through Vc) that would conflict if forced
into a single seed. Which chain `ADCImm` joins first is itself a
decision the priority queue makes naturally — based on relative merge
costs, with the slack check ensuring feasibility.

#### Per-merge attempt

```
function try_merge(cluster_set S):
    best ← None

    // Try both directions for binary merges (each cluster as base).
    // For N-way merges, try the cluster with the shortest schedule
    // as base, plus optionally the def cluster if different.
    for each candidate base b in S:
        others ← S \ {b}

        // Enumerate valid insertion positions FRONT-TO-BACK.
        // A position is INVALID if the inserted block would clobber
        // a value that still has intra-cluster uses after that point
        // (unless the value is one of the coalesced %v1–%vn).
        // "At the end" is always valid.
        // Front-to-back: first position at minimum cost wins
        // (shortest coalesced-value live ranges).
        for each insertion plan P (a position in b.schedule for each
                                   cluster in others, scanning front-to-back,
                                   respecting data deps and validity rule):
            candidate ← copy of b.schedule

            for each (cluster c, position pos) in P (in topological order):
                candidate ← insert c.schedule into candidate at pos
                // Resolve c's COPY-ins:
                //   for each COPY-in `%v_local = COPY %v_external` in c,
                //   substitute %v_external's current location in candidate.
                //   If source class = dest class, mark for elision.

            // Save-and-rewrite. Only allowed for:
            //   (a) the coalesced values %v1–%vn, and
            //   (b) live-out values with no remaining intra-cluster
            //       uses after the insertion point.
            // Two shapes:
            //   (1) Inserted block clobbers a base-cluster value
            //       satisfying (b) → save, rewrite external COPY-ins.
            //   (2) Base cluster clobbers a value the inserted cluster
            //       needs via COPY-in → save, rewrite that COPY-in.
            // The COPY-in IS the reload in both cases.
            for each value v that would be clobbered and qualifies:
                save_pos ← latest valid position before the first clobber
                wide_class ← cheapest broad class reachable from v's class
                insert `%v_save = COPY v` at save_pos
                for each COPY-in that reads v (external or inserted):
                    rewrite source from v to %v_save

            if not check_colorable(candidate): continue

            cost ← total copyCost of unelided copies in candidate
            if best is None or cost < best.cost:
                best = (candidate, cost)

    if best: return Success(best.schedule, cluster_set=S, actual_cost=best.cost)
    return Fail    // very rare; truly infeasible without memory spills
```

The merge enumerates valid insertion plans across both base directions,
scanning front-to-back, and picks the earliest minimum-cost plan. The
validity rule rejects positions where the inserted block would clobber
a value that still has intra-cluster uses; the "at the end" fallback
is always available. The earliest-position tiebreaker minimizes the
coalesced values' live ranges, reducing future merge conflicts.

Saves are restricted to the coalesced values and to live-out values
with no remaining intra-cluster uses after the insertion point.
Save-and-rewrite targets **COPY-in sources** — either in external
clusters or in the inserted cluster itself. The COPY-in instruction
serves as the reload; no separate reload is ever inserted.

**Why no interleaving tier**: each cluster's internal schedule was
already chosen to be optimal for its own constraints. Splitting a
cluster apart into individual instructions and re-interleaving would
fight against that prior commitment. By keeping each cluster as a
contiguous block during merging, we preserve every cluster's local
optimality and only commit to "where does this block go" decisions.

**Why no concatenation tier**: it's just the special case "insert at
the end of base's schedule," which is one valid insertion position
considered by the same loop.

#### Candidate enumeration: by vreg

The natural way to enumerate merge candidates is **by vreg**. For each
vreg `V` in the current cluster graph, the candidate cluster set is:

```
S(V) = {producer's cluster} ∪ {cluster of each consumer of V}
```

This is the **minimal** merge that fully internalizes `V`. For a binary
vreg this is a 2-cluster set; for an N-use vreg it's at most an
(N+1)-cluster set (fewer if some endpoints already share a cluster).

Different vregs may produce the same cluster set (when their endpoints
coincide). De-duplicate by `S` so each cluster set is attempted once.

#### Priority queue main loop

```
function form_clusters(mir):
    // Each instruction starts as its own cluster (singleton schedule)
    for each instruction I:
        clusters[I] = new Cluster(instructions=[I], schedule=[I])

    PQ ← empty min-priority queue keyed by actual_cost

    // Initial population
    for each vreg V in the program:
        S ← S(V)
        if |S| < 2: continue
        result ← try_merge(S)
        if result is Success:
            PQ.insert(result)

    while |clusters| > 1:
        result ← PQ.pop_min()

        // Lazy validity check
        if any cluster in result.cluster_set no longer exists:
            continue                              // stale, skip
        if result was computed against a stale cluster state:
            result ← try_merge(result.cluster_set)
            if result is None:
                // try_merge always succeeds (end-insertion with
                // save-and-rewrite); None means a bug
                assert(false, "try_merge unexpectedly failed")
            // If cost rose, push back and re-pop the queue
            if PQ.peek().cost < result.cost:
                PQ.insert(result)
                continue

        commit(result) → produces merged cluster M

        // Repopulate: for each vreg now touching M, attempt merging
        for each vreg V touching M:
            S ← S(V)
            if |S| < 2 or S already attempted in current state: continue
            attempt ← try_merge(S)
            if attempt is Success:
                PQ.insert(attempt)

    return the single remaining cluster
```

After each commit, we re-attempt merges involving the new merged cluster
`M`. The loop terminates when only one cluster remains. Because every
`try_merge` is guaranteed to succeed (the save-and-rewrite mechanism
makes any insertion position feasible), the queue can never get stuck
— we always find some merge to commit.

#### Why "always merge, cheapest first"

Cluster merging is the only mechanism by which we schedule and allocate
within a basic block. There's no separate inter-cluster phase. The
algorithm runs until exactly one cluster remains, at which point its
internal schedule is the final block schedule.

Cheapest-first ordering matches the natural intuition: 0-copy merges
(tied chains, cheap class transitions) happen first. They commit easy
register choices that don't constrain anything else. Expensive merges
happen last, when most of the structure is already determined and the
algorithm has the most context.

We don't compute regret or savings explicitly. The previous "savings"
formulation was a proxy for regret of deferring a merge. With
always-merge, the question is no longer *whether* to merge but *when*,
and `actual_cost` is the most direct measure of "what does this merge
cost right now."

#### Complexity

Per commit:
- One pop from the queue: `O(log Q)` where `Q` is the queue size.
- Re-attempt M's pairs with neighbors: at most `O(|clusters|)` re-attempts,
  each of cost `attempt_cost ≈ O(|merged_cluster|² × max_class_size)`.

Total commits: at most `|initial_clusters| - 1` (each commit reduces
the cluster count by 1).

Total work: `O(|clusters|² × attempt_cost)`. For typical basic blocks
(~50 instructions, ~50 starting clusters), this is around `2500 ×
attempt_cost ≈` a few million operations per block — comfortable for a
compiler pass.

If benchmarks show this is too slow, cache attempt results and only
invalidate the ones involving committed clusters.

#### Tied-def consumers and the destructive-last rule

A tied-def consumer (e.g., `LSR A` or `ORAImag8 %a, %b` where `%a` is
tied to the output) destroys its input by overwriting the input's
register with the output's value. For the merge to be valid, the
tied-def consumer must run AFTER all other uses of the tied input.

This is encoded as a dependency: when a value `%v` has multiple uses,
the tied-def consumer that destroys `%v` is treated as having a data
dependency on every other use of `%v`. The insertion mechanism respects
this naturally — it can't place a tied-def consumer of `%v` before
another use of `%v`.

If a cluster contains multiple destructive consumers of the same
multi-use value (only one of which can be "the" destructive use), the
merge must insert a save copy and rewrite one of the consumers to read
the saved value instead. This is the same save-and-rewrite mechanism
described above.

#### Top-down slack-based colorability check

Allocate the cluster's schedule by a top-down treescan: process
instructions in order, at each def compute its **slack** against the
already-defined live values, and cascade exclusions if slack falls below 1.

Each live value tracks:
- `class`: its declared register class (immutable)
- `effective`: the registers it can still go to (initially `class`)

The **slack** of a value `v` is computed against the currently live set:

```
slack(v) = |v.effective| - |{u live, u ≠ v, u.effective ∩ v.effective ≠ ∅}|
```

This is the **overlap** count: every other live value whose effective
class shares at least one register with `v`'s effective. Worst case,
each of those overlapping actives could take a register from `v.effective`,
leaving `slack(v)` registers free. If `slack(v) ≥ 1`, `v` is guaranteed
a register no matter how the other choices land.

Why overlap (and not "subset of class")? Because the only legal final
allocation is one where each value occupies a singleton register. The
overlap form is the eager committment to that endpoint: any conflict
with a live value's *current effective* counts. Subset-based Hall's
checking is more permissive at intermediate stages, but every legal
final allocation collapses to singletons anyway, so we might as well
commit early. The cascade is monotonic for descendants — shrinking one
value can only *increase* others' slack, never decrease — which gives
the algorithm a clean inductive shape.

```
function check_colorable(schedule):
    active ← {}
    for each instruction in schedule order:
        // Retire dead values (last use is at this instruction)
        for each operand op consumed by instruction (last use):
            active.remove(op)
        // Add each def
        for each def d of instruction with declared class C:
            d.effective ← C
            active.add(d)
            if not ensure_slack(d, active):
                return Fail
    return Success

function ensure_slack(d, active):
    // Bring d's slack up to ≥ 1 by shrinking other actives
    while slack(d, active) < 1:
        u ← pick u ∈ active such that
              u ≠ d
              and u.effective ∩ d.effective ≠ ∅
              and u.effective \ d.effective ≠ ∅
              (heuristic: choose u with largest |u.effective \ d.effective|)
        if no such u: return False
        u.effective ← u.effective \ d.effective
        // Shrinking u may have left u itself slack-deficient
        if slack(u, active) < 1:
            if not ensure_slack(u, active): return False
    return True
```

#### Why the cascade is monotonic for descendants

When we shrink some `u` by excluding `d.effective` from it, `u.effective`
gets smaller. For any other live value `w`:

- If `w` previously overlapped `u` *only* through registers in
  `d.effective`, that overlap is now empty. `w`'s overlap count drops
  by 1, so `slack(w)` **increases** by 1.
- If `w` overlapped `u` through other registers too, the overlap remains.
  `slack(w)` is unchanged.

`slack(w)` never decreases as a result of shrinking some other value.
This means the cascade can only ever *help* descendants — it never
introduces new violations elsewhere. The recursive `ensure_slack(u)`
call only fires for `u` itself (because `u`'s own size dropped while
its conflict count may have stayed the same).

#### Termination

Each `ensure_slack` call either fails or strictly shrinks some
`effective_class` by at least one register. There are finitely many
elements across all `effective` sets (bounded by `Σ |class(u)|`), so
the cascade terminates.

For a 20-active cluster with classes of size up to 32, the worst-case
total cascade work is bounded by `20 × 32 = 640` shrinks per merge
attempt. In practice, most merges trigger 0–2 shrinks because the
cluster's structure rarely produces deep cascades.

#### Soundness and the "singleton" justification

If `check_colorable` succeeds, every live value has a non-empty
`effective` set and a slack of at least 1 at its def time. The cluster
has a valid coloring, constructed as follows: for each value, in def
order, commit it to any single register from its `effective` set; the
slack guarantee ensures one is always available no matter how earlier
choices were made.

Trivially, the singleton-per-value coloring is a valid endpoint: if the
cluster is colorable at all, then *some* assignment of one register per
value exists. The overlap-slack cascade is committing to such an
assignment incrementally. This gives up some intermediate flexibility
(a more permissive subset-slack check would defer commitments), but
since every valid final coloring is a singleton assignment anyway, the
early commitment costs nothing in the limit.

The algorithm is **not complete**: the choice of which `u` to shrink
during cascade is greedy, and a different choice could in principle
save a future violation. When the greedy cascade fails for a
particular insertion position, the algorithm tries the next position
(or the other base direction). Soundness is what matters — if the
cascade succeeds, the coloring is valid.

### 2.2 The single final cluster

At Phase 2.1's termination, exactly one cluster remains: it contains
every instruction in the basic block, with a fully determined internal
schedule. This single cluster's schedule **is** the final instruction
sequence for the block.

Live-in physregs entered the algorithm via the `K_livein` start
cluster (§1.2) and were absorbed into the final cluster through
ordinary merges. By the time a single cluster remains, every livein
has been internalized.

There are no "boundary" copies in the inter-cluster sense — the only
copies in the final schedule are those inserted by save-and-rewrite
during merge attempts, plus the original COPY instructions from the
input MIR. These are SSA-form COPY instructions that may be coalesced
in Phase 4 if their source and destination land on the same physreg.

## Phase 3: Effect computation (sanity check)

After Phase 2.1 produces the single merged cluster, Phase 3 reads off
its observable behavior at the block boundary:

- **Inputs**: live-in physregs and any vregs defined outside the block
  that this block consumes.
- **Outputs**: any vreg the block defines that is live across its
  boundary (used by successor blocks).
- **Clobbers**: physregs the block writes to without producing a
  live-out value. Includes anything reachable by Cc/Vc side-effects.

For straight-line single-block code (the current scope), this is mostly
informational. It becomes important when extending to multi-block
functions, where the effects of one block influence the scheduling
constraints of its predecessors and successors.

## Phase 4: Coalescing and Concrete Register Assignment

### 4.1 Copy Coalescing

Walk the final schedule. For each `COPY` instruction (whether already
in the input MIR or inserted by Phase 2.1's merges):

- If its source physreg equals its destination physreg (after Phase 4.2's
  assignment): delete the COPY.
- Otherwise: keep it as a real instruction.

This is the same coalescing LLVM does for its existing register
allocators. No MOS-specific logic.

### 4.2 Concrete Register Assignment

Phase 2.1's slack-cascade colorability check leaves each value with an
**effective class** — possibly narrowed from its declared class by
exclusions during cascading. Phase 4 commits each value to a single
concrete physreg from its effective class.

Linear scan over all values in the schedule:
- For each value, find its live range.
- For singleton effective classes (Ac, Cc, Vc, or any value narrowed to
  a single register): the choice is forced.
- For multi-register effective classes: assign the first concrete register
  that is free for the value's entire live range.

With 32 RC registers (the practical Imag8 size) and typical peak usage
of 5–10 simultaneously-live Imag8 values, conflicts in the assignment
step are rare. When they do occur, the slack cascade in Phase 2.1 has
already ruled out the truly infeasible cases.

**Key property**: All cost-relevant decisions happened in Phase 2.1
during the merges. Phase 4 is bookkeeping — it can never make the code
worse than what Phase 2.1 produced.

## Summary

| Phase | Concern | Output |
|-------|---------|--------|
| 1 | Setup | Single `K_livein` start cluster, terminator identification (no copy propagation, no class widening) |
| 2 | Form a single cluster | The block's complete schedule via priority-queue merging |
| 3 | Effect summary | The block's externally-visible inputs/outputs/clobbers |
| 4 | Finalize | Coalesced schedule with concrete physregs |

The key ideas:

1. **Cluster merging is the only mechanism.** Scheduling and register
   allocation both happen as side effects of priority-queue cluster
   merging in Phase 2.1. There is no separate inter-cluster scheduling
   phase, no separate COPY insertion phase. The whole basic block is
   produced by repeated merging until one cluster remains.

2. **Cluster invariant**: every value used within a cluster is defined
   within the cluster. External values are brought in via COPY-in
   instructions whose sources are vregs in other clusters. COPY-ins
   that end up at the same physreg as their source are coalesced away
   in Phase 4.

3. **Insertion is the merge mechanism.** When merging clusters, each
   non-base cluster's schedule is inserted as a contiguous block at
   some position in the base cluster's schedule. The algorithm tries
   both base directions and picks the cheapest. There's no
   "interleave" mode that splits clusters apart; cluster internal
   order is preserved.

4. **Insertion validity and restricted saves.** A position is invalid
   if the inserted block would clobber a value with remaining
   intra-cluster uses (unless it's a coalesced value). Saves are
   only allowed for the coalesced values and for live-out values with
   no remaining intra-cluster uses after the insertion point. Among
   equal-cost positions, **earliest wins** (shortest coalesced-value
   live ranges → fewest future conflicts).

5. **Save-and-rewrite handles register conflicts.** When a qualifying
   value would be clobbered, the merge inserts a save copy before the
   clobber and rewrites the relevant COPY-in source to read the saved
   vreg. The COPY-in is the reload — no separate reload is ever
   inserted. This keeps the mechanism purely cross-cluster.

6. **Cheapest first.** The priority queue orders merges by
   `actual_cost` ascending. Cheap merges (especially 0-copy tied
   chains) commit early and accumulate progress without burning
   resources. Expensive merges happen last, against a state where
   most easy commitments are in place. No `max_cost` or `savings`
   proxy is needed.

7. **Per-use kill flags.** A value is dead after its `killed` use
   (existing MIR semantics). A cluster must keep a value live-out
   unless it contains the value's killed use.

8. **N-ary merges for multi-use vregs.** Multi-use values drive
   (N+1)-way merges of all their endpoint clusters in a single
   decision.

9. **Slack as the colorability invariant.** Each value tracks an
   effective class; the cascade shrinks effective classes to maintain
   slack ≥ 1 at every program point. Sound (non-empty effective ⇒
   valid coloring exists), monotonic for descendants, terminating.
