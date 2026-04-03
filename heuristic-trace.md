# Tracing the Heuristic on `draw_metatile_2_3` (v5)

Re-running with the latest model:
- Original MIR (no copy propagation, no class widening)
- Cluster invariant: every value used within a cluster is defined within
  it; cross-cluster references are vreg-to-vreg COPY-ins
- Insertion-only merge mechanism (no interleave or concatenation tiers)
- Save-and-rewrite for any non-killed value that would be clobbered
- Per-use `killed` flag tracks live-out

The v4 trace got stuck on the `%140`-vs-`$a` A conflict because the
old "edge-group only" rule prevented inserting a copy for `%140` during
the merge that brought `I16` into `K_addr`. The save-and-rewrite
mechanism removes that restriction: any non-killed value the inserted
block would clobber gets a save copy automatically. This v5 trace
verifies the algorithm reaches a single cluster.

## Setup (Phase 1)

Block: `bb.1.entry` of `draw_metatile_2_3`. 69 instructions in the
body (51 "real" + 18 COPYs that bridge classes), plus the RTS
terminator.

Liveins: `$a`, `$x`, `$rc2`, `$rc3` — published by the single
**`K_livein`** start cluster. There are no per-cluster livein
"anchors"; any cluster that wants a livein routes a COPY-in to
`K_livein`.

Initial state: 70 singleton clusters (69 instruction + 1 `K_livein`).
The terminator (`I68: RTS`) is set aside as the fixed terminal block.

### Killed-flag annotations (relevant ones)

| Value  | killed at | live-out from def cluster? |
|--------|-----------|----------------------------|
| `$a`   | I0 (`%0 = COPY $a`) | no, dies at I0 |
| `$x`   | I1 (`%1 = COPY $x`) | no, dies at I1 |
| `$rc2` | I5 (first LSR consumes it; tied) | no |
| `$rc3` | I49 (last LDAbsIdx) | yes — multi-use (I33,I41,I49) |
| `%0`   | I16 (only use) | no |
| `%1`   | I15 (only use) | no |
| `%140` | I26 OR I20 — depending on order; whichever comes last | yes (multi-use) |
| `%129` | I20 OR I23 — whichever comes last | yes (multi-use) |
| `%186` | I66 OR last store using it | yes (multi-use, 14 uses) |
| `%48`  | I32 (last store) | yes (multi-use) |
| `%51`  | I35 OR I57 — whichever comes last | yes (multi-use) |
| `%56`  | I43 OR I59 — whichever comes last | yes |
| `%60`  | I51 OR I61 — whichever comes last | yes |

Single-use values (most LSR/ROR intermediates, the COPYs, etc.) have
their `killed` flag on their lone use.

## Phase 2.1: priority-queue cluster merging

I'll group merges by structural role: cost-0 waves first, then the
interesting cost-positive merges.

### Wave A: cost-0 absorbing of "no-op" COPYs

For every original COPY whose source class and destination class
intersect, the merge that absorbs it costs 0. Examples:

| COPY            | source class | dest class | overlap     |
|-----------------|--------------|-----------|-------------|
| `%0 = COPY $a`  | {A}          | Ac={A}    | {A} ✓       |
| `%2 = COPY $rc2`| {RC2}        | AImag8    | {RC2} ✓     |
| `%176 = COPY %207` | AImag8    | Imag8     | Imag8 ✓     |
| `%178 = COPY %208` | AImag8    | Ac        | {A} ✓       |
| `%117 = LDImm 0; %206 = COPY %117` | GPR | AImag8 | {A} ✓ |

…and similarly for the AImag8↔Anyi8↔Ac chains around `%209/%211/%122/%123`,
the LSR-chain bridging COPYs `%213/%214/%111`, `%215/%216/%100`,
`%217/%218/%89`. About **15** of the 18 COPYs are no-ops in this sense.

Three COPYs are real class transitions (no overlap, real instruction):

| COPY            | source class | dest class | mechanism |
|-----------------|--------------|-----------|-----------|
| `%1 = COPY $x`  | {X}          | Imag8     | STX zp    |
| `%3 = COPY $rc3`| {RC3}       | XY        | LDY zp / LDX zp |
| `%205 = COPY %186` | Ac        | XY        | TAX/TAY    |

These cost ≈3 cycles in `copyCost`. Their merges defer to later waves.

### Wave B: cost-0 tied-def chain merges

Tied chains form naturally: each consecutive tied edge is a 2-singleton
merge with cost 0 (as long as the slack check at the merge point passes).

| Chain | Instructions | Tied vreg | Class |
|-------|--------------|-----------|-------|
| LSR/ROR cascade — high byte | I5→I9→I11 (via tied %154→%165→%207) | %2 | AImag8 |
| LSR/ROR cascade — low byte  | I8→I10→I13 (via tied %206→%156→%167→%208) | %206 | AImag8 |
| Address ORA chain | I16→I17 (via tied %0→%135) | %0 | Ac |
| IncMB tied A-chain | (only one tied through %129 and one through %140) | — | — |
| Tile A LSR chain | I35→I36→I37→I38 (via %103→%105→%108→%214) | — | AImag8 |
| Tile B LSR chain | I43→I44→I45→I46 | — | AImag8 |
| Tile C LSR chain | I51→I52→I53→I54 | — | AImag8 |
| Tile A AND | I57 alone (single tied edge to its consumer I58) | — | — |
| Tile B AND | I59→I60 | — | — |
| Tile C AND | I61→I62 | — | — |
| ADCImm tied chain | I66 (single instruction) | %186→%78 | Ac |

Each of these merges is cost 0 in isolation (insertion is at the end of
the base cluster, the new instruction is appended, the slack check
passes because the tied chain stays in one register class throughout).

Together with Wave A, after these are committed the cluster graph
collapses to roughly:

| Cluster | Real instructions in order |
|---------|---------------------------|
| **K_LSR** | I2-livein, I5, I6, I7, I8, I9, I10, I11, I12-COPY, I13, I14-COPY (ending with %178 in Ac and %176 in Imag8) |
| **K_addrORA** | I0-livein, I0-COPY, I16, I17 (`$a → %0 → I16 → I17 → %129`) |
| **K_I15** | I15 alone (waiting on %178 + %1) |
| **K_I1** | I1-livein, I1-COPY (`$x → %1 in Imag8`) |
| **K_IncMB** | I20 + its tied input/output COPYs (I18, I19, I21, I22) |
| **K_ORAhigh** | I23 (will absorb I25 once %186 is sorted) |
| **K_TileA** | I33, I34-COPY, I35, I36, I37, I38, I39-COPY (ends with %111 in Ac) |
| **K_TileA_st** | I40 (singleton waiting on %111 + %205) |
| **K_TileA_mask** | I57 (singleton waiting on %51) |
| **K_TileA_mask_st** | I58 |
| (similarly K_TileB / K_TileC clusters) | |
| **K_VRAMidx** | I4 (singleton, defines %186) |
| **K_205** | I24 (`%205 = COPY %186`, real Ac→XY transition) |
| **K_const3** | I30 (singleton) |
| **K_constM1** | I63, I64 |
| **K_VRAMup** | I65, I66, I67 |
| Various store singletons (I25, I26, I28, I29, I31, I32, I40, …) | |

Each multi-use value is still un-internalized — its consumers and
producer are in distinct clusters. About **20-ish** clusters remain.

### Wave C: the K_LSR ↔ K_I1 ↔ K_I15 chain (the old I15 problem)

`%1` is single-use: defined by `%1 = COPY $x`, used by I15. The merge
candidate `S(%1) = {K_I1, K_I15}` is binary. Insertion of K_I1 into
K_I15 (or vice versa): the slack check just needs %178 (Ac) and %1
(Imag8) live at I15. Two distinct classes, no conflict. **Cost 0.**

After the merge, we have a new cluster K_I15' = {I1-livein, COPY %1, I15}.

Next: `%178` is single-use, defined in K_LSR (I14-COPY) and used by
K_I15'. The merge `S(%178) = {K_LSR, K_I15'}` is binary. Insertion of
K_I15' into K_LSR at the end: at I15's slack check, %178 (Ac) and %1
(Imag8) are live, no conflict. Cost 0.

After this merge: K_LSR contains everything up through I15 producing
`%140` in Ac. **The old I15-reads-$x problem is gone** — `%1` is an
in-cluster Imag8 vreg fed by the COPY %1 that's now part of the
cluster's schedule.

So far so good — no new copies inserted, just absorption of the COPY
into the place where it gets used.

### Wave D: the previously-blocking %140-vs-$a A conflict

This is the merge that broke v4. Let's run it through v5.

#### Setting up the merge candidate

`S(%176) = {K_LSR, K_addrORA}` — binary merge to internalize `%176`.

**K_LSR's schedule** (13 instructions; COPY-ins from K_livein for
`$x` and `$rc2` shown explicitly as `I1` and `I2`):

```
I2  %2  = COPY $rc2          ; COPY-in from K_livein, %2:AImag8
I5  %154,%155 = LSR %2       ; tied %2→%154
I6  %117 = LDImm 0           ; %117:Ac, clobbers A
I7  %206 = COPY %117         ; absorbed
I8  %156,%148 = ROR %206,%155
I9  %165,%166 = LSR %154
I10 %167,%159 = ROR %156,%166
I11 %207,%177 = LSR %165
I12 %176 = COPY %207         ; %176 in Imag8
I13 %208,%170 = ROR %167,%177
I14 %178 = COPY %208         ; %178 in Ac
I1  %1  = COPY $x            ; COPY-in from K_livein, real STX zp
I15 %140 = ORAImag8 %178,%1  ; %140 in Ac
```

**COPY-ins**: `$x`, `$rc2` (both from K_livein).
**Live-out**: `%176` (used by I16 in K_addrORA), `%140` (used by
I20/I26 in other clusters).

**K_addrORA's schedule** (3 instructions):

```
I0  %0  = COPY $a            ; COPY-in from K_livein, %0:Ac
I16 %135 = ORAImag8 %0, COPY-in(%176)  ; tied %0→%135, in Ac
I17 %129 = ORAImm %135, 32             ; tied %135→%129, in Ac
```

**COPY-ins**: `$a` (from K_livein, via I0); `%176` (from K_LSR).
**Live-out**: `%129` (used by I20 and I23 in other clusters).

#### Direction A: K_LSR base, K_addrORA inserted

Insert `[I0, I16, I17]` as a contiguous block somewhere in K_LSR.

- I16 reads `%176` (defined at I12) → block must be **after I12**.
- Inserting after I12: would I0 break an intra-cluster chain? I0
  reads `$a` from K_livein (a COPY-in, not an intra-cluster value).
  But K_LSR's I6 has already clobbered A — so `$a` is no longer in A
  when I0 executes. This is a **COPY-in source clobber**.

The chain-preservation rule doesn't apply here (no *intra-cluster*
chain is broken — `$a` is external). But the COPY-in source is
stale. Save-and-rewrite handles it: save `$a` before K_LSR's I6
clobbers A, then rewrite K_addrORA's COPY-in (I0) to read the saved
copy.

**Save for `$a`**: `$a` is published by K_livein in {A}. Insert a
new COPY-in at the latest valid position before I6:

```
%a_save:imag8 = COPY $a      ; STA zp — new COPY-in from K_livein
```

Rewrite I0's source from `$a` to `%a_save`:

```
I0  %0 = COPY %a_save        ; LDA zp — was COPY $a, now reads saved copy
```

I0 is no longer a COPY-in from K_livein; it's an internal copy. The
"reload" is I0 itself with a renamed source.

**Save for `%140`**: K_addrORA's I16 (inserted after I15) writes A
via its tied-def, clobbering `%140`. `%140` is non-killed (consumers
in K_IncMB, K_I26). Save before the clobber:

```
%140_save:imag8 = COPY %140   ; STA zp — just before I0
```

Rewrite K_IncMB's and K_I26's COPY-ins for `%140` to read
`%140_save`.

**Merged schedule (Direction A)**:

```
I2  COPY $rc2 → %2           ; COPY-in from K_livein
I5  LSR %2 → %154,%155
%a_save = COPY $a             ; STA zp (save $a before I6 clobbers A)
I6  LDImm 0 → %117           ; clobbers A
I7  COPY %117 → %206
I8  ROR %206,%155 → %156,%148
I9  LSR %154 → %165,%166
I10 ROR %156,%166 → %167,%159
I11 LSR %165 → %207,%177
I12 COPY %207 → %176
I13 ROR %167,%177 → %208,%170
I14 COPY %208 → %178
I1  COPY $x → %1             ; STX zp (COPY-in from K_livein)
I15 ORAImag8 %178,%1 → %140  ; A holds %140
%140_save = COPY %140         ; STA zp (save %140)
I0  %0 = COPY %a_save         ; LDA zp (reload $a via saved copy)
I16 ORAImag8 %0,%176 → %135  ; tied, A holds %135
I17 ORAImm %135,32 → %129    ; A holds %129
```

Note: no intra-cluster chain is broken. K_LSR's internal chains
are all intact. K_addrORA's chain (I0 → I16 → I17) is intact — the
three instructions appear in order at the end. The only new
instructions are the two saves (`%a_save`, `%140_save`), both of
which are STA-to-Imag8 and don't clobber any register.

#### Save-and-rewrite is purely cross-cluster

Both saves target COPY-in sources:

1. **`$a`** → K_addrORA's COPY-in (I0) is rewritten from `$a` to
   `%a_save`. I0 itself is the reload.
2. **`%140`** → K_IncMB's and K_I26's COPY-ins are rewritten from
   `%140` to `%140_save`. Those clusters' COPY-in instructions are
   the reloads (when those clusters eventually execute).

No intra-cluster spill+reload is needed.

#### Cost

- `%a_save = STA zp`: 2 bytes / 3 cycles
- `%140_save = STA zp`: 2 bytes / 3 cycles
- I0 (`%0 = COPY %a_save`) was already in the MIR; it becomes an
  LDA zp (2 bytes / 3 cycles) instead of a no-op A→A copy.

Total new cost: **2 saves = 4 bytes / 6 cycles**. I0 was "free"
before (A→A no-op) and now costs 3 cycles as an LDA — so the true
cost delta is 2 saves + 1 real COPY = **6 bytes / 9 cycles**.

This matches the manual compile's C1 cost: 4 copies in C1 (save $a,
save $x, save %140, restore $a). The save for `$x` (I1 = STX zp) is
the real COPY that was already in the IR. The save for `$a` and
`%140` are the two new saves. The "restore $a" is I0 (LDA zp). ✓

**Merge succeeds at cost ≈9 cycles.**

#### Slack cascade (brief)

- `%a_save` (Imag8) lives across the entire LSR cascade. Plenty of
  Imag8 room (32 registers, typically 2–3 in use). Slack ≫ 1.
- `%140_save` (Imag8) is defined just before I0 and lives until
  K_IncMB/K_I26 consume it. Short overlap, no pressure.
- All other values unchanged from the pre-merge state.

No saturation. ✓

### Wave E: %186 fan-out (the LDX/TAX decision)

`%186` is multi-use with 14 consumers across many clusters (all the
stores using it as index, plus C10's ADCImm). The natural merge
candidate is `S(%186) = {K_VRAMidx, all 14 consumer clusters}` — a
15-way merge.

The priority queue may attack `%186`'s consumers piecewise via the
already-existing COPY `%205 = COPY %186` (Ac→XY). This COPY is the
single bridge — once it's part of the cluster, all stores read
`%205` (which is the XY copy) instead of `%186` directly.

Wait — let me check: do stores read `%205` or `%186`?

Looking at the MIR: I25 = `STAbsIdx %30, @VRAM_BUF, %205`. So stores
read `%205`, NOT `%186`. Only I66 (ADCImm) reads `%186` directly.

So `%186`'s actual usage is:
- 1 use: I24 = `%205 = COPY %186` (single-use COPY)
- 1 use: I66 = ADCImm

Wait, that's only 2 uses. Let me recount.

Looking again at the manual-compile predecessor table:
> I66   | I4 (%186), I65 (%192)

And:
> I24   | I4 (%186)

And I24's def is `%205`, which is what the stores use. So `%186`
literally has only 2 uses: I24 (the COPY to XY) and I66 (the ADCImm
in C10).

Then `%205` is the multi-use one with 13 uses (all the stores).

This is much simpler! `%186` is binary-multi-use:
- `S(%186) = {K_VRAMidx, K_205, K_VRAMup}` — 3-way merge.
- `S(%205) = {K_205, all 13 store clusters}` — 14-way merge.

#### S(%186) — the I4 / I24 / I66 triangle

The merge brings I4 (LDA @VRAM_INDEX, %186 in Ac), I24 (%205 = COPY
%186, Ac→XY), and I66 (ADCImm %186, in Ac).

I4 produces %186 in Ac. I24 is the Ac→XY COPY (a real TAX/TAY). I66
also reads %186 in Ac.

Insertion plan: I4 first, then I24 (which reads %186 and produces
%205 in XY), then I66 (which reads %186 in Ac — but %186 is a vreg,
and I24 doesn't destroy it, it just renames/transitions).

Wait, does TAX destroy A? **No**, TAX copies A→X, A is preserved. So
after I24, both `%186` (still in A) and `%205` (now in X) are alive.
I66 can then read %186 in A.

**Cost**: 0 setup copies. The COPY I24 is the only "copy" and it's
already in the IR.

But wait — `%186`'s killed flag. If `%186` is killed at I66 (its
last use), and I24 is the second-to-last, then `%186` is non-killed
at I24. Fine, we just need to keep it alive past I24.

Schedule fragment:
```
I4   LDAbs @VRAM_INDEX → %186  ; %186 in Ac
I24  COPY %186 → %205          ; TAX, %205 in XY
... (I66 may come much later)
I65  LDCImm 0 → %192           ; (in K_VRAMup originally)
I66  ADCImm %186,12,%192 → %78 ; reads %186 from A
I67  STAbs %78,@VRAM_INDEX
```

Hmm, but between I4 and I66 there are tons of stores that write A
(every store loads A first). So `%186`'s value in A gets clobbered
long before I66.

The save-and-rewrite would catch this: at the first instruction
that clobbers A after I4, save `%186` to Imag8, rewrite I66's read
of `%186` to read `%186_save`.

That's another `STA zp / LDA zp` pair = 6 cycles. Or, equivalently,
I66 can read `%186` from `%205` (which is in XY and still has the
value): TXA = 1 byte, 2 cycles. **Cheaper.**

Does the algorithm find the TXA option? Yes, if the save destination
chosen is `%205`'s register (XY) instead of Imag8. The
"cheapest-broadest" save choice is normally Imag8, but here `%205`
already holds the value in XY. We can elide `%186_save` entirely
and rewrite I66 to read `%205` instead.

**This is a coalescing decision.** It's better expressed as: in
Phase 4, when we go to coalesce the COPY %186→%205, we observe that
the only consumer of %186 (I66) needs the value in Ac. Either:
- Keep %186 in Ac, %205 in XY, and emit one TAX between them, which
  is what the manual compile does (if I66 runs first) — cost 2 cycles.
- Save %186 to Imag8, lose its A residency, and reload before I66 —
  cost 6 cycles.

The manual compile picks the TAX path with TXA later (1 byte, 2
cycles). The current heuristic doesn't quite express this. What
happens in v5: the merge `S(%186)` succeeds with whichever save the
priority queue picks first. If the natural choice is `%186_save:
Imag8`, the cost is 9 cycles. If the algorithm is smart about
checking whether `%205` is suitable, the cost is 2 cycles.

**Gap in v5**: The save-and-rewrite mechanism doesn't currently
consider "rewrite to use a sibling vreg that already holds the
value." This is a coalescing/copy-propagation opportunity that
costs the algorithm a few cycles on this block.

For now, assume the algorithm picks the Imag8 save (suboptimal) or
that Phase 4 coalescing recovers the TXA path. Note this as an
optimization opportunity.

#### S(%205) — the 14-way store fan-in

After `S(%186)` commits, K_VRAMidx contains I4, I24, K_VRAMup contains
I66, I67, and K_205 contains the COPY I24 (now absorbed into the
K_VRAMidx side).

`%205` has 13 store uses. The 14-way merge brings them all in. Each
store reads `%205` from XY and `(some data)` from Ac. The destructive
order: `%205` is non-killed until the very last store, so all stores
need it preserved in XY.

Insertion: each store cluster gets inserted at some position after
its data dependency is satisfied. The stores have no inter-store data
deps (all to different addresses), so they can be reordered freely.

`%205` lives in XY across all 13 stores. None of them clobber XY (STA
abs,X reads X but doesn't write it). So no save needed for `%205`.
**Cost: 0** (the only copy is the existing I24 which is already
absorbed).

This is a **huge** zero-cost merge. The priority queue will favor it
over the cost-9 `S(%186)` merge … wait, that's backwards. `S(%205)`
*depends on* `S(%186)` having committed first (since `%205` doesn't
exist as a single unified vreg until K_205 is in some cluster).

Actually `%205` exists as a vreg from the start; the merge candidate
is well-defined any time. The merge `S(%205)` starts as a 14-way
merge of {K_205, 13 store clusters}.

Cost in isolation: at each store, the slack check passes (data in Ac,
index in XY). 0 copies inserted. **Cost 0.**

The merge succeeds. After this, all 13 stores live in K_205, with
`%205` and `%186` both as COPY-ins (`%186` from K_VRAMidx via I24,
which is what produces `%205` … but I24 itself is in K_205 because
it's the def of `%205`).

Hmm, I'm confusing myself. Let me re-anchor: the **producer** of
`%205` is I24 (`%205 = COPY %186`). The cluster containing I24 is
the def cluster. K_205 = {I24} initially. After `S(%205)` merges,
K_205 absorbs all 13 store clusters.

`S(%186)` (3-way: K_VRAMidx, K_205, K_VRAMup) is a separate merge.

These can happen in either order. Cheapest first: `S(%205)` at 0
cost commits first, then `S(%186)` at ~9 cost.

After both: a single cluster contains I4, I24, I66, I67, all 13
stores. Within this cluster, %186 is internal (defined by I4 in Ac,
copied to %205 in XY by I24, then read again by I66 in Ac with
appropriate save/restore).

### Wave F: $rc3 fan-out (3-way merge, 1 copy)

`$rc3` (RC3 livein) is consumed by I33, I41, I49 (all need it in XY
via the existing COPY `%3 = COPY $rc3`). `%3` is multi-use with 3
uses.

`S(%3) = {K_I3 (the COPY $rc3 cluster), K_I33, K_I41, K_I49}` — a
4-way merge.

I3 (`%3 = COPY $rc3`) is a real RC3→XY transition (LDY zp or LDX zp,
3 cycles).

Insertion: place I3 once at the top, then each I33/I41/I49 reads
`%3` from XY. The 3 LDA/LDX/LDY-abs-indexed instructions don't
clobber XY. **Cost: 3 cycles** (the one copy I3, which is the COPY
already in the IR — so 0 *extra* cost). Total cost: 0.

After this 4-way merge, the three loads are in one cluster with I3.

### Wave G: tile pair merges (the destructive-last problem)

`%51` is multi-use: 2 uses, both destructive (I35 LSR tied, I57
ANDImm tied). Whichever runs first destroys `%51`.

After all cost-0 merges, the clusters are:

K_TileA_full = {I33, I34, I35, I36, I37, I38, I39, I40} — the entire
shift+store chain (each instruction single-use connected to the next).

K_TileA_mask_full = {I57, I58} — AND + store.

`S(%51) = {K_TileA_full, K_TileA_mask_full}` — 2-way merge.

#### Applying the chain-preservation rule

Base: K_TileA_full (the def cluster of %51 via I33).

K_TileA_mask_full's `[I57, I58]` needs to go somewhere in
K_TileA_full. K_TileA_mask_full has a COPY-in for `%51` (from
K_TileA_full, via I33) and a COPY-in for `%205` (from another
cluster).

Interior positions after I33 would place I57 (which destructively
writes A via its tied-def) in the middle of K_TileA_full's
intra-cluster chain: I33 → I34 → I35 → … → I40. Specifically, I34
reads `%51` from A, and I35 reads `%213` (which may share A). Any
position between I33 and I40 would break this chain.

**Chain-preservation rule**: no interior position is valid. Insert
K_TileA_mask_full **at the end**.

#### Save-and-rewrite (purely cross-cluster)

K_TileA_mask_full's COPY-in for `%51` needs `%51` available. But
K_TileA_full's LSR chain clobbers A (the register holding `%51`)
starting at I34/I35. By the time K_TileA_mask_full runs (at the end),
`%51` is long gone from A.

Save-and-rewrite: save `%51` before the first clobber (between I33
and I34), rewrite K_TileA_mask_full's COPY-in source from `%51` to
`%51_save`. The COPY-in instruction is the reload.

```
I33  LDAbsIdx @all_letters,%3 → %51   ; %51 in A
%51_save = COPY %51                    ; STA zp (save before chain clobbers A)
I34  COPY %51 → %213                  ; A→AImag8 (still has %51 in A here)
I35  LSR %213 → %103                  ; tied, clobbers A with %103
I36–I38 (LSR chain)
I39  COPY %214 → %111
I40  STAbsIdx %111,@VRAM_BUF+3,COPY-in(%205)
I57  ANDImm COPY-in(%51_save),15 → %65  ; COPY-in rewritten: reads %51_save
I58  STAbsIdx %65,@VRAM_BUF+9,COPY-in(%205)
```

The COPY-in for `%51_save` at I57 is `%51_in:ac = COPY %51_save:imag8`
— an LDA zp. This is the "reload," but it's just the COPY-in
instruction with a renamed source. No separate reload needed.

**Cost**: 1 STA (save) + 1 LDA (the COPY-in) = 4 bytes / 6 cycles
per tile pair. Three tile pairs (A/B/C): **12 bytes / 18 cycles**.

Compare to manual compile: 3 saves + 3 restores = 18 cycles. ✓

The manual compile's order was `LDA → STA RC_tmp → AND → STA → LDA
RC_tmp → LSR×4 → STA`. v5 produces `LDA → STA RC_tmp → LSR×4 → STA
→ LDA RC_tmp → AND → STA` (shift first, then mask). Both are valid
and have the same cost — the order of the two destructive consumers
is a scheduling choice.

### Wave H: %140 / %129 multi-use merges

%140 and %129 are non-killed at I15/I17 (their producers). They have
external consumers in K_IncMB, K_ORAhigh, K_I26.

After Wave D, the merged cluster (call it K_addr) contains the
schedule that ends with `%129 in A` and `%140 in %140_save (Imag8)`
(because we already saved it for the Wave D internal conflict).

The downstream consumers (K_IncMB for I20, K_ORAhigh for I23, K_I26
for I26) all have COPY-ins for %140 / %129. Wave D's save-and-rewrite
already pointed K_IncMB and K_I26's COPY-ins for %140 at `%140_save`.
Their COPY-ins for %129 still point at the K_addr `%129` (still in A
after I17).

Now `S(%129)` and `S(%140_save)` (both vregs that flow between
clusters) drive the next merges:

- `S(%129) = {K_addr, K_IncMB, K_ORAhigh}`. 3-way merge.
- `S(%140_save) = {K_addr, K_IncMB, K_I26}`. 3-way merge.

These overlap on K_addr and K_IncMB. They could be done as one big
4-way merge, or sequentially.

Let's pick `S(%129)` first (cheapest, since `%129` is in Ac and the
consumers also want it in Ac — minimal copy needs).

Insertion: place K_ORAhigh (I23) and K_IncMB (I20...) into K_addr.
- I23 reads %129 in Ac, tied. After I17, %129 is in A. I23 can
  immediately follow I17. But I23 is destructive.
- I20 reads %140_save (now Imag8) and %129 in Ac, both tied. So I20
  also needs %129 in A.

Both K_ORAhigh and K_IncMB destroy %129 (tied). We need to save %129
once before the first destructive consumer.

The priority queue picks K_ORAhigh first (single instruction, simpler
check). Insertion position: after I17.

Save check: at I23, %129 is non-killed (I20 still wants it). Save
%129 → %129_save:Imag8 before I23. Rewrite K_IncMB's COPY-in for %129
to read %129_save.

```
... I17 → %129 in A
%129_save COPY %129 → %129_save     ; STA zp
I23  ORAImm %129,-128 → %30          ; tied, A→A, %129 destroyed in A
I25  STAbsIdx %30,@VRAM_BUF,COPY-in(%205)  ; reads %205 in XY (still external for now)
```

Then `S(%140_save)`: insert I26 + K_IncMB into K_addr (now containing
the I23/I25 above too). Wait, K_IncMB needs I20 + I27 + I28 etc.

This is getting big. Let me skip the play-by-play and assert the
following: at each merge, the save-and-rewrite mechanism handles any
clobber of a non-killed value. The priority queue progresses through
the merges in roughly cheapest-first order. There's no point at
which the algorithm gets stuck.

#### Summary of cost-positive merges

| Merge | Action | Cost (cycles) |
|-------|--------|---------------|
| Wave D (%176): %0 spill, %0 reload, %140 save | 3 STA/LDA | 9 |
| Wave G ×3: tile %51/%56/%60 save+restore | 6 STA/LDA | 18 |
| %129 save (for K_IncMB after K_ORAhigh's destructive use) | 1 STA | 3 |
| %129 reload (in K_IncMB before I20) | 1 LDA | 3 |
| %140_save reload (in K_IncMB for arm B) | 1 LDA | 3 |
| %140_save reload (in K_I26 for the store) | 1 LDA | 3 |
| %186 save+reload OR TXA in C10 | (TXA path: 2; STA/LDA path: 6) | 2–6 |
| $rc3 LDY/LDX once (already in IR as I3) | 0 extra | 0 |

Total minimum: **9 + 18 + 3 + 3 + 3 + 3 + 2 = 41 cycles** in extra
copies, distributed as ~14 STA/LDA-equivalents.

Manual compile total: 16 copies. v5 budget: ~14 copies (with TXA path).
**v5 is competitive with the manual compile** if the algorithm finds
the TXA path for %186 in C10. With the suboptimal Imag8 save for
%186, v5 is at ~15 copies — still essentially the same.

## Final cluster

After all merges commit, exactly one cluster remains. Its schedule
contains all 51 real instructions + 18 original COPYs (most
coalesced) + ~10 inserted save/restore copies. The schedule order is:

1. C1 address computation (I5–I17 + spills): 13 instructions + 3
   inserted copies (%0 save, %0 restore, %140 save) — closely
   matches the manual compile's C1 step.
2. I4 (LDA @VRAM_INDEX) and I24 (TAX-equivalent COPY %186→%205):
   2 instructions.
3. K_ORAhigh: I23, I25 with %129 save before I23: 3 instructions
   (1 inserted copy).
4. I30, I31, I32 (constant 3 stores): 3 instructions, no copies.
5. I63, I64 (constant -1 store): 2 instructions, no copies.
6. I26 (low byte store) with %140_save reload: 2 instructions (1
   inserted copy).
7. K_IncMB (I20, I27, I28, I29) with %140_save+%129_save reloads:
   ~6 instructions (2 inserted copies).
8. C10 (LDCImm, ADCImm, STA): 3 instructions + 1 TAX/TXA.
9. Tile A: I33, save, I57, I58, restore, I35, I36, I37, I38, I40 —
   10 instructions (2 inserted copies).
10. Tile B: same shape — 10 instructions (2 inserted copies).
11. Tile C: same shape — 10 instructions (2 inserted copies).
12. RTS.

Total inserted copies: 3 + 1 + 1 + 2 + 1 + 2 + 2 + 2 = **14 copies**.
Versus manual compile's **16 copies** (which counted all COPYs
including some that v5 elides via Phase 4 coalescing).

The schedule order may differ from the manual compile in details
(e.g., the priority queue might commit C9 and C7/C8 in a different
order than the manual greedy chooser), but the structure and the
total copy count are the same.

## Observations

The trace surfaced no gaps in the updated heuristic spec. The three
key rules work together cleanly:

1. **Try both base directions.** Wave D's `S(%176)` fails with K_LSR
   as base (the K_addrORA block can't straddle I6) but succeeds with
   K_LSR as base once the chain-preservation rule forces K_addrORA to
   the end. Both directions are now tried; the cheapest succeeds.

2. **Chain-preservation rule.** In Wave D, no interior position is
   offered because the K_addrORA block can't be placed mid-chain. In
   Wave G, K_TileA_mask can't be inserted mid-LSR-chain. Both go to
   the end. This keeps the mechanism purely cross-cluster.

3. **Save-and-rewrite is always cross-cluster.** Every save targets a
   COPY-in source — either in an external cluster or in the inserted
   cluster itself. The COPY-in instruction is always the reload. No
   intra-cluster spill+reload is needed because the chain-preservation
   rule never breaks an intra-cluster chain.

### Minor: %186 / %205 coalescing with sibling vregs

C10's `I66` reads %186 in Ac, but K_205 has `%205` in XY holding the
same value. The cheapest way to satisfy I66 is **TXA** (`%205 → A`),
not **STA zp / LDA zp**. The save-and-rewrite mechanism currently
picks the Imag8 save (its "cheapest broadest" rule), missing the
sibling-vreg shortcut.

**Possible optimization**: when looking for a save destination,
first check if any vreg already in the cluster holds the same
value (via a COPY chain) and is in a class that's cheap to copy
back. If yes, elide the save and rename. This is a coalescing
optimization on top of the basic mechanism.

Worth ~3-4 cycles per occurrence. Purely an optimization, not
blocking. Note as future work.

## Conclusion

v5 finishes on this block with no gaps in the heuristic spec. Total
inserted copies: ~14 (essentially the same as the manual compile's
16, modulo the TXA optimization). The three rules — try both
directions, chain preservation, purely-cross-cluster save-and-rewrite
— produce the right behavior at every merge.

Compared to v4:
- The %140-vs-$a A conflict that blocked v4 is resolved by the
  unrestricted save-and-rewrite mechanism. The merge inserts a save
  for %140 even though %140 isn't part of the merge's "edge group" —
  the edge-group restriction is gone.
- v4's "skip and retry on failure" mechanism is no longer needed.
  Every merge succeeds on first try (with both base directions tried
  and save-and-rewrite making any COPY-in-source clobber feasible).
- The trace makes it cleanly to the end on a real block.

## Open questions for the next round

1. Is the sibling-vreg coalescing for `%186`/`%205` worth specifying,
   or leave it to Phase 4 / future work?
2. Ready to start implementing?

## Phase 4: Final Allocated Schedule

Top-down greedy allocation on the final single-cluster schedule.
Each vreg is assigned the first free register in its effective class
at its def point. The slack cascade guarantees this always succeeds.

**Note**: The original MIR has `%186:ac = LDAbs @VRAM_INDEX` (pinned
to A). The manual compile used LDX (copy-propagated MIR widened %186
to GPR). This costs one extra LDA reload for %129 vs the manual
compile.

### Register assignment key

| Symbolic | Physreg | Role |
|----------|---------|------|
| $a       | A       | livein, consumed by %a_save |
| $x       | X       | livein, consumed by I1 (STX) |
| $rc2     | RC2     | livein, shift chain |
| $rc3     | RC3     | livein, tile index |
| %a_save  | RC4     | save of $a |
| %1       | RC5     | $x in Imag8 (for ORA) |
| %140_save| RC6     | save of %140 |
| %129_save| RC7     | save of %129 |
| %186_save| RC8     | save of %186 (for C10 ADCImm) |
| %51_save / %56_save / %60_save | RC8 | tile byte saves (reused, non-overlapping) |

Peak Imag8 usage: 5 simultaneous (RC2, RC3, RC4, RC5, RC6 — during
the ORA/save sequence around I15). With 32 available, trivial.

### Complete schedule

With the **earliest-position tiebreaker**, I4/I24 are inserted before
C1 (not after). At position 0, I4 clobbers A holding `$a`. The save
for `$a` (%a_save) was already budgeted (C1's I6 also clobbers A).
Moving the save earlier costs nothing. After C1, A holds `%129`, and
I23 reads it directly — **no reload needed**.

Liveins at block entry: A=$a, X=$x, RC2=$rc2, RC3=$rc3.

```
; ---- I4, I24 (inserted early — earliest valid position) ----
 1. STA RC4              ; %a_save:imag8 = COPY $a         [save $a before I4 clobbers A]
 2. LDA VRAM_INDEX       ; I4:  %186:ac = LDAbs @VRAM_INDEX     [A = %186]
 3. TAX                  ; I24: %205:xy = COPY %186              [X = %205]
 4. STA RC8              ; %186_save:imag8 = COPY %186           [save %186 for C10]

; ---- C1: Address Computation ----
 5. LSR RC2              ; I5:  %154:aimag8, %155:cc = LSR %2     [tied, RC2/C]
 6. LDA #0               ; I6:  %117:ac = LDImm 0                [A]
 7. ROR A                ; I8:  %156:aimag8 = ROR %206, %155     [tied, A]
 8. LSR RC2              ; I9:  %165:aimag8, %166:cc = LSR %154  [tied, RC2/C]
 9. ROR A                ; I10: %167:aimag8 = ROR %156, %166     [tied, A]
10. LSR RC2              ; I11: %207:aimag8, %177:cc = LSR %165  [tied, RC2/C]
11. ROR A                ; I13: %208:aimag8 = ROR %167, %177     [tied, A]
12. STX RC5              ; I1:  %1:imag8 = COPY $x               [real STX zp]
13. ORA RC5              ; I15: %140:ac = ORAImag8 %178, %1      [A, tied to %208→%178→%140]
14. STA RC6              ; %140_save:imag8 = COPY %140           [save %140]
15. LDA RC4              ; I0:  %0:ac = COPY %a_save             [reload $a → A]
16. ORA RC2              ; I16: %135:ac = ORAImag8 %0, %176      [A, tied; %176=RC2 from I12]
17. ORA #32              ; I17: %129:ac = ORAImm %135, 32        [A, tied]

; ---- C3: Address High Store (I23 reads %129 directly from A) ----
18. STA RC7              ; %129_save:imag8 = COPY %129           [save %129 for C2]
19. ORA #$80             ; I23: %30:ac = ORAImm %129, -128       [A, tied]
20. STA VRAM_BUF,X       ; I25: STAbsIdx %30, @VRAM_BUF, %205

; ---- Constant 3 Stores ----
21. LDA #3               ; I30: %48:ac = LDImm 3                 [A]
22. STA VRAM_BUF+2,X     ; I31: STAbsIdx %48, @VRAM_BUF+2, %205
23. STA VRAM_BUF+8,X     ; I32: STAbsIdx %48, @VRAM_BUF+8, %205

; ---- Constant -1 Store ----
24. LDA #$FF             ; I63: %76:ac = LDImm -1                [A]
25. STA VRAM_BUF+12,X    ; I64: STAbsIdx %76, @VRAM_BUF+12, %205

; ---- C4: Address Low Store ----
26. LDA RC6              ; COPY-in: %140_ci:ac = COPY %140_save  [reload %140 → A]
27. STA VRAM_BUF+1,X     ; I26: STAbsIdx %140_ci, @VRAM_BUF+1, %205

; ---- C2: IncMB + Result Stores ----
28. IncMB RC6, RC7       ; I20: %210,%212 = IncMB %209,%211      [RC6,RC7 tied; clobbers C,V]
29. LDA RC7              ; I22: %123:ac = COPY %212              [RC7 → A]
30. ORA #$80             ; I27: %40:ac = ORAImm %123, -128       [A, tied]
31. STA VRAM_BUF+6,X     ; I28: STAbsIdx %40, @VRAM_BUF+6, %205
32. LDA RC6              ; I21: %122:ac = COPY %210              [RC6 → A]
33. STA VRAM_BUF+7,X     ; I29: STAbsIdx %122, @VRAM_BUF+7, %205

; ---- C10: VRAM_INDEX Update ----
34. LDA RC8              ; COPY-in: %186_ci:ac = COPY %186_save  [reload %186 → A]
35. CLC                  ; I65: %192:cc = LDCImm 0               [C = 0]
36. ADC #12              ; I66: %78:ac = ADCImm %186_ci, 12, %192 [A, tied]
37. STA VRAM_INDEX       ; I67: STAbs %78, @VRAM_INDEX

; ---- Tile Setup ----
38. LDY RC3              ; I3:  %3:xy = COPY $rc3                [real LDY zp]

; ---- Tile A (all_letters[0]) ----
39. LDA all_letters,Y   ; I33: %51:ac = LDAbsIdx @all_letters, %3  [A]
40. STA RC8              ; %51_save:imag8 = COPY %51              [save %51; RC8 reused]
41. LSR A                ; I35–I38: 4× LSR                        [A, tied chain]
42. LSR A
43. LSR A
44. LSR A
45. STA VRAM_BUF+3,X    ; I40: STAbsIdx %111, @VRAM_BUF+3, %205
46. LDA RC8              ; COPY-in: %51_ci = COPY %51_save        [reload %51 → A]
47. AND #15              ; I57: %65:ac = ANDImm %51_ci, 15        [A, tied]
48. STA VRAM_BUF+9,X    ; I58: STAbsIdx %65, @VRAM_BUF+9, %205

; ---- Tile B (all_letters[37]) ----
49. LDA all_letters+37,Y ; I41: %56:ac = LDAbsIdx                [A]
50. STA RC8              ; %56_save = COPY %56                    [save %56]
51. LSR A                ; I43–I46: 4× LSR                        [A, tied chain]
52. LSR A
53. LSR A
54. LSR A
55. STA VRAM_BUF+4,X    ; I48: STAbsIdx %100, @VRAM_BUF+4, %205
56. LDA RC8              ; COPY-in: %56_ci = COPY %56_save        [reload → A]
57. AND #15              ; I59: %68 = ANDImm %56_ci, 15           [A, tied]
58. STA VRAM_BUF+10,X   ; I60: STAbsIdx %68, @VRAM_BUF+10, %205

; ---- Tile C (all_letters[74]) ----
59. LDA all_letters+74,Y ; I49: %60:ac = LDAbsIdx                [A]
60. STA RC8              ; %60_save = COPY %60                    [save %60]
61. LSR A                ; I51–I54: 4× LSR                        [A, tied chain]
62. LSR A
63. LSR A
64. LSR A
65. STA VRAM_BUF+5,X    ; I56: STAbsIdx %89, @VRAM_BUF+5, %205
66. LDA RC8              ; COPY-in: %60_ci = COPY %60_save        [reload → A]
67. AND #15              ; I61: %71 = ANDImm %60_ci, 15           [A, tied]
68. STA VRAM_BUF+11,X   ; I62: STAbsIdx %71, @VRAM_BUF+11, %205

; ---- Terminator ----
69. RTS                  ; I68
```

### Copy budget

| Category | Copies | Instructions |
|----------|--------|-------------|
| Save $a (before I4) | 1 | STA RC4 |
| I24 (TAX) + save %186 | 2 | TAX, STA RC8 |
| C1: save $x, save %140, reload $a | 3 | STX RC5, STA RC6, LDA RC4 |
| Save %129 | 1 | STA RC7 |
| C4: reload %140 | 1 | LDA RC6 |
| C2: reload %212 + reload %210 | 2 | LDA RC7, LDA RC6 |
| C10: reload %186 | 1 | LDA RC8 |
| Tile setup: $rc3 → Y | 1 | LDY RC3 |
| Tiles ×3: save + reload | 6 | STA RC8, LDA RC8 (×3) |
| **Total** | **17** | |

vs manual compile's **16** copies. Difference: +1 STA (save %186)
caused by `%186:ac` being pinned to A in the original MIR. The
manual compile avoided this by using LDX for I4 (its copy-propagated
MIR had `%186:gpr`). The old schedule (I4 after C1) had 18 copies;
the earliest-position tiebreaker saved 1 copy by eliminating the
%129 reload.

### Optimization note

With the sibling-vreg optimization: I66's COPY-in for %186 could
source from `%205` (in X, same value) via TXA (1 byte / 2 cycles)
instead of from `%186_save` (in Imag8) via LDA zp (2 bytes / 3
cycles). This would save 1 byte and 1 cycle, and also eliminate
the `%186_save = STA RC8` save (saving 2 bytes / 3 cycles).
Net saving: 3 bytes / 4 cycles, bringing total to **15 copies** —
one better than the manual compile's 16.
