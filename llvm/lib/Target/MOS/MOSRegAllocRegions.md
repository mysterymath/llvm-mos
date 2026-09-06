# Sieve: a first manual region decomposition

This is an exploration of the SSA input to MOSRegAlloc, not an implemented
allocator or a claim of context-independent optimality. The existing allocator
is an abandoned single-block experiment; it rejects sieve before allocation.
Leave it intact while determining what a replacement should combine.

## Reproduce the input

From the repository root, after building clang and llc:

```sh
~/bin/idledo ninja -C build -j2 clang llc
~/bin/idledo build/bin/mos-sim-clang -Os -S -fno-lto build/sieve.c \
  -o build/sieve-regions.mir -mllvm -stop-before=mos-regalloc \
  -mllvm -verify-machineinstrs -mllvm -debug-pass=Structure \
  2>build/sieve-regions-pipeline.txt
```

The September 6, 2026 capture has 20 blocks, 115 machine instructions including
14 PHIs, and `isSSA: true`. The pipeline reaches MOS Copy Insertion without
running PHI elimination, register coalescing, MachineScheduler, or an allocator.
Earlier SSA optimizations, including loop-invariant code motion, have run.
The numbered values below refer to this capture, not the older `build/sieve.s`.

## Inner-loop graph

In `bb.6.while.body`, name the boundary values as follows:

| Meaning | SSA values, low byte then high byte |
| --- | --- |
| Current k | %178, %179 (PHIs) |
| Prime/step | %264, %265 |
| Next k | %158, %160 |
| Address of flags | %324, %325 (relocatable byte constants) |
| Clear carry | %241 (`LDImm1 0`) |
| Store value, index | %285, %14 (both zero) |

The executable body between the PHIs and terminators splits into:

```text
                          flags base
                              |
    k ------------------> address ----> store zero
    |                     (5 MIs)        (1 MI)
    |
    +------ prime ------> add16 ------> next k ------> compare/branch
                          (4 MIs)           |          (multiple blocks)
                                           +--------> backedge PHIs
```

The two computations share inputs. There is no SSA value edge from the store
to the addition. Reusing k's storage in place creates an additional ordering
requirement; it must not be assumed before choosing placements.

### Address: five instructions

```text
%315 = COPY %178
%170, %171, %314 = ADCImag8 %324, %315, %241
%313 = COPY %179
%172, %173, %312 = ADCImag8 %325, %313, %171
%165 = REG_SEQUENCE %170, sublo, %172, subhi
```

Only %165 escapes this set of definitions. %171 is an internal carry;
%173, %314 and %312 have no uses. %315 and %313 express operand-placement
constraints, not additional computations. The region boundary is two k bytes,
the base address and clear carry in, and a paired pointer out.

This is the first useful contraction candidate: it hides a carry chain and
pointer assembly. A region's implementation can write the low pointer byte
to its final location before computing the high byte, avoiding simultaneous
residency of both results in A. The input MIR order alone does not select that
implementation.

### Address plus store: six instructions

Adding `STIndirIdx %285, %165, %14` hides the pointer value too. Now the region
has a memory effect and no escaping SSA definitions. It still needs a pointer
pair as scratch. Zero escaping results does not mean zero resource demand.

The store's memory operand and ordering constraints must survive contraction.
Unrelated live values also matter: the summary must describe scratch and
preservation, not just the explicit SSA inputs and outputs.

### Update: four instructions

```text
%311 = COPY %264
%158, %159, %310 = ADCImag8 %178, %311, %241
%309 = COPY %265
%160, %161, %308 = ADCImag8 %179, %309, %159
```

Only %158 and %160 escape. Their consumers include both the comparisons and
the PHIs on the two backedges. %159 is internal; the other flags are unused.
This is an add16 candidate whose result placements remain a boundary choice.

### Comparison: a control-flow boundary

The high-byte equality test in bb.6 selects bb.23 or bb.24. Those blocks test
the low or high byte respectively and branch back to bb.6 or exit via bb.25.
Together they implement unsigned `next_k < 8191`.

Contracting just an arithmetic carry chain needs no CFG rewrite. Recovering
a single comparison region here would cross blocks and must preserve edge
identity and PHI uses. Keep this distinct from the first local experiment.

## Concrete implementation contracts

These are feasible recipes for ordinary NMOS 6502 binary arithmetic, not
proven optimal implementations or edits to the generated program. Symbolic
`klo`, `khi`, `plo`, and `phi` denote separate zero-page bytes. `ptr` is a
separate adjacent zero-page pair, disjoint from the inputs and other live
storage. The inputs reside there on entry; acquiring those placements is a
cost outside these recipes. Fixed Ac classes on selected operands do not
imply that the boundary values must live in A throughout a region: internal
copies must bridge the chosen boundary placements to those operands.

For address plus store, with X preserved and A/Y available:

```asm
lda #<flags
clc
adc klo
sta ptr
lda #>flags
adc khi
sta ptr+1
ldy #0
tya
sta (ptr),y
```

This has 10 instructions and 18 bytes. On exit A=0 and Y=0, X is preserved,
NZ reflects zero, and C/V reflect the high-byte address addition. The scratch
pair still contains the address. A smaller recipe applies when Y is already
zero: omit `ldy #0`, giving 9 instructions and 16 bytes while preserving Y.
Thus the same internal computation has a useful boundary-dependent choice.
The D flag must be clear, as assumed by these binary-arithmetic recipes.

For the update, with the old k dead after the region and its zero-page
locations reusable:

```asm
clc
lda klo
adc plo
sta klo
lda khi
adc phi
sta khi
```

This has 7 instructions and 13 bytes. X/Y and the prime bytes are preserved;
A contains the next high byte. Omitting the final `sta khi` produces a
6-instruction, 11-byte alternative with low result in zero page and high
result only in A (the high-byte memory location retains the old value).
Both are useful contracts: the next comparison can use A directly, but the
backedge and next address calculation decide whether storing the high byte
now or later is worthwhile. Neither contract dominates the other just from
these byte counts.

Likewise, the first store recipe produces Y=0 and the update preserves Y.
A loop arrangement could retain that fact across iterations and pay for
initialization only on entry. This depends on the intervening comparisons
and edge transfers also preserving Y. It is a concrete reason to export
known contents, not only clobber sets.

The shared constants stay outside the graph contractions. %241 is defined
in the entry block and has many uses; making it an internal node of one
region would be incorrect. Its known value can instead specialize each
region recipe, allowing a local `CLC` without carrying a stored Boolean.
The same distinction applies to the address constants and the two zeros.

## Other candidates in the same function

| Location | Candidate | Important boundary |
| --- | --- | --- |
| bb.3 | Two ADCs plus REG_SEQUENCE, optionally LDIndirIdx and the zero test | Same address motif as the inner loop, without the two input COPYs; memory read and branch are additional effects |
| bb.4 | ASL/ROL and their operand COPYs | Shift carry is internal; doubled i then feeds two additions |
| bb.8 | ADCImm pairs adding 3 and 2 | Two separate add16-immediate regions; both results feed outer-loop PHIs |
| bb.7 | IncMB | An existing region-like pseudo: internal carry/branching is already hidden, placement remains open |
| bb.8, bb.9 | IncMB with adjacent input/output COPYs | The same semantic increment wrapped in placement constraints |

Start with these small contractions rather than treating an entire basic
block as the unit. In bb.6, nine value-producing/copy instructions can be
described by the address and update motifs; folding the store into the
address motif yields two regions covering ten instructions total (6+4).
The two PHIs and two terminators remain exposed.

## What this experiment establishes, and what it does not

There are repeated, small-interface motifs in actual SSA input, including
ones already represented by pseudos. Their internal carry values and
placement-only copies need not become global allocation decisions.

It does not establish that executing a region contiguously is optimal in
every context. A surrounding operation may exploit an internal availability
window, or consume an early result. The next useful experiment is to retain
the address, store, and update as separate pieces, then compare merging them
under explicit boundary contracts. A successful merge must preserve the
useful placement alternatives; merely finding an allocation with no extra
copies is insufficient.

The old MOSRegAlloc merges clusters until a block is one cluster, using
colorability/copy-count heuristics, without modeling memory dependencies.
Reusing that contraction policy would assume the property we are trying to
investigate. No allocator changes have been made for this manual experiment.

## Survey of overlapping regions

`llvm/utils/mos-region-survey.py` enumerates all nonempty connected,
dependency-convex subsets of selected block interiors. Connectivity includes
shared nonconstant inputs as well as producer/consumer links. PHIs and
terminators are excluded from the candidate sets, but their uses still count
at region boundaries. The full report can be regenerated with:

```sh
~/bin/idledo python3 llvm/utils/mos-region-survey.py build/sieve-regions.mir \
  --block bb.3.for.body6 --block bb.4.if.then \
  --block bb.6.while.body --block bb.8.for.inc14 \
  > build/sieve-region-survey.md
```

This is a structural survey, not an exhaustive search of code implementations.
Its text reader does not model memory dependencies or physical-register
dependencies. Convexity refers only to virtual-register def-use paths. These
regions are not certified atomic scheduling units. Constants are recognized
at direct immediate definitions, without propagation through copies.

| Block interior | Instructions | Connected convex regions |
| --- | ---: | ---: |
| Read flags[i] | 4 | 10 |
| Compute 3*i+3 for the entry test | 10 | 103 |
| Clear flags[k] and update k | 10 | 139 |
| Increment i and update two recurrences | 9 | 26 |

All 278 candidate sets were enumerated. This small sample does not exhibit a
structural enumeration explosion. It says nothing yet about the number of
placement/scheduling implementations each set might require.

### Widening can expose a carry and then hide it again

In bb.4, the six instructions implementing the shift and its COPYs have two
dynamic inputs (i low/high) and two outputs (2*i low/high). Add the next
low-byte ADC: now three values escape, including its carry. Add the matching
high-byte ADC: the carry becomes internal and there are two outputs again.
Adding the final +3 carry chain repeats the same behavior.

| Region size | Computation included | Dynamic inputs + outputs |
| ---: | --- | ---: |
| 6 | 2*i | 4 |
| 7 | Also low byte of 3*i | 5 |
| 8 | All of 3*i | 4 |
| 9 | Also low byte of 3*i+3 | 5 |
| 10 | All of 3*i+3 | 4 |

These are counts of SSA values, not bytes or register pressure. Nevertheless,
they disprove a monotonic boundary-width stopping rule for this example.
Widening must sometimes pass through a less attractive boundary to complete
an internal computation. Consuming an exposed carry is a concrete candidate
for directed widening rather than considering only one-instruction gains.

### Widening through a shared input changes preservation requirements

The six-instruction address/store region has two dynamic inputs, k low/high,
and no value outputs. Both inputs are also used by the update outside it.
The four-instruction update region has four dynamic inputs and two outputs;
the old k bytes are also used by address construction outside it.

Their ten-instruction union still has four dynamic inputs and two outputs,
but the old k bytes have no uses outside the union. The prime bytes do. Thus
the union can consider overwriting k's locations once their last internal
reads complete, without requiring the enclosing problem to preserve old k.

There is no produced value connecting the two smaller regions. They are
connected through the shared inputs. A growth algorithm limited to
producer/consumer edges would miss this union when PHIs are boundary inputs.

The report's general retained-input metric is deliberately conservative: it
counts any use outside a region, including uses that may execute earlier. It
does not claim path-sensitive live-out information. For the inner-loop union,
the absence of any outside uses establishes the particular fact above.

### Resource interaction extends beyond value connectivity

With the shared clear-carry constant treated separately, bb.8 has three value
components: five instructions around IncMB, two ADCs for the +3 recurrence,
and two ADCs for the +2 recurrence. The largest connected region therefore
contains five instructions, even though the block interior contains nine.

The two additions still need A and C; implementation choices for IncMB may
need overlapping resources. Components that are disconnected in this value
graph cannot automatically be optimized independently. A later growth
algorithm may need to unite components when their implementations conflict,
rather than insist that every region stay connected through values.

### Implications for an algorithm

Starting with single-instruction regions need not mean searching instruction
pairs or committing to a hierarchy. The candidate sets can overlap, grow
through shared inputs, and eventually unite separate value components.

These measurements suggest three growth triggers to investigate: close an
exposed internal dependency such as a carry; include the remaining consumers
of an input to discharge its preservation requirement; or include another
computation whose selected implementation conflicts over a machine resource.
The first two are visible in the SSA graph. The third requires information
from actual local implementations. None of these is yet a proof of
quiescence or a complete search algorithm.
