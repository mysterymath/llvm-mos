# Mid-Term Roadmap: CP Register Allocator for MOS 6502

## Goal

Replace the current greedy register allocator for the MOS 6502 with a
Unison-style constraint programming allocator using Chuffed, capable of
producing optimal or near-optimal register allocation and instruction
scheduling for real programs (target: sieve.c benchmark).

Reference: Castañeda Lozano et al., "Combinatorial Register Allocation
and Instruction Scheduling", TOPLAS 2019 (arXiv:1804.02452).

## Current State

- **M1 complete**: Basic CP register allocator in MOSRegAlloc.cpp.
  Assigns physical registers to vregs using Chuffed with pairwise !=
  interference constraints. LiveVariables for dead/kill flags.
  5 lit tests passing.
- **Chuffed integrated**: Vendored in third-party/, builds with LLVM.
- **Custom diffn propagator**: Written and tested standalone (0.38s for
  35-op block with LCG explanations). Not yet integrated into the pass.

## Target: sieve.c

The sieve benchmark (`build/sieve.s`, stopped before mos-regalloc) has:
- 11 basic blocks, ~80 MOS instructions total
- 339 virtual registers spanning 20+ register classes
- Key patterns: tied operands (IncMB), physical reg defs/uses,
  implicit-def/use, undef operands, COPY, REG_SEQUENCE, JSR, PHI

## Milestones

### Milestone 1: Single Block, Trivial Test ✅

Basic CP allocator: one IntVar per vreg, pairwise != interference,
SSA backwards walk for liveness, LiveVariables for flags.

### Milestone 2: Tied Operands

**Goal**: Handle instructions where a def is tied to a use (same register).

**Scope**:
- Add tied operand constraint: reg(def) == reg(use) when tied
- Tied pairs don't interfere with each other in liveness walk
- Test: IncMB-like instruction with tied def/use

### Milestone 3: Physical Register Constraints

**Goal**: Model physical register defs, uses, and clobbers.

**Scope**:
- Physical reg defs (implicit-def $c) create interference with live vregs
  at that register
- Physical reg uses ($a = COPY %x) constrain the source vreg
- Undef physical reg operands (undef $z) — clobber without value
- Handle COPY to/from physical registers as preassignment constraints

### Milestone 4: Copy Extension

**Goal**: Insert optional copies so the solver can split live ranges and
move values between register classes (Unison Section 4.4).

**Scope**:
- Transform MachineFunction: insert optional store-moves/load-moves
- Alternative copy instructions (STA/LDA/TAX/TAY/TXA/TYA/etc.)
- Full Unison variables: temp(p), ins(o), active(o), live(t)
- Element constraint: reg(temp(p)) for temp selection
- Dominance breaking: active copy requires src ≠ dst register
- Apply: remove inactive copies, lower active copies, reorder instrs

### Milestone 5: Instruction Scheduling

**Goal**: Integrated register allocation and instruction scheduling.

**Scope**:
- Add cycle counts to MOSInstrFormats.td (per addressing mode)
- issue(o) cycle variable per instruction
- Dependency constraints from def-use chains (using cycle latency)
- Memory ordering constraints (loads/stores with MMOs, calls)
- Physical register scheduling (physreg defs/uses at variable cycles)
- Single-issue constraint (all active instrs at different cycles)
- Objective: minimize cost using MOSInstrCost model (bytes + cycles)
- Reorder instructions according to solved schedule

See MOSRegAllocSchedulingPlan.md for detailed design notes.

### Milestone 6: No-Overlap (DiffnProp)

**Goal**: Replace pairwise != with global no-overlap propagator.

**Scope**:
- Port DiffnProp from mos_cp_chuffed.cpp into MOSRegAlloc.cpp
- start(t)/end(t) variables channeled from instruction positions
- Register array packing using start/end/reg variables
- LCG explanations for conflict analysis
- Requires M5 (scheduling) since both dimensions must be variable for
  the global propagator to outperform pairwise !=

### Milestone 7: Spilling

**Goal**: Handle register pressure by spilling to the soft stack.

**Scope**:
- Spill instructions (LDStk, STStk) as copy alternatives
- Model spill cost (Y register + ZP pointer consumed by spill ops)
- Cost model that penalizes spills

### Milestone 8: Multi-Block + Congruences


**Goal**: Handle full functions with multiple basic blocks.

**Scope**:
- PHI nodes → congruence constraints (Unison Section 5 / LSSA)
- Per-block scheduling with cross-block register congruences
- Frequency-weighted objective function
- Branch expansion blocks, function calls (JSR clobbers)

### Milestone 9: 16-bit Pairs

**Goal**: Handle REG_SEQUENCE and imag16 register class.

**Scope**:
- Decompose 16-bit operands into constrained 8-bit pairs
  (reg(hi) = reg(lo) + 1, reg(lo) must be even ZP)
- Adjacency constraints in the model

### Milestone 10: Sieve End-to-End

**Goal**: Run sieve.c through the CP allocator and produce working asm.

**Scope**:
- Integration testing: clang → ... → mos-regalloc → ... → asm
- Performance tuning: solve time target < 5s for sieve
- Fallback: if CP times out, fall back to greedy allocator
- Compare code quality vs current greedy allocator

## What We're Not Planning (Yet)

- Decomposition-based solver (Unison Section 10.1) — performance opt
- Rematerialization (Section 4.5) — quality opt
- Global instruction scheduling across blocks — future work
- Operand forwarding — not relevant for 6502
