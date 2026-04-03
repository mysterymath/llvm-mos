# Manual Compile: draw_metatile_2_3

## Register Classes (for reference)

| Class   | Physical Registers             |
|---------|-------------------------------|
| ac      | A                             |
| imag8   | RC0..RC255                    |
| aimag8  | A, RC0..RC255                 |
| xy      | X, Y                         |
| cc      | C                             |
| vc      | V                             |
| anyi8   | A, X, Y, RC0..RC255          |
| any     | (all)                         |

## MIR (bb.1.entry)

Liveins: $a, $x, $rc2, $rc3

```
I0:  %0:ac = COPY $a
I1:  %1:imag8 = COPY $x
I2:  %2:aimag8 = COPY $rc2
I3:  %3:xy = COPY $rc3
I4:  %186:ac = LDAbs @VRAM_INDEX
I5:  %154:aimag8, %155:cc = LSR %2
I6:  %117:ac = LDImm 0
I7:  %206:aimag8 = COPY %117
I8:  %156:aimag8, %148:cc = ROR %206, %155
I9:  %165:aimag8, %166:cc = LSR %154
I10: %167:aimag8, %159:cc = ROR %156, %166
I11: %207:aimag8, %177:cc = LSR %165
I12: %176:imag8 = COPY %207
I13: %208:aimag8, %170:cc = ROR %167, %177
I14: %178:ac = COPY %208
I15: %140:ac = ORAImag8 %178, %1
I16: %135:ac = ORAImag8 %0, %176
I17: %129:ac = ORAImm %135, 32
I18: %209:anyi8 = COPY %140
I19: %211:anyi8 = COPY %129
I20: %210:anyi8, %212:anyi8 = IncMB %209(tied-def 0), %211(tied-def 1), implicit-def $c, implicit-def $v
I21: %122:ac = COPY %210
I22: %123:ac = COPY %212
I23: %30:ac = ORAImm %129, -128
I24: %205:xy = COPY %186
I25: STAbsIdx %30, @VRAM_BUF, %205
I26: STAbsIdx %140, @VRAM_BUF + 1, %205
I27: %40:ac = ORAImm %123, -128
I28: STAbsIdx %40, @VRAM_BUF + 6, %205
I29: STAbsIdx %122, @VRAM_BUF + 7, %205
I30: %48:ac = LDImm 3
I31: STAbsIdx %48, @VRAM_BUF + 2, %205
I32: STAbsIdx %48, @VRAM_BUF + 8, %205
I33: %51:ac = LDAbsIdx @all_letters, %3
I34: %213:aimag8 = COPY %51
I35: %103:aimag8, %104:cc = LSR %213
I36: %105:aimag8, %106:cc = LSR %103
I37: %108:aimag8, %109:cc = LSR %105
I38: %214:aimag8, %112:cc = LSR %108
I39: %111:ac = COPY %214
I40: STAbsIdx %111, @VRAM_BUF + 3, %205
I41: %56:ac = LDAbsIdx @all_letters + 37, %3
I42: %215:aimag8 = COPY %56
I43: %92:aimag8, %93:cc = LSR %215
I44: %94:aimag8, %95:cc = LSR %92
I45: %97:aimag8, %98:cc = LSR %94
I46: %216:aimag8, %101:cc = LSR %97
I47: %100:ac = COPY %216
I48: STAbsIdx %100, @VRAM_BUF + 4, %205
I49: %60:ac = LDAbsIdx @all_letters + 74, %3
I50: %217:aimag8 = COPY %60
I51: %81:aimag8, %82:cc = LSR %217
I52: %83:aimag8, %84:cc = LSR %81
I53: %86:aimag8, %87:cc = LSR %83
I54: %218:aimag8, %90:cc = LSR %86
I55: %89:ac = COPY %218
I56: STAbsIdx %89, @VRAM_BUF + 5, %205
I57: %65:ac = ANDImm %51, 15
I58: STAbsIdx %65, @VRAM_BUF + 9, %205
I59: %68:ac = ANDImm %56, 15
I60: STAbsIdx %68, @VRAM_BUF + 10, %205
I61: %71:ac = ANDImm %60, 15
I62: STAbsIdx %71, @VRAM_BUF + 11, %205
I63: %76:ac = LDImm -1
I64: STAbsIdx %76, @VRAM_BUF + 12, %205
I65: %192:cc = LDCImm 0
I66: %78:ac, dead %191:cc, dead %190:vc = ADCImm %186, 12, %192
I67: STAbs %78, @VRAM_INDEX
I68: RTS
```

## Scheduling Graph (Data Dependencies)

Each entry shows: **instruction ← predecessors** (via which vreg).

Stores to @VRAM_BUF at different offsets are to provably different addresses — no
memory ordering between them. All stores must precede RTS.

### Roots (no predecessors)

| Instr | Description |
|-------|-------------|
| I0    | COPY $a (livein) |
| I1    | COPY $x (livein) |
| I2    | COPY $rc2 (livein) |
| I3    | COPY $rc3 (livein) |
| I4    | LDAbs @VRAM_INDEX (memory load, no vreg input) |
| I6    | LDImm 0 |
| I30   | LDImm 3 |
| I63   | LDImm -1 |
| I65   | LDCImm 0 |

### Shift chain (computing address fields from $rc2)

```
I2 ──→ I5 ──┬→ I8 ──→ I10 ──→ I13 ──→ I14 ──→ I15
  (LSR)  │   (ROR)   (ROR)    (ROR)   (COPY)  (ORA %1)
         │    ↑                  ↑
I6→I7 ──┘    │                  │
  (LDImm→COPY)                 │
         │                      │
         └→ I9 ──┬→ I10        │
           (LSR) │   ↑(via %166)
                 │
                 └→ I11 ──┬→ I12 ──→ I16
                   (LSR)  │  (COPY)  (ORA %0)
                          │    ↑
                          │  I0 ┘
                          │
                          └→ I13
                            ↑(via %177)
```

### Address computation chain

```
I15 ──→ I18 ──→ I20 ──→ I21 ──→ I29 (ST @VRAM_BUF+7)
(ORA)  (COPY)  (IncMB) (COPY)
                  │
I16→I17──→I19 ──→I20 ──→ I22 ──→ I27 ──→ I28 (ST @VRAM_BUF+6)
  (ORA)(ORAImm)(COPY)         (COPY)  (ORAImm)
    │
    ├──→ I23 ──→ I25 (ST @VRAM_BUF+0)
    │   (ORAImm)
    │
    └──→ I19 (already shown)

I15 ──→ I26 (ST @VRAM_BUF+1)  [direct use of %140]

I1 ──→ I15  (via %1)
I0 ──→ I16  (via %0)
```

### Index register for stores

```
I4 ──→ I24 ──→ all STAbsIdx instructions (I25,I26,I28,I29,I31,I32,I40,I48,I56,I58,I60,I62,I64)
(LDAbs)(COPY)
```

### Three symmetric load→shift→store clusters

**Cluster A** (all_letters[0]):
```
I3 ──→ I33 ──→ I34 ──→ I35 ──→ I36 ──→ I37 ──→ I38 ──→ I39 ──→ I40 (ST @VRAM_BUF+3)
     (LDAbsIdx)(COPY) (LSR)   (LSR)   (LSR)   (LSR)   (COPY)
          │
          └──→ I57 ──→ I58 (ST @VRAM_BUF+9)
             (ANDImm)
```

**Cluster B** (all_letters[37]):
```
I3 ──→ I41 ──→ I42 ──→ I43 ──→ I44 ──→ I45 ──→ I46 ──→ I47 ──→ I48 (ST @VRAM_BUF+4)
     (LDAbsIdx)(COPY) (LSR)   (LSR)   (LSR)   (LSR)   (COPY)
          │
          └──→ I59 ──→ I60 (ST @VRAM_BUF+10)
             (ANDImm)
```

**Cluster C** (all_letters[74]):
```
I3 ──→ I49 ──→ I50 ──→ I51 ──→ I52 ──→ I53 ──→ I54 ──→ I55 ──→ I56 (ST @VRAM_BUF+5)
     (LDAbsIdx)(COPY) (LSR)   (LSR)   (LSR)   (LSR)   (COPY)
          │
          └──→ I61 ──→ I62 (ST @VRAM_BUF+11)
             (ANDImm)
```

### Constant stores

```
I30 ──→ I31 (ST @VRAM_BUF+2)
    └──→ I32 (ST @VRAM_BUF+8)

I63 ──→ I64 (ST @VRAM_BUF+12)
```

### VRAM_INDEX update

```
I4 ──→ I66 (ADCImm %186, 12)
I65 ──→ I66
I66 ──→ I67 (ST @VRAM_INDEX)
```

### Terminator

```
I68 (RTS) ← all stores: I25, I26, I28, I29, I31, I32, I40, I48, I56, I58, I60, I62, I64, I67
```

## Complete Predecessor Table

| Instr | Predecessors (via vreg) |
|-------|------------------------|
| I0    | — |
| I1    | — |
| I2    | — |
| I3    | — |
| I4    | — |
| I5    | I2 (%2) |
| I6    | — |
| I7    | I6 (%117) |
| I8    | I7 (%206), I5 (%155) |
| I9    | I5 (%154) |
| I10   | I8 (%156), I9 (%166) |
| I11   | I9 (%165) |
| I12   | I11 (%207) |
| I13   | I10 (%167), I11 (%177) |
| I14   | I13 (%208) |
| I15   | I14 (%178), I1 (%1) |
| I16   | I0 (%0), I12 (%176) |
| I17   | I16 (%135) |
| I18   | I15 (%140) |
| I19   | I17 (%129) |
| I20   | I18 (%209), I19 (%211) |
| I21   | I20 (%210) |
| I22   | I20 (%212) |
| I23   | I17 (%129) |
| I24   | I4 (%186) |
| I25   | I23 (%30), I24 (%205) |
| I26   | I15 (%140), I24 (%205) |
| I27   | I22 (%123) |
| I28   | I27 (%40), I24 (%205) |
| I29   | I21 (%122), I24 (%205) |
| I30   | — |
| I31   | I30 (%48), I24 (%205) |
| I32   | I30 (%48), I24 (%205) |
| I33   | I3 (%3) |
| I34   | I33 (%51) |
| I35   | I34 (%213) |
| I36   | I35 (%103) |
| I37   | I36 (%105) |
| I38   | I37 (%108) |
| I39   | I38 (%214) |
| I40   | I39 (%111), I24 (%205) |
| I41   | I3 (%3) |
| I42   | I41 (%56) |
| I43   | I42 (%215) |
| I44   | I43 (%92) |
| I45   | I44 (%94) |
| I46   | I45 (%97) |
| I47   | I46 (%216) |
| I48   | I47 (%100), I24 (%205) |
| I49   | I3 (%3) |
| I50   | I49 (%60) |
| I51   | I50 (%217) |
| I52   | I51 (%81) |
| I53   | I52 (%83) |
| I54   | I53 (%86) |
| I55   | I54 (%218) |
| I56   | I55 (%89), I24 (%205) |
| I57   | I33 (%51) |
| I58   | I57 (%65), I24 (%205) |
| I59   | I41 (%56) |
| I60   | I59 (%68), I24 (%205) |
| I61   | I49 (%60) |
| I62   | I61 (%71), I24 (%205) |
| I63   | — |
| I64   | I63 (%76), I24 (%205) |
| I65   | — |
| I66   | I4 (%186), I65 (%192) |
| I67   | I66 (%78) |
| I68   | I25, I26, I28, I29, I31, I32, I40, I48, I56, I58, I60, I62, I64, I67 |

## Copy-Propagated MIR (Value Calculus)

All COPYs propagated. All register classes widened to max (Anyi8 / Anyi1).
Each instruction annotates the **constraints** it imposes on its operands.
Tied-defs noted as `=` (output must share register with input).
Dead outputs shown as `_` (clobbered but unused).

Liveins: $a, $x, $rc2, $rc3 (all Anyi8)

```
I4:  %186:Anyi8 = LDAbs @VRAM_INDEX           ; out ∈ GPR
I5:  %154:Anyi8, %155:Anyi1 = LSR $rc2        ; in ∈ AImag8, %154=in(tied), out2 ∈ Cc
I6:  %117:Anyi8 = LDImm 0                     ; out ∈ GPR
I8:  %156:Anyi8, _ = ROR %117, %155           ; in1 ∈ AImag8, in2 ∈ Cc, %156=in1(tied)
I9:  %165:Anyi8, %166:Anyi1 = LSR %154        ; in ∈ AImag8, %165=in(tied), out2 ∈ Cc
I10: %167:Anyi8, _ = ROR %156, %166           ; in1 ∈ AImag8, in2 ∈ Cc, %167=in1(tied)
I11: %207:Anyi8, %177:Anyi1 = LSR %165        ; in ∈ AImag8, %207=in(tied), out2 ∈ Cc
I13: %208:Anyi8, _ = ROR %167, %177           ; in1 ∈ AImag8, in2 ∈ Cc, %208=in1(tied)
I15: %140:Anyi8 = ORAImag8 %208, $x           ; in1 ∈ Ac, in2 ∈ Imag8, %140=in1(tied)
I16: %135:Anyi8 = ORAImag8 $a, %207           ; in1 ∈ Ac, in2 ∈ Imag8, %135=in1(tied)
I17: %129:Anyi8 = ORAImm %135, 32             ; in ∈ Ac, %129=in(tied)
I20: %210:Anyi8, %212:Anyi8 = IncMB %140, %129  ; %210=%140(tied), %212=%129(tied), clobbers $c,$v
I23: %30:Anyi8 = ORAImm %129, -128            ; in ∈ Ac, %30=in(tied)
I25: STAbsIdx %30, @VRAM_BUF, %186            ; data ∈ Ac, idx ∈ XY
I26: STAbsIdx %140, @VRAM_BUF + 1, %186       ; data ∈ Ac, idx ∈ XY
I27: %40:Anyi8 = ORAImm %212, -128            ; in ∈ Ac, %40=in(tied)
I28: STAbsIdx %40, @VRAM_BUF + 6, %186        ; data ∈ Ac, idx ∈ XY
I29: STAbsIdx %210, @VRAM_BUF + 7, %186       ; data ∈ Ac, idx ∈ XY
I30: %48:Anyi8 = LDImm 3                      ; out ∈ GPR
I31: STAbsIdx %48, @VRAM_BUF + 2, %186        ; data ∈ Ac, idx ∈ XY
I32: STAbsIdx %48, @VRAM_BUF + 8, %186        ; data ∈ Ac, idx ∈ XY
I33: %51:Anyi8 = LDAbsIdx @all_letters, $rc3  ; out ∈ GPR, idx ∈ XY
I35: %103:Anyi8, _ = LSR %51                  ; in ∈ AImag8, %103=in(tied)
I36: %105:Anyi8, _ = LSR %103                 ; in ∈ AImag8, %105=in(tied)
I37: %108:Anyi8, _ = LSR %105                 ; in ∈ AImag8, %108=in(tied)
I38: %214:Anyi8, _ = LSR %108                 ; in ∈ AImag8, %214=in(tied)
I40: STAbsIdx %214, @VRAM_BUF + 3, %186       ; data ∈ Ac, idx ∈ XY
I41: %56:Anyi8 = LDAbsIdx @all_letters+37, $rc3 ; out ∈ GPR, idx ∈ XY
I43: %92:Anyi8, _ = LSR %56                   ; in ∈ AImag8, %92=in(tied)
I44: %94:Anyi8, _ = LSR %92                   ; in ∈ AImag8, %94=in(tied)
I45: %97:Anyi8, _ = LSR %94                   ; in ∈ AImag8, %97=in(tied)
I46: %216:Anyi8, _ = LSR %97                  ; in ∈ AImag8, %216=in(tied)
I48: STAbsIdx %216, @VRAM_BUF + 4, %186       ; data ∈ Ac, idx ∈ XY
I49: %60:Anyi8 = LDAbsIdx @all_letters+74, $rc3 ; out ∈ GPR, idx ∈ XY
I51: %81:Anyi8, _ = LSR %60                   ; in ∈ AImag8, %81=in(tied)
I52: %83:Anyi8, _ = LSR %81                   ; in ∈ AImag8, %83=in(tied)
I53: %86:Anyi8, _ = LSR %83                   ; in ∈ AImag8, %86=in(tied)
I54: %218:Anyi8, _ = LSR %86                  ; in ∈ AImag8, %218=in(tied)
I56: STAbsIdx %218, @VRAM_BUF + 5, %186       ; data ∈ Ac, idx ∈ XY
I57: %65:Anyi8 = ANDImm %51, 15               ; in ∈ Ac, %65=in(tied)
I58: STAbsIdx %65, @VRAM_BUF + 9, %186        ; data ∈ Ac, idx ∈ XY
I59: %68:Anyi8 = ANDImm %56, 15               ; in ∈ Ac, %65=in(tied)
I60: STAbsIdx %68, @VRAM_BUF + 10, %186       ; data ∈ Ac, idx ∈ XY
I61: %71:Anyi8 = ANDImm %60, 15               ; in ∈ Ac, %71=in(tied)
I62: STAbsIdx %71, @VRAM_BUF + 11, %186       ; data ∈ Ac, idx ∈ XY
I63: %76:Anyi8 = LDImm -1                     ; out ∈ GPR
I64: STAbsIdx %76, @VRAM_BUF + 12, %186       ; data ∈ Ac, idx ∈ XY
I65: %192:Anyi1 = LDCImm 0                    ; out ∈ Cc
I66: %78:Anyi8, _, _ = ADCImm %186, 12, %192  ; in1 ∈ Ac, in2 ∈ Cc, %78=in1(tied)
I67: STAbs %78, @VRAM_INDEX                    ; data ∈ GPR
I68: RTS
```

### Multi-use values (cluster boundaries)

After copy propagation, these values have multiple uses:

| Value | Uses | Consumers |
|-------|------|-----------|
| $rc3  | 3    | I33, I41, I49 |
| %186  | 14   | I25,I26,I28,I29,I31,I32,I40,I48,I56,I58,I60,I62,I64,I66 |
| %140  | 2    | I20(tied), I26 |
| %129  | 2    | I20(tied), I23 |
| %48   | 2    | I31, I32 |
| %51   | 2    | I35, I57 |
| %56   | 2    | I43, I59 |
| %60   | 2    | I51, I61 |

All other values are single-use.

## Clusters (connected components via single-use vregs)

**C1: Address Computation** {I5, I6, I8, I9, I10, I11, I13, I15, I16, I17}
The entire shift/rotate cascade plus the ORA combine. Liveins $a, $x, $rc2
enter directly — no COPY intermediaries.
- Inputs: $a, $x, $rc2
- Outputs: %140, %129

**C2: IncMB + Result Stores** {I20, I27, I28, I29}
16-bit increment, high-byte ORA, and both stores.
- Inputs: %140, %129, %186
- Outputs: stores to @VRAM_BUF+6, +7

**C3: Address High Store** {I23, I25}
ORAImm -128 then store.
- Inputs: %129, %186
- Outputs: store to @VRAM_BUF+0

**C4: Address Low Store** {I26}
Direct store of %140.
- Inputs: %140, %186
- Outputs: store to @VRAM_BUF+1

**C5a: Tile A Shift** {I35, I36, I37, I38, I40}
4x LSR then store (high nybble of all_letters[0]).
- Inputs: %51, %186
- Outputs: store to @VRAM_BUF+3

**C5b: Tile B Shift** {I43, I44, I45, I46, I48}
Same pattern for all_letters[37].
- Inputs: %56, %186
- Outputs: store to @VRAM_BUF+4

**C5c: Tile C Shift** {I51, I52, I53, I54, I56}
Same pattern for all_letters[74].
- Inputs: %60, %186
- Outputs: store to @VRAM_BUF+5

**C6a: Tile A Mask** {I57, I58}
ANDImm 15 then store (low nybble of all_letters[0]).
- Inputs: %51, %186
- Outputs: store to @VRAM_BUF+9

**C6b: Tile B Mask** {I59, I60}
Same for all_letters[37].
- Inputs: %56, %186
- Outputs: store to @VRAM_BUF+10

**C6c: Tile C Mask** {I61, I62}
Same for all_letters[74].
- Inputs: %60, %186
- Outputs: store to @VRAM_BUF+11

**C7: Constant 3 Store A** {I31}
- Inputs: %48, %186
- Outputs: store to @VRAM_BUF+2

**C8: Constant 3 Store B** {I32}
- Inputs: %48, %186
- Outputs: store to @VRAM_BUF+8

**C9: Constant -1 Store** {I63, I64}
LDImm then store.
- Inputs: %186
- Outputs: store to @VRAM_BUF+12

**C10: VRAM_INDEX Update** {I65, I66, I67}
LDCImm, ADCImm, store.
- Inputs: %186
- Outputs: store to @VRAM_INDEX

**Singletons** (produce multi-use values):
- {I4}: LDAbs @VRAM_INDEX → %186
- {I30}: LDImm 3 → %48
- {I33}: LDAbsIdx @all_letters → %51 (needs $rc3)
- {I41}: LDAbsIdx @all_letters+37 → %56 (needs $rc3)
- {I49}: LDAbsIdx @all_letters+74 → %60 (needs $rc3)

**Terminator**: {I68}: RTS (depends on all stores)

Note: $rc3 is a livein consumed by three singletons (I33, I41, I49). It is not
itself an instruction — just an input value.

### Inter-cluster dependency graph

```
$a,$x,$rc2 ──→ C1 ──→ %140 ──→ C2, C4
                  └──→ %129 ──→ C2, C3

$rc3 ──→ I33 ──→ %51 ──→ C5a, C6a
     ──→ I41 ──→ %56 ──→ C5b, C6b
     ──→ I49 ──→ %60 ──→ C5c, C6c

I4 ──→ %186 ──→ C2, C3, C4, C5a-c, C6a-c, C7, C8, C9, C10
I30 ──→ %48 ──→ C7, C8

All store clusters ──→ I68 (RTS)
```

## Is it always easy to schedule a cluster?

**Linear chains: yes.** The tile shift clusters (C5a/b/c) are pure chains:
LSR→LSR→LSR→LSR→store. There's exactly one valid scheduling order. The
tied-defs chain the register assignment through the whole sequence — one
register for the shifting value, done.

**DAGs with fan-out/fan-in: no.** C1 (Address Computation) is a DAG. I5 (LSR)
produces both %154 and %155, consumed by different instructions (I9 and I8).
I11 produces both %207 and %177, consumed by I16 and I13. At each fan-out
point, we must choose which successor to schedule first, and that choice
affects how long the other output stays live — which affects register pressure
and whether register constraints can be met.

Concretely in C1: I15 requires its first input (%208) in Ac, and I16 also
requires its first input ($a) in Ac. They can't both hold A simultaneously.
The order in which we schedule the two "arms" of the DAG determines whether
the A register is available when each ORA needs it. Getting this wrong could
make the cluster unschedulable without spilling, even though a valid ordering
exists.

**Multi-use inputs with tied-defs: tricky.** C2 has IncMB with tied-defs on
%140 and %129. The tied-def means IncMB *overwrites* these values in-place.
But %140 is also used by C4 (store), and %129 by C3 (ORAImm + store). If we
schedule C2 before C3/C4, the tied-def destroys the values they need. So the
inter-cluster ordering of C2 vs C3/C4 is constrained — but that's an
inter-cluster concern, not intra-cluster.

## Chain Schedules

For each linear chain cluster: the forced instruction order and register
assignment derived from constraints + tied-defs.

Key constraint propagation: every STAbsIdx needs data ∈ Ac (= A) and
idx ∈ XY. This pulls backward through tied-defs — if the store needs A,
and the value is tied through a chain, the whole chain runs in A.

### C5a: Tile A Shift {I35, I36, I37, I38, I40}

```
I35: LSR %51       → %103    ; tied, AImag8
I36: LSR %103      → %105    ; tied, AImag8
I37: LSR %105      → %108    ; tied, AImag8
I38: LSR %108      → %214    ; tied, AImag8
I40: STAbsIdx %214           ; data ∈ Ac
```
Store needs %214 ∈ Ac. Tied-defs chain: %51 = %103 = %105 = %108 = %214.
So the whole chain must be in A (= AImag8 ∩ Ac).
**Registers**: shift value = **A**, index %186 = **X or Y**

Note: %51 is multi-use (also C6a). Tied-def in I35 destroys %51.
C6a must consume %51 before C5a runs, OR a copy is needed.

### C5b: Tile B Shift {I43, I44, I45, I46, I48}

Identical structure. Whole chain in **A**. Index %186 in **XY**.
%56 is multi-use (also C6b) — same destruction issue.

### C5c: Tile C Shift {I51, I52, I53, I54, I56}

Identical structure. Whole chain in **A**. Index %186 in **XY**.
%60 is multi-use (also C6c) — same destruction issue.

### C6a: Tile A Mask {I57, I58}

```
I57: ANDImm %51, 15 → %65   ; in ∈ Ac, tied
I58: STAbsIdx %65            ; data ∈ Ac
```
ANDImm requires Ac. Tied: %51 = %65. Whole chain in **A**. Index %186 in **XY**.
Also destroys %51 (tied-def). Same value, same problem — whoever runs second
sees a destroyed %51.

### C6b: Tile B Mask {I59, I60}

Identical. Chain in **A**. Destroys %56.

### C6c: Tile C Mask {I61, I62}

Identical. Chain in **A**. Destroys %60.

### C3: Address High Store {I23, I25}

```
I23: ORAImm %129, -128 → %30  ; in ∈ Ac, tied
I25: STAbsIdx %30              ; data ∈ Ac
```
%129 = %30, both **A**. Index %186 in **XY**.
Destroys %129 (multi-use: also C2). C2 must get %129 before C3 runs,
or a copy is needed.

### C4: Address Low Store {I26}

```
I26: STAbsIdx %140             ; data ∈ Ac, idx ∈ XY
```
%140 in **A**. Index %186 in **XY**.
Does not destroy %140 (no tied-def, just reads it).

### C7: Constant 3 Store A {I31}

```
I31: STAbsIdx %48             ; data ∈ Ac, idx ∈ XY
```
%48 in **A**. %186 in **XY**. (Does not destroy %48.)

### C8: Constant 3 Store B {I32}

Same. %48 in **A**. %186 in **XY**.

### C9: Constant -1 Store {I63, I64}

```
I63: LDImm -1 → %76          ; out ∈ GPR
I64: STAbsIdx %76             ; data ∈ Ac, idx ∈ XY
```
GPR ∩ Ac = {A}. So %76 = **A**. %186 in **XY**.
No multi-use issues (%76 is single-use, %186 only read).

### C10: VRAM_INDEX Update {I65, I66, I67}

```
I65: LDCImm 0 → %192         ; out ∈ Cc → C
I66: ADCImm %186, 12, %192   ; in1 ∈ Ac, in2 ∈ Cc, %78=in1(tied)
     → %78
I67: STAbs %78                ; data ∈ GPR
```
%192 = **C**. %186 must be in **A** (for ADCImm). %78 = %186 (tied) = **A**.

Problem: %186 must be in **A** here, but every STAbsIdx needs %186 in
**XY** as an index. %186 can't be in both. This means we need a copy of
%186 — one in XY (for stores) and one in A (for ADCImm). This is
unavoidable.

### Singletons

- **I4**: LDAbs @VRAM_INDEX → %186 ∈ GPR. Will need copies to both A and XY.
- **I30**: LDImm 3 → %48 ∈ GPR. Stores need Ac → **A**.
- **I33**: LDAbsIdx @all_letters, $rc3 → %51 ∈ GPR. Stores need Ac → **A**. Index $rc3 ∈ **XY**.
- **I41**: same → %56 ∈ **A**. $rc3 ∈ **XY**.
- **I49**: same → %60 ∈ **A**. $rc3 ∈ **XY**.

### C1: Address Computation {I5, I6, I8, I9, I10, I11, I13, I15, I16, I17}

C1 is a DAG with fan-out at each LSR (producing both an 8-bit value and a
carry). Two principles linearize it and determine register assignments:

1. **Cc is hard to copy; A is cheap.** Each LSR→ROR pair must be adjacent
   (carry consumed immediately). The 8-bit values are flexible.
2. **The value inheriting A goes first.** At fan-out points where two
   consumers both eventually need A, schedule the one whose value is
   already in A — it proceeds without a copy.

#### Tied-def chains (two interleaved values)

The LSR/ROR cascade operates on two values that stay in fixed registers
via tied-defs:

- **"High byte"**: $rc2 → %154 → %165 → %207 (all tied via LSRs)
  Must be in one AImag8 register. $rc2 starts in RC2, so: **RC2**.
- **"Low byte"**: %117 → %156 → %167 → %208 (all tied via RORs)
  Must be in one AImag8 register. %117 is loaded via LDImm (out ∈ GPR).
  Choosing **A** is natural — LDA #0 is the obvious instruction.

After the cascade: **%208 is in A**, %207 is in RC2.

#### I15 vs I16: which goes first?

Both need their first operand in Ac (= A):
- I15: ORAImag8 %208, $x — %208 is **already in A** (inherited from ROR chain)
- I16: ORAImag8 $a, %207 — $a is NOT in A (A was clobbered by LDImm 0)

**I15 goes first**: 0 copies needed to set up its Ac operand.
If I16 went first, we'd need to save %208 AND restore $a — 2 extra copies.

#### Full schedule with register assignments

Livein state: $a=**A**, $x=**X**, $rc2=**RC2**

```
 copy: STA RC_a         ; save $a (A will be clobbered by LDImm)
 I5:   LSR RC2          ; → %154=RC2, %155=C
 I6:   LDA #0           ; → %117=A (clobbers $a — already saved)
 I8:   ROR A            ; consumes C from I5. → %156=A
 I9:   LSR RC2          ; → %165=RC2, %166=C
 I10:  ROR A            ; consumes C from I9. → %167=A
 I11:  LSR RC2          ; → %207=RC2, %177=C
 I13:  ROR A            ; consumes C from I11. → %208=A
 copy: STX RC_x         ; save $x to Imag8 (needed by I15 as in2 ∈ Imag8)
 I15:  ORA RC_x         ; %208(=A) ORA $x(=RC_x). → %140=A. No copy needed!
 copy: STA RC_140       ; save %140 (multi-use: C2, C4)
 copy: LDA RC_a         ; restore $a to A
 I16:  ORA RC2          ; $a(=A) ORA %207(=RC2). → %135=A
 I17:  ORA #32          ; → %129=A (tied to %135)
```

**Register state after C1:**
- %140 = **RC_140** (saved)
- %129 = **A** (last computed value)
- %207 = **RC2** (still there, but no longer needed after I16)

**Copies introduced: 4** (save $a, save $x, save %140, restore $a).
All are 2-cycle zero-page STA/LDA instructions.

**Outputs**: %140 = RC_140, %129 = A (both multi-use).

**Why I15 before I16?** %208 inherits A from the ROR tied-def chain.
I15 consumes %208 in Ac — it runs directly, no copy. If I16 went first,
we'd need to save %208 and restore $a (2 extra copies) before I16 could
use A, and then another copy to get %208 back for I15. The A-inheriting
consumer always goes first.

### C2: IncMB + Result Stores {I20, I27, I28, I29}

C2 is a DAG: I20 fans out to Arm A (I27→I28) and Arm B (I29).

#### A-inheritance determines arm ordering

After C1: %129 = **A**, %140 = **RC_140**.
I20 (IncMB) has tied-defs: %210 = %140, %212 = %129.
So after I20: %210 = **RC_140**, %212 = **A**.

- **Arm A** (I27→I28): I27 (ORAImm) needs %212 in Ac. %212 is **already in A**. 0 copies.
- **Arm B** (I29): I29 (STAbsIdx) needs %210 in Ac. %210 is in RC_140. 1 copy.

Arm A first: **1 copy total**. Arm B first would require saving %212,
restoring %210, then restoring %212 — **3 copies**. Clear winner.

#### Full schedule with register assignments

```
 I20:  IncMB RC_140, A    ; → %210=RC_140, %212=A. clobbers C,V.
 I27:  ORA #$80           ; %212(=A) ORA -128. → %40=A. No copy needed!
 I28:  STA VRAM_BUF+6,idx ; store %40(=A). idx=%186 ∈ XY.
 copy: LDA RC_140         ; get %210 into A
 I29:  STA VRAM_BUF+7,idx ; store %210(=A). idx=%186 ∈ XY.
```

**Copies introduced: 1** (load %210 from RC_140 into A).

**Why Arm A first?** %212 inherits A (via tied-def from %129, which was
the last value in A after C1). The A-inheriting arm goes first.

### Observations

1. **Everything flows through A.** Every store's data operand must be in A.
   Every ORA/AND/ADC needs A. The 6502's accumulator-centric architecture
   means A is the bottleneck — chains take turns using it.

2. **%186 has a split personality.** The stores need it in XY (index), but
   C10 needs it in A (arithmetic). A copy is mandatory.

3. **Paired clusters share a victim.** C5a/C6a both destroy %51 via
   tied-defs. C5b/C6b destroy %56. C5c/C6c destroy %60. Within each pair,
   one must run before the other (getting the original value), and the
   second needs either a copy or to be scheduled first. Similarly C2/C3
   compete over %129, and C2/C4 over %140.

4. **The three tile pairs (C5+C6) are symmetric.** They have identical
   structure, identical constraints, and no data dependencies between them.
   Scheduling any permutation of A/B/C is equivalent.

## Cluster Effects

Each cluster is treated as an atomic "super-instruction." Effects:
- **Use**: value read at entry. *(kill)* if this cluster is the last consumer.
- **Def**: value produced and live at exit.
- **Clobber**: register written during execution but dead at exit.

A cluster has no dead defs — a written register that's dead at exit is a
clobber, not a def. Clobbers interfere with defs (they can't share a
register). In the conservative model, all clobbers interfere with all defs.
Where the actual interference is sparser, an interference graph is noted.

Kill annotations below are *tentative* — the final kill depends on
inter-cluster scheduling (which cluster runs last among a value's consumers).

### Singletons

**I4**: LDAbs @VRAM_INDEX
- Uses: (none)
- Defs: %186 ∈ GPR
- Clobbers: (none)

**I30**: LDImm 3
- Uses: (none)
- Defs: %48 ∈ GPR
- Clobbers: (none)

**I33**: LDAbsIdx @all_letters, $rc3
- Uses: $rc3 ∈ XY
- Defs: %51 ∈ GPR
- Clobbers: (none)
- Note: GPR def and XY use can't be the same register (LDA abs,X → def=A;
  LDX abs,Y → def=X, use=Y; etc.)

**I41**: LDAbsIdx @all_letters+37, $rc3
- Same shape as I33. Defs %56 ∈ GPR. Uses $rc3 ∈ XY.

**I49**: LDAbsIdx @all_letters+74, $rc3
- Same shape. Defs %60 ∈ GPR. Uses $rc3 ∈ XY.

### C1: Address Computation

- Uses: $a ∈ Ac (kill), $x ∈ XY (kill), $rc2 ∈ AImag8 (kill)
- Defs: %129 ∈ Ac, %140 ∈ Imag8
- Clobbers: 1×Cc, 2×Imag8 (temporaries RC_a, RC_x)

The use $rc2's register is reused for the shift chain and consumed by I16.
After I16, it's stale — but it's freed before %140's def is born, so it
does NOT interfere with %140. Similarly RC_x (saves $x) is freed at I15,
before %140's def. RC_a (saves $a) is alive until just after %140's def,
so it DOES interfere.

**Interference graph** (Imag8 class only):

```
%140(def) ── interferes ── RC_a(clobber)
%140(def) ── no interference ── RC_x(clobber)
%140(def) ── no interference ── $rc2's register
```

Peak Imag8 demand: **3** ($rc2's reg + RC_a + %140, all live at the
STA RC_140 instruction). Total distinct Imag8 regs used: 4, but not
all simultaneously.

### C2: IncMB + Result Stores

- Uses: %140 ∈ Imag8, %129 ∈ Ac, %186 ∈ XY
- Defs: (none — all values consumed by stores within the cluster)
- Clobbers: 1×Ac, 1×Cc, 1×Vc, 1×Imag8 (%140's register, stale after store)

Note: %140 and %129 are consumed via tied-defs (IncMB overwrites them
in-place). Other clusters needing %140/%129 must run before C2, or copies
are needed. Whether C2 "kills" %140/%129 depends on inter-cluster order.

### C3: Address High Store

- Uses: %129 ∈ Ac, %186 ∈ XY
- Defs: (none)
- Clobbers: 1×Ac (%30 dead after store)

Tied-def: %129 overwritten by %30 (ORAImm). Destructive use.

### C4: Address Low Store

- Uses: %140 ∈ Ac, %186 ∈ XY
- Defs: (none)
- Clobbers: (none — just reads %140 and %186, no tied-def)

Note: %140 is defined in Imag8 by C1, but C4 needs it in Ac. This
requires a copy Imag8→Ac before C4 runs.

### C5a/b/c: Tile Shift Chains

All three are identical in structure (using %51/%56/%60 respectively).

- Uses: %51 ∈ Ac, %186 ∈ XY
- Defs: (none)
- Clobbers: 1×Ac (shift result, dead after store), 4×Cc (dead carries from LSRs)

Tied-def: input value (%51) destroyed by LSR chain. Destructive use.

### C6a/b/c: Tile Mask Chains

- Uses: %51 ∈ Ac, %186 ∈ XY
- Defs: (none)
- Clobbers: 1×Ac (mask result, dead after store)

Tied-def: input value (%51) destroyed by ANDImm. Destructive use.
Does NOT clobber Cc (AND affects N/Z but not C).

### C7, C8: Constant 3 Stores

- Uses: %48 ∈ Ac, %186 ∈ XY
- Defs: (none)
- Clobbers: (none — just reads)

### C9: Constant -1 Store

- Uses: %186 ∈ XY
- Defs: (none)
- Clobbers: 1×Ac (LDImm writes A, dead after store)

### C10: VRAM_INDEX Update

- Uses: %186 ∈ Ac, %186 ∈ XY — **CONFLICT**: same value needs two classes
- Defs: (none)
- Clobbers: 1×Ac, 1×Cc, 1×Vc

Actually: %186 needs to be in Ac for ADCImm. But all stores need %186 in
XY. This is an inter-cluster conflict requiring a copy. Within C10, the
use of %186 is ∈ Ac only.

Corrected:
- Uses: %186 ∈ Ac
- Defs: (none)
- Clobbers: 1×Ac (%78, dead after store), 1×Cc, 1×Vc

### I68: RTS

- Uses: (all stores must have completed — ordering constraint, not a register use)
- Defs: (none)
- Clobbers: (none)

### Summary Table

| Cluster | Uses | Defs | Clobbers |
|---------|------|------|----------|
| I4      | —    | %186:GPR | — |
| I30     | —    | %48:GPR | — |
| I33     | $rc3:XY | %51:GPR | — |
| I41     | $rc3:XY | %56:GPR | — |
| I49     | $rc3:XY | %60:GPR | — |
| C1      | $a:Ac, $x:XY, $rc2:AImag8 | %129:Ac, %140:Imag8 | Cc, 2×Imag8 |
| C2      | %140:Imag8†, %129:Ac†, %186:XY | — | Ac, Cc, Vc, Imag8 |
| C3      | %129:Ac†, %186:XY | — | Ac |
| C4      | %140:Ac‡, %186:XY | — | — |
| C5a     | %51:Ac†, %186:XY | — | Ac, Cc |
| C5b     | %56:Ac†, %186:XY | — | Ac, Cc |
| C5c     | %60:Ac†, %186:XY | — | Ac, Cc |
| C6a     | %51:Ac†, %186:XY | — | Ac |
| C6b     | %56:Ac†, %186:XY | — | Ac |
| C6c     | %60:Ac†, %186:XY | — | Ac |
| C7      | %48:Ac, %186:XY | — | — |
| C8      | %48:Ac, %186:XY | — | — |
| C9      | %186:XY | — | Ac |
| C10     | %186:Ac | — | Ac, Cc, Vc |
| I68     | (ordering only) | — | — |

† = destructive use (tied-def overwrites the value in-place)
‡ = class mismatch: C1 defs %140 in Imag8, but C4 needs it in Ac (copy required)

## Merging Clusters (greedy register-state matching)

**Rule**: At each step, schedule the ready cluster whose inputs best match
the current register state (fewest copies to set up). When a def register
can be chosen (e.g., GPR = {A, X, Y}), pick the one that doesn't clobber
live values.

Multi-use values that will be destroyed (tied-def) must be saved before
the first destructive consumer. This save is charged as 1 copy. The save
instruction (STA zp) does NOT disturb A, so the value is still available
in A immediately after.

### Step 1: C1 (3/3 match)

**State**: A=$a, X=$x, RC2=$rc2, RC3=$rc3

Ready: C1, I4, I30, I33, I41, I49.

| Candidate | Inputs needed | Match | Copies |
|-----------|--------------|-------|--------|
| C1 | $a∈Ac, $x∈XY, $rc2∈AImag8 | A=$a ✓, X=$x ✓, RC2=$rc2 ✓ | 0 |
| I4 | (none) | — | 0 |
| I30 | (none) | — | 0 |
| I33 | $rc3∈XY | RC3∉XY ✗ | 1 |

**C1** has 3 inputs, all matching. Best choice. (I4/I30 have 0 inputs — no
positive match. I33 needs a copy.)

→ Schedule **C1** (with its 4 internal copies: save $a, save $x, save %140,
  restore $a).

**State after**: A=%129, X=$x(dead), RC_140=%140, RC2=free, RC_a=free, RC_x=free

### Step 2: I4 as LDX (0 copies, preserves A)

Ready: I4, I30, I33, I41, I49. (C2/C3/C4 need %186, not yet produced.)

| Candidate | Inputs needed | Match | Copies | Notes |
|-----------|--------------|-------|--------|-------|
| I4 | (none) | — | 0 | Def %186∈GPR. Use LDX: X is free, preserves A=%129 |
| I30 | (none) | — | 0 | Def %48∈GPR. LDA clobbers %129. LDX takes X. |
| I33 | $rc3∈XY | ✗ | 1 | |

I4 and I30 both cost 0, but **I4** using LDX places %186 in X (which stores
need as idx∈XY) **without clobbering A=%129**. I30 has no register choice
that avoids clobbering something useful.

→ Schedule **I4** as LDX @VRAM_INDEX.

**State after**: A=%129, X=%186

### Step 3: save %129 + C3 (0 setup copies)

Ready: C2, C3, C4, C10, I30, I33, I41, I49.

| Candidate | Inputs needed | Match | Copies |
|-----------|--------------|-------|--------|
| C3 | %129∈Ac, %186∈XY | A=%129 ✓, X=%186 ✓ | 0 |
| C2 | %129∈Ac, %140∈Imag8, %186∈XY | all ✓ | 0 — but destroys %140,%129 needed by C3,C4 |
| C4 | %140∈Ac, %186∈XY | A≠%140 ✗, X=%186 ✓ | 1 |
| C10 | %186∈Ac | X=%186 but need Ac ✗ | 1+ (also must save %129) |

C3 and C2 both have perfect matches. But **C2 destroys %129 and %140 via
tied-defs**, which C3 and C4 still need. Scheduling C2 now would force
extra saves. **C3** is the safe choice — it only destroys %129, and we need
just one save (for C2's later use).

→ Save: STA RC_129 (A still = %129).
→ Schedule **C3**: ORA #$80, STA VRAM_BUF,X.

**State after**: A=free, X=%186, RC_140=%140, RC_129=%129

### Step 4: I30+C7+C8 merge (0 copies)

Ready: C2, C4, C10, I30, I33, I41, I49.

| Candidate | Inputs needed | Match | Copies |
|-----------|--------------|-------|--------|
| I30 | (none) | — | 0 (A is free, LDA #3) |
| C9 | %186∈XY | X=%186 ✓ | 0 (A is free) |
| C4 | %140∈Ac, %186∈XY | ✗, ✓ | 1 |
| C2 | %129∈Ac, %140∈Imag8, %186∈XY | ✗, ✓, ✓ | 1 |
| C10 | %186∈Ac | ✗ | 1 |
| I33 | $rc3∈XY | ✗ | 1 |

**I30** costs 0 (A is free). After I30: A=%48. Now C7 and C8 become ready
(need %48∈Ac, %186∈XY). Both match perfectly — **A=%48 ✓, X=%186 ✓**.
STA doesn't modify A, so %48 stays in A for both stores.

→ Schedule **I30**: LDA #3 → A=%48.
→ Schedule **C7**: STA VRAM_BUF+2,X (A=%48 still live).
→ Schedule **C8**: STA VRAM_BUF+8,X.

**State after**: A=free, X=%186, RC_140=%140, RC_129=%129

### Step 5: C9 (0 copies)

Ready: C2, C4, C10, I33, I41, I49.

C9 needs only %186∈XY (✓). A is free for LDImm.

→ Schedule **C9**: LDA #-1, STA VRAM_BUF+12,X.

**State after**: A=free, X=%186

### Step 6: C4 (1 copy)

Ready: C2, C4, C10, I33, I41, I49.

| Candidate | Copies | Notes |
|-----------|--------|-------|
| C4 | 1 (LDA RC_140) | Unblocks C2 (C4 must precede C2 — C2 destroys %140) |
| C10 | 1 (TXA) | |
| I33 | 1 (LDY RC3) | |

All cost 1 copy. **C4** is chosen because it unblocks C2 (C4 must consume
%140 before C2's tied-def destroys it).

→ Schedule **C4**: LDA RC_140, STA VRAM_BUF+1,X.

**State after**: A=free, X=%186, RC_140=%140 (still there — LDA doesn't clear source)

### Step 7: C2 (1 setup copy + 1 internal)

Ready: C2, C10, I33, I41, I49.

C2 needs %129∈Ac (need LDA RC_129: 1 copy), %140∈Imag8 (✓ RC_140),
%186∈XY (✓ X). 1 setup copy.

→ Schedule **C2**: LDA RC_129, IncMB(RC_140, A), ORA #$80, STA VRAM_BUF+6,X,
  LDA RC_140 (now %210), STA VRAM_BUF+7,X.

**State after**: A=free, X=%186, RC_140=free, RC_129=free

### Step 8: C10 (1 copy)

Ready: C10, I33, I41, I49.

C10 needs %186∈Ac. %186 is in X. TXA copies X→A (X unchanged). 1 copy.

→ Schedule **C10**: TXA, CLC, ADC #12, STA @VRAM_INDEX.

**State after**: A=free, X=%186, Y=free, C=free

### Step 9: Tile setup (1 copy for $rc3)

Ready: I33, I41, I49.

All three need $rc3∈XY. $rc3 is in RC3 (Imag8). Load once: LDY RC3. 1 copy.
Y will hold $rc3 for all three loads. X holds %186 for all stores.

→ Copy: LDY RC3. **Y=$rc3, X=%186.**

### Step 10: Tile A — I33+C6a+C5a (2 copies)

Ready: I33, I41, I49.

All three loads cost the same (all use Y=$rc3, output to A which is free).
Symmetric — pick I33 arbitrarily.

→ **I33**: LDA @all_letters,Y → **A=%51**.

Now C5a and C6a are both ready, both need %51∈Ac. A=%51 ✓. Perfect match
for both — symmetric. But both are destructive (tied-def). Must save %51
first for the second consumer.

→ Save: STA RC_tmp → RC_tmp=%51, **A=%51 still live**.
→ **C6a**: AND #15, STA VRAM_BUF+9,X. (A=%51, matches. 0 copies.)
→ Restore: LDA RC_tmp → **A=%51**.
→ **C5a**: LSR, LSR, LSR, LSR, STA VRAM_BUF+3,X.

**State after**: A=free, X=%186, Y=$rc3, RC_tmp=free

### Step 11: Tile B — I41+C6b+C5b (2 copies)

Same pattern. A is free, Y=$rc3 still live.

→ **I41**: LDA @all_letters+37,Y → **A=%56**.
→ Save: STA RC_tmp.
→ **C6b**: AND #15, STA VRAM_BUF+10,X.
→ Restore: LDA RC_tmp.
→ **C5b**: LSR×4, STA VRAM_BUF+4,X.

### Step 12: Tile C — I49+C6c+C5c (2 copies)

Same pattern.

→ **I49**: LDA @all_letters+74,Y → **A=%60**.
→ Save: STA RC_tmp.
→ **C6c**: AND #15, STA VRAM_BUF+11,X.
→ Restore: LDA RC_tmp.
→ **C5c**: LSR×4, STA VRAM_BUF+5,X.

### Step 13: RTS

→ **I68**: RTS.

### Copy budget

| Phase | Copies | Reason |
|-------|--------|--------|
| C1 internal | 4 | save $a, save $x, save %140, restore $a |
| save %129 | 1 | multi-use: C2 needs it later |
| I4 (LDX) | 0 | X was free |
| I30+C7+C8 | 0 | A was free, %48 stays in A for both stores |
| C9 | 0 | A was free |
| C4 | 1 | LDA RC_140 |
| C2 | 2 | LDA RC_129 (setup) + LDA RC_140 (arm B internal) |
| C10 | 1 | TXA |
| $rc3 setup | 1 | LDY RC3 (amortized over 3 loads) |
| Tile A | 2 | save + restore %51 |
| Tile B | 2 | save + restore %56 |
| Tile C | 2 | save + restore %60 |
| **Total** | **16** | |

## Final Routine

### Register Assignment

| Symbolic | Physical | Lifetime |
|----------|----------|----------|
| RC_a     | RC4      | C1 internal (save $a → restore $a) |
| RC_x     | RC5      | C1 internal (save $x → ORA RC5) |
| RC_140   | RC6      | C1 save → C2 arm B load (long-lived) |
| RC_129   | RC7      | step 3 save → C2 setup load |
| RC_tmp   | RC8      | per-tile save → restore (reused 3×) |

No conflicts: peak simultaneous Imag8 usage is 4 (RC2, RC3, RC4, RC6 during
C1). With 256 RC registers available, assignment is trivial — just pick any
unused RC for each symbolic name.

A, X, Y assignments are fully determined by the schedule. Cc is implicit
(C flag is a singleton).

### Assembly

```asm
; draw_metatile_2_3
; Liveins: A=$a(nmt), X=$x(x_coord), RC2=$rc2(y_coord), RC3=$rc3(mtile_idx)

; ---- C1: Address Computation ----
    STA RC4             ; save $a
    LSR RC2             ; high byte: shift $rc2 right
    LDA #0              ; zero for low byte ROR chain
    ROR A               ; rotate carry into low byte
    LSR RC2             ; high byte: shift again
    ROR A               ; rotate carry into low byte
    LSR RC2             ; high byte: shift again
    ROR A               ; rotate carry → A=%208 (low byte done)
    STX RC5             ; save $x to Imag8
    ORA RC5             ; A = %208 | $x = %140
    STA RC6             ; save %140
    LDA RC4             ; restore $a
    ORA RC2             ; A = $a | %207 = %135
    ORA #32             ; A = %135 | 32 = %129

; ---- I4 + save %129 + C3: addr high store ----
    LDX VRAM_INDEX      ; X = %186 (store index for rest of routine)
    STA RC7             ; save %129 (for C2)
    ORA #$80            ; A = %129 | $80 = high addr byte
    STA VRAM_BUF,X      ; store @VRAM_BUF+0

; ---- I30 + C7 + C8: constant 3 stores ----
    LDA #3
    STA VRAM_BUF+2,X
    STA VRAM_BUF+8,X

; ---- C9: constant -1 store ----
    LDA #$FF
    STA VRAM_BUF+12,X

; ---- C4: addr low store ----
    LDA RC6             ; A = %140
    STA VRAM_BUF+1,X

; ---- C2: IncMB + result stores ----
    LDA RC7             ; A = %129
    INC RC6             ; low byte (%140) += 1
    BNE +4              ; skip if no wrap
    CLC
    ADC #1              ; high byte (%129) += 1
    ORA #$80            ; A = high(addr+1) | $80
    STA VRAM_BUF+6,X   ; store @VRAM_BUF+6
    LDA RC6             ; A = low(addr+1)
    STA VRAM_BUF+7,X   ; store @VRAM_BUF+7

; ---- C10: VRAM_INDEX update ----
    TXA                 ; A = %186 (VRAM_INDEX)
    CLC
    ADC #12
    STA VRAM_INDEX

; ---- Tile setup ----
    LDY RC3             ; Y = $rc3 (mtile_idx)

; ---- Tile A (all_letters[0]) ----
    LDA all_letters,Y
    STA RC8             ; save tile byte
    AND #15
    STA VRAM_BUF+9,X   ; low nybble
    LDA RC8             ; restore tile byte
    LSR A
    LSR A
    LSR A
    LSR A
    STA VRAM_BUF+3,X   ; high nybble

; ---- Tile B (all_letters[37]) ----
    LDA all_letters+37,Y
    STA RC8
    AND #15
    STA VRAM_BUF+10,X
    LDA RC8
    LSR A
    LSR A
    LSR A
    LSR A
    STA VRAM_BUF+4,X

; ---- Tile C (all_letters[74]) ----
    LDA all_letters+74,Y
    STA RC8
    AND #15
    STA VRAM_BUF+11,X
    LDA RC8
    LSR A
    LSR A
    LSR A
    LSR A
    STA VRAM_BUF+5,X

    RTS
```

**Instruction count**: 58 (42 original + 16 copies).
**Imag8 registers used**: 7 (RC2–RC8), peak 4 simultaneous.
**Register assignment**: trivial — schedule fully determines A/X/Y at every
point; Imag8 is unconstrained (256 available, 7 needed).
