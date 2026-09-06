#!/usr/bin/env python3
"""Survey overlapping value-graph regions in a single-function SSA MIR dump.

This is an exploratory text reader, not a MIR parser or a scheduling legality
analysis. It ignores physical-register and memory-ordering edges. PHIs and
terminators bound the surveyed block interiors; all virtual-register uses,
including uses outside those interiors, participate in the boundary counts.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import re
from statistics import median


@dataclass
class Instruction:
    number: int
    block: str
    text: str
    opcode: str
    defs: set
    uses: set


def registers(text):
    # Preserve subregister uses as uses of their whole virtual register.
    return {int(m[1]) for m in re.finditer(r"%(\d+)\b", text)}


def read_mir(path):
    text = path.read_text()
    bodies = re.split(r"^body:\s*\|\s*$", text, flags=re.M)
    if len(bodies) != 2 or not re.search(r"^isSSA:\s+true$", bodies[0], re.M):
        raise ValueError("expected one machine function with isSSA: true")
    instructions = []
    block = None
    for line in bodies[1].splitlines():
        match = re.match(r"  (bb\.\d+[^:]*):", line)
        if match:
            block = match[1]
            continue
        if not line.startswith("    "):
            continue
        raw = line.strip()
        if raw.startswith(("successors:", "liveins:", ";")):
            continue
        lhs, separator, rhs = raw.partition(" = ")
        if not separator:
            lhs, rhs = "", raw
        opcode = rhs.split()[0]
        # Ignore undef operands and MMO annotations for value dependence.
        operands = re.sub(r"\bundef\s+%\d+(?:[.:]\w+)*", "", rhs.split(" :: ")[0])
        instructions.append(Instruction(len(instructions), block, raw, opcode,
                                        registers(lhs), registers(operands)))
    definitions = {}
    users = defaultdict(set)
    constants = set()
    for inst in instructions:
        for value in inst.defs:
            if value in definitions:
                raise ValueError(f"multiple definitions of %{value}")
            definitions[value] = inst.number
        for value in inst.uses:
            users[value].add(inst.number)
        if inst.opcode in {"LDImm", "LDImm1", "LDImm16", "LDImm16Remat"}:
            # Only the first def is a constant; another def may be scratch.
            constants.add(int(re.search(r"%(\d+)", inst.text)[1]))
    return instructions, definitions, users, constants


def boundary(region, instructions, users, constants):
    defs = set().union(*(instructions[n].defs for n in region))
    uses = set().union(*(instructions[n].uses for n in region))
    inputs = uses - defs
    outputs = {v for v in defs if users[v] - region}
    retained = {v for v in inputs - constants if users[v] - region}
    internal = {v for v in defs if users[v] and not users[v] - region}
    return inputs - constants, inputs & constants, outputs, retained, internal


def graph(nodes, instructions, definitions, constants):
    successors = {n: set() for n in nodes}
    adjacent = {n: set() for n in nodes}
    readers = defaultdict(list)
    for n in nodes:
        for value in instructions[n].uses:
            pred = definitions.get(value)
            if pred in nodes:
                successors[pred].add(n)
                adjacent[pred].add(n)
                adjacent[n].add(pred)
            if value not in constants:
                readers[value].append(n)
    for uses in readers.values():
        for a, b in combinations(uses, 2):
            adjacent[a].add(b)
            adjacent[b].add(a)
    reach = {n: set(succs) for n, succs in successors.items()}
    for n in reversed(nodes):
        for successor in successors[n]:
            if successor <= n:
                raise ValueError("expected topologically ordered block input")
            reach[n].update(reach[successor])
    return adjacent, reach


def connected(region, adjacent):
    seen = {next(iter(region))}
    todo = list(seen)
    while todo:
        for n in adjacent[todo.pop()] & region - seen:
            seen.add(n)
            todo.append(n)
    return seen == region


def convex(region, nodes, reach):
    # No path between members may pass through an excluded instruction.
    return not any(outside in reach[a] and reach[outside] & region
                   for a in region for outside in nodes - region)


def fmt(values):
    return ", ".join(f"%{v}" for v in sorted(values)) or "—"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mir", type=Path)
    parser.add_argument("--block", action="append", required=True,
                        help="exact block name, e.g. bb.6.while.body")
    parser.add_argument("--max-nodes", type=int, default=14)
    args = parser.parse_args()
    instructions, definitions, users, constants = read_mir(args.mir)
    print("# Overlapping SSA value-region survey\n")
    print("Connectivity includes shared nonconstant inputs. Convexity uses only "
          "virtual-register def-use paths. Counts are structural, not allocation "
          "costs or certificates of safe contraction.\n")
    print("Constant inputs are listed separately; equal constants retain their "
          "distinct SSA identities. Input retention conservatively counts any "
          "use outside the region, without path-sensitive liveness.\n")
    for block in args.block:
        nodes = [i.number for i in instructions if i.block == block
                 and i.opcode != "PHI"
                 and not i.opcode.startswith(("CmpBr", "GBR", "JMP", "RTS"))]
        if not nodes or len(nodes) > args.max_nodes:
            raise ValueError(f"{block}: expected 1..{args.max_nodes} body instructions")
        adjacent, reach = graph(nodes, instructions, definitions, constants)
        all_nodes = set(nodes)
        print(f"## {block}\n")
        for index, node in enumerate(nodes):
            print(f"{index}: `{instructions[node].text}`  ")
        print("\n| Size | Regions | Min / median boundary values | Min / median retained inputs |")
        print("| --- | --- | --- | --- |")
        regions = []
        for size in range(1, len(nodes) + 1):
            widths, retained_counts = [], []
            for combination in combinations(nodes, size):
                region = set(combination)
                if not connected(region, adjacent) or not convex(region, all_nodes, reach):
                    continue
                inputs, consts, outputs, retained, internal = boundary(
                    region, instructions, users, constants)
                widths.append(len(inputs) + len(outputs))
                retained_counts.append(len(retained))
                regions.append((region, inputs, consts, outputs, retained, internal))
            if widths:
                print(f"| {size} | {len(widths)} | {min(widths)} / {median(widths):g} "
                      f"| {min(retained_counts)} / {median(retained_counts):g} |")
        print("\nBoundary width excludes constant inputs and counts a paired "
              "pointer as one SSA value, not two storage bytes.\n")
        print("### Region details\n")
        print("| Members | Inputs | Constants | Outputs | Retained inputs | Internal values |")
        print("| --- | --- | --- | --- | --- | --- |")
        for region, inputs, consts, outputs, retained, internal in regions:
            members = ",".join(str(nodes.index(n)) for n in sorted(region))
            print(f"| {members} | {fmt(inputs)} | {fmt(consts)} | {fmt(outputs)} "
                  f"| {fmt(retained)} | {fmt(internal)} |")
        print()


if __name__ == "__main__":
    main()
