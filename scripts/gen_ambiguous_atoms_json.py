#!/usr/bin/env python3
"""Extract `ambiguous_atoms` from Boltz `src/boltz/data/const.py` into boltr-io JSON.

Usage:
  python3 scripts/gen_ambiguous_atoms_json.py /path/to/const.py \\
    boltr-io/data/ambiguous_atoms.json
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path


def expr_to_pyval(node: ast.expr):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Dict):
        return {expr_to_pyval(k): expr_to_pyval(v) for k, v in zip(node.keys, node.values)}
    raise TypeError(f"unsupported AST {type(node)!r}")


def main() -> None:
    if len(sys.argv) != 3:
        print(__doc__.strip(), file=sys.stderr)
        sys.exit(2)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    tree = ast.parse(src.read_text(encoding="utf-8"))
    ambiguous = None
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "ambiguous_atoms":
                    ambiguous = expr_to_pyval(node.value)
                    break
    if ambiguous is None:
        print("ambiguous_atoms assignment not found in const.py", file=sys.stderr)
        sys.exit(1)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fp:
        json.dump(ambiguous, fp, sort_keys=True, separators=(",", ":"))
    print(f"wrote {len(ambiguous)} top-level keys -> {dst}")


if __name__ == "__main__":
    main()
