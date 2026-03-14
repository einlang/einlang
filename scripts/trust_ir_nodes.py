#!/usr/bin/env python3
"""
Trust IR nodes: replace getattr(obj, "attr", default) with direct attribute access
when obj is likely an IR node and attr is a known IR slot.

For optional attributes on mixed AST/IR (e.g. defid, name), use the helpers in
shared.optional_attr (opt_defid, opt_name, opt_attr) instead of getattr so the
script leaves them unchanged and call sites stay clear.

Usage:
  python3 scripts/trust_ir_nodes.py [--dry-run] [path]
  path: default src/einlang (scans .py under it)
  --dry-run: print changes only, do not edit files
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional


def collect_ir_slots(nodes_path: Path) -> set:
    """Parse ir/nodes.py and return set of all __slots__ attribute names."""
    text = nodes_path.read_text(encoding="utf-8")
    attrs = set()
    for m in re.finditer(r"__slots__\s*=\s*\(([^)]+)\)", text):
        for part in m.group(1).split(","):
            part = part.strip().strip("'\"")
            if part and not part.startswith("_"):
                attrs.add(part)
    return attrs


# Names we treat as likely IR node variables (visitor args, loop vars over IR).
IR_LIKE_NAMES = frozenset({
    "node", "expr", "e", "n", "stmt", "statement", "clause", "clauses",
    "idx", "index", "indices", "arm", "arms", "loop", "loops", "binding", "bindings",
    "func", "func_def", "function", "access", "pattern", "pat", "constraint",
    "body", "value", "left", "right", "operand", "arg", "args", "argument", "arguments",
    "call", "mod", "module", "item", "items", "guard", "guards",
    "reduction", "red", "lowered", "initial", "rloop", "rec_loop", "specialized_func",
    "generic_func", "new_func", "enclosing_function", "ast_func", "ast_call",
    "scrutinee", "where_clause", "wc", "range_ir", "range_obj", "rng",
    "array_expr", "decl", "var_def", "ident", "method_ir", "idx_expr", "idx_with_var",
    "lp", "new_idx", "elem", "element", "part", "parts", "alt", "alternatives",
    "new_node", "new_body", "new_val", "new_expr", "new_indices", "new_array",
    "shape_access", "array_ir", "constraint_expr", "left_expr", "right_expr",
    "index_expr", "var_ident", "einstein_node", "expr_node", "reduction_expr",
    "mul_left", "mul_right", "add_left", "add_right", "body_expr", "second_idx",
    "loop_var", "red0_var", "red1_var", "loop_variable", "variable",
    "arr", "loc", "func", "decl", "idx_expr", "constraint", "c", "red",
    "cur_func", "iterable", "range_obj", "range_ir", "enc", "left", "right",
})

# Object names we never replace (non-IR or optional API).
# "clause": can be EinsteinClauseIR (.value) or LoweredEinsteinClauseIR (.body).
SKIP_OBJ_NAMES = frozenset({
    "self", "tcx", "ty", "type_info", "source_type", "target_type",
    "definition", "definitions", "resolver", "opts", "sexpr",
    "obj", "x", "a", "v", "val", "result", "out", "c", "g", "b",
    "f", "m", "name", "tag", "method_name", "type_obj", "ti", "t", "pt", "meta",
    "clause",
})

# Attributes often used on non-IR (AST, type objects, tcx, optional API). Do not replace.
# Also "expr" when object can be any ExpressionIR (e.g. BuiltinCallIR has no .expr).
SKIP_ATTRS = frozenset({
    "param_type", "type_annotation",  # AST Parameter has type_annotation
    "module_path", "function_ir_map", "specialized_functions", "reporter",
    "source_files", "program_ir", "_variable_decl_stack", "_current_function",
    "_target_defid", "_current_declaration", "_current_einstein_clause",
    "_rest_pattern_names", "_profile_functions", "_resolver_defid_to_lowered",
    "array_name",  # optional on some nodes
    "pattern",  # string on BindingIR; also used as attr on AST
    "kind", "element_type", "shape", "is_dynamic_rank", "__name__",  # type objects
    "accept", "value",  # value is both slot and method on enums
    "I",  # serialization
    "file", "line", "column", "end_line", "end_column",  # SourceLocation; keep getattr for compatibility
    "expr",  # not all ExpressionIR have .expr (e.g. BuiltinCallIR has args, not expr)
    "arguments",  # FunctionCallIR has .arguments, BuiltinCallIR has .args; call sites may see either
    "defid",  # AST nodes (FunctionDefinition, ModuleAccess) may not have .defid; LambdaIR has no .defid
    "name",  # ArrayLiteralIR and some IR nodes have no .name; AST vs IR differ
    "inner_pattern",  # AST BindingPattern has .pattern, IR BindingPatternIR has .inner_pattern
    "operation",  # AST ReductionExpression has .function_name, IR has .operation
})


def should_replace(obj_name: str, attr: str, ir_attrs: set) -> bool:
    if obj_name in SKIP_OBJ_NAMES:
        return False
    if attr in SKIP_ATTRS:
        return False
    if attr not in ir_attrs:
        return False
    if obj_name not in IR_LIKE_NAMES:
        return False
    return True


def replacement_for(obj: str, attr: str, default: str) -> str | None:
    """Return replacement string for getattr(obj, attr, default), or None if skip."""
    if default == "None":
        return f"{obj}.{attr}"
    if default in ("[]", "[ ]"):
        return f"({obj}.{attr} or [])"
    if default == "{}":
        return f"({obj}.{attr} or {{}})"
    if default.startswith('"') or default.startswith("'"):
        return f"({obj}.{attr} or {default})"
    return None


def replace_getattr_line(line: str, ir_attrs: set) -> tuple:
    """Return (new_line, changed). Replaces all getattr(ir_like, ir_attr, default) on the line."""
    # Match getattr(obj, "attr", default) anywhere on the line
    pattern = re.compile(
        r"getattr\s*\(\s*(\w+)\s*,\s*[\"'](\w+)[\"']\s*,\s*(None|\[\]|\{\}|\[\s*\]|\"[^\"]*\"|'[^']*')\s*\)"
    )
    new_line = line
    changed = False

    def repl(m):
        nonlocal changed
        obj, attr, default = m.group(1), m.group(2), m.group(3)
        if not should_replace(obj, attr, ir_attrs):
            return m.group(0)
        rep = replacement_for(obj, attr, default)
        if rep is None:
            return m.group(0)
        changed = True
        return rep

    new_line = pattern.sub(repl, line)
    return new_line, changed


def process_file(path: Path, ir_attrs: set, dry_run: bool) -> int:
    content = path.read_text(encoding="utf-8")
    lines = content.splitlines(keepends=True)
    changes = []
    for i, line in enumerate(lines):
        if "getattr(" not in line:
            continue
        new_line, changed = replace_getattr_line(line.rstrip("\n"), ir_attrs)
        if changed:
            if not new_line.endswith("\n"):
                new_line += "\n"
            changes.append((i + 1, line, new_line))
    if not changes:
        return 0
    if dry_run:
        for ln, old, new in changes:
            print(f"{path}:{ln}")
            print(f"  - {old.rstrip()}")
            print(f"  + {new.rstrip()}")
        return len(changes)
    for i, (ln, old, new) in enumerate(changes):
        idx = ln - 1
        if lines[idx] == old or lines[idx].rstrip("\n") == old.rstrip("\n"):
            lines[idx] = new if new.endswith("\n") else new + "\n"
    path.write_text("".join(lines), encoding="utf-8")
    return len(changes)


def _find_root_with_ir_nodes(start: Path) -> Optional[Path]:
    """Walk up from start until we find a directory containing ir/nodes.py."""
    p = start.resolve().parent if start.is_file() else start.resolve()
    while p != p.parent:
        if (p / "ir" / "nodes.py").exists():
            return p
        p = p.parent
    return None


def main() -> None:
    dry_run = "--dry-run" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--dry-run"]
    path_arg = Path(args[0]).resolve() if args else Path(__file__).resolve().parent.parent / "src" / "einlang"
    if not path_arg.exists():
        path_arg = Path(__file__).resolve().parent.parent
        if not (path_arg / "src" / "einlang").exists():
            print("Usage: python3 scripts/trust_ir_nodes.py [--dry-run] [path]", file=sys.stderr)
            print("  path: directory (default src/einlang) or a single .py file", file=sys.stderr)
            sys.exit(2)
        path_arg = path_arg / "src" / "einlang"

    if path_arg.is_file() and path_arg.suffix == ".py":
        root = _find_root_with_ir_nodes(path_arg)
        if root is None:
            print("ir/nodes.py not found above path", file=sys.stderr)
            sys.exit(1)
        files_to_process = [path_arg]
    else:
        root = path_arg
        nodes_path = root / "ir" / "nodes.py"
        if not nodes_path.exists():
            print("ir/nodes.py not found", file=sys.stderr)
            sys.exit(1)
        files_to_process = sorted(root.rglob("*.py"))

    nodes_path = root / "ir" / "nodes.py"
    if not nodes_path.exists():
        print("ir/nodes.py not found", file=sys.stderr)
        sys.exit(1)
    ir_attrs = collect_ir_slots(nodes_path)

    total = 0
    for path in files_to_process:
        if "trust_ir_nodes" in str(path):
            continue
        n = process_file(path, ir_attrs, dry_run)
        total += n
    if dry_run and total:
        print(f"\nWould change {total} occurrence(s). Run without --dry-run to apply.")
    elif total and not dry_run:
        print(f"Updated {total} occurrence(s).")
    elif total == 0 and not dry_run:
        print("No getattr(ir_like, ir_attr, default) to replace.")


if __name__ == "__main__":
    main()
