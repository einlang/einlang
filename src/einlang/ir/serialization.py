"""
IR Serialization to S-Expressions
====================================

Converts IR to canonical S-expression format for testing and debugging.
Serialization is lossless for all information used by the runtime: DefIds
(function_defid, defid on calls/bindings/params/identifiers/index-vars),
module_path on calls, program source_files/bindings (defid_to_name derived on demand),
lowered IR (clause body, loops, bindings, guards, reduction_ranges, indices),
and all expression/statement structure the backend visits.

Uses structured sexpr (nested lists + sexpdata.Symbol),
then pretty-prints for readable output.
"""

from typing import Any, List, Optional

try:
    import sexpdata
except ImportError:
    sexpdata = None  # type: ignore

from ..ir.nodes import IRNode, ExpressionIR

# Keywords/symbols (from _sym) - unquoted. Names (node.name, etc.) use "".
_KNOWN_SYMBOLS = frozenset({
    "nil", "program", "variable", "literal", "binary-op", "unary-op",
    "rectangular-access", "function-call", "builtin-call", "cast", "array-literal",
    "index", "index-var", "index-rest",
    "array-comprehension", "range", "loop", "binding", "lowered-einstein", "lowered-einstein-clause",
    "lowered-comprehension", "lowered-reduction", "lowered-iteration",
    "param", "block", "if", "tuple", "tuple-access", "lambda", "reduction",
    "function-value", "einstein-value", "type", "unknown", "rectangular-type", "jagged-type",
    "tuple-type", "function-type", "callee", "true", "false", "...",
    "i32", "i64", "f32", "f64", "bool", "str", "+", "-", "*", "/", "%", "**",
    "==", "!=", "<", "<=", ">", ">=", "&&", "||", "!", "sum", "prod", "min", "max",
})


def _is_symbol(s: str) -> bool:
    """True if s is a keyword/symbol (unquoted), not a name."""
    return s.startswith(":") or s in _KNOWN_SYMBOLS


def _pretty_dumps(sexpr: Any, indent: int = 0, indent_str: str = "  ", max_line: int = 100) -> str:
    """
    Pretty-print structured sexpr. Keeps short forms on one line; breaks only when needed.
    """
    if sexpr is None:
        return "()"
    if isinstance(sexpr, bool):
        return "true" if sexpr else "false"
    if isinstance(sexpr, (int, float)):
        return str(sexpr)
    # Check Symbol before str (sexpdata.Symbol subclasses str)
    if sexpdata and isinstance(sexpr, sexpdata.Symbol):
        return sexpr.value()
    if isinstance(sexpr, str):
        # Names (i, j, c) use ""; keywords from _sym use Symbol and stay unquoted
        if _is_symbol(sexpr):
            return sexpr
        escaped = sexpr.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if sexpdata and isinstance(sexpr, sexpdata.Brackets):
        inner = " ".join(_pretty_dumps(e, indent + 1, indent_str, max_line) for e in sexpr.I)
        return f"[{inner}]"
    if isinstance(sexpr, list):
        if not sexpr:
            return "()"
        parts = [_pretty_dumps(e, indent + 1, indent_str, max_line) for e in sexpr]
        one_line = "(" + " ".join(parts) + ")"
        if len(one_line) <= max_line and "\n" not in one_line:
            return one_line
        prefix = indent_str * indent
        next_prefix = indent_str * (indent + 1)
        # First element on same line as ( to avoid orphan (; no space after (
        rest = "\n".join(next_prefix + p for p in parts[1:])
        inner = parts[0] + ("\n" + rest if rest else "")
        return f"({inner}\n{prefix})"
    return str(sexpr)


def serialize_ir(node: IRNode, include_location: bool = False, include_type_info: bool = False, pretty: bool = True) -> str:
    """
    Serialize IR node to S-expression string.
    
    Build structured sexpr, then pretty-print by default.
    
    Args:
        node: IR node to serialize
        include_location: Include source location metadata
        include_type_info: Include type information
        pretty: Use pretty-printed format (default True). Set False for compact single-line.
    
    Returns:
        S-expression string (pretty-printed by default)
    """
    serializer = IRSerializer(include_location=include_location, include_type_info=include_type_info)
    sexpr = serializer.serialize_to_sexpr(node)
    if pretty:
        return _pretty_dumps(sexpr)
    if sexpdata:
        return sexpdata.dumps(sexpr)
    return str(sexpr)


class IRSerializer:
    """
    IR to structured S-expression serializer.
    
    Returns nested lists (structured sexpr) like einlang/ir/serialization.py.
    Use sexpdata.Symbol for keywords to avoid quotes.
    """
    
    def __init__(self, include_location: bool = False, include_type_info: bool = False):
        self.include_location = include_location
        self.include_type_info = include_type_info
    
    def _sym(self, s: str) -> Any:
        """Convert string to symbol (no quotes in output). pattern."""
        if sexpdata:
            return sexpdata.Symbol(s)
        return s

    def _brackets(self, items: list) -> Any:
        """DefId pair as [krate index] for golden snapshot compatibility."""
        if sexpdata:
            return sexpdata.Brackets(items)
        return items
    
    def serialize_to_sexpr(self, node: Any) -> Any:
        """Serialize any IR node to structured sexpr (list/Symbol/str)."""
        if node is None:
            return [self._sym("nil")]
        
        node_type = type(node).__name__
        method_name = f"_serialize_{node_type}"
        if hasattr(self, method_name):
            method = getattr(self, method_name)
            return method(node)
        
        return self._serialize_generic(node)
    
    def serialize(self, node: Any) -> str:
        """Serialize to string (legacy)."""
        return _pretty_dumps(self.serialize_to_sexpr(node))
    
    def _serialize_generic(self, node: Any) -> list:
        """Generic serialization for unknown nodes"""
        return [self._sym(type(node).__name__), self._sym("...")]
    
    def _add_expr_metadata(self, node: Any, core: list) -> list:
        """Add expression metadata (inferred_type, :loc, :shape_info) to serialized expression"""
        from ..ir.nodes import ExpressionIR
        
        result = list(core)
        if isinstance(node, ExpressionIR) and self.include_type_info:
            if hasattr(node, 'type_info') and node.type_info:
                result.extend([self._sym(":inferred_type"), self._serialize_type(node.type_info)])
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            result.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        if hasattr(node, 'shape_info') and getattr(node, 'shape_info', None) is not None:
            result.extend([self._sym(":shape_info"), self._serialize_shape_info(node.shape_info)])
        return result

    def _serialize_shape_info(self, shape_info: Any) -> list:
        if shape_info is None:
            return []
        out = []
        for x in shape_info:
            if x is None or (isinstance(x, str) and x.strip() == "?"):
                out.append(self._sym("?"))
            elif isinstance(x, int):
                out.append(x)
            else:
                out.append(self._sym("?"))
        return out
    
    def _serialize_type(self, type_obj: Any) -> Any:
        """Serialize type objects to structured sexpr"""
        from ..shared.types import PrimitiveType, RectangularType, JaggedType, TupleType, FunctionType, UNKNOWN, Type, TypeKind

        if type_obj is UNKNOWN or (isinstance(type_obj, Type) and type_obj.kind == TypeKind.UNKNOWN):
            return [self._sym("type"), self._sym("unknown")]
        if isinstance(type_obj, PrimitiveType):
            return [self._sym("type"), self._sym(type_obj.name)]
        elif isinstance(type_obj, RectangularType):
            element = self._serialize_type(type_obj.element_type)
            dims = []
            if type_obj.shape:
                for d in type_obj.shape:
                    if d is None:
                        dims.append(self._sym("?"))
                    elif isinstance(d, (int, float, str)):
                        dims.append(d)
                    else:
                        dims.append(self._serialize_type(d))
            out = [self._sym("rectangular-type"), element]
            if dims:
                out.append(dims)
            if getattr(type_obj, "is_dynamic_rank", False):
                out.append(self._sym("true"))
            return out
        elif isinstance(type_obj, JaggedType):
            element = self._serialize_type(type_obj.element_type)
            depth = type_obj.nesting_depth if type_obj.nesting_depth is not None else 1
            return [self._sym("jagged-type"), element, depth]
        elif isinstance(type_obj, TupleType):
            elements = [self._serialize_type(t) for t in type_obj.element_types]
            return [self._sym("tuple-type"), elements]
        elif isinstance(type_obj, FunctionType):
            params = [self._serialize_type(p) for p in type_obj.param_types]
            ret = self._serialize_type(type_obj.return_type)
            return [self._sym("function-type"), params, ret]
        elif isinstance(type_obj, list):
            if len(type_obj) == 1:
                return [self._sym("rectangular-type"), self._serialize_type(type_obj[0])]
            return [self._sym("tuple-type"), [self._serialize_type(t) for t in type_obj]]
        return [self._sym("type"), str(type_obj)]
    
    # === Literals ===
    
    def _serialize_LiteralIR(self, node) -> list:
        """Serialize literal: (literal value type)"""
        from ..shared.types import PrimitiveType
        
        value = node.value
        
        if isinstance(value, range):
            start = [self._sym("literal"), value.start, self._sym("i32")]
            stop = [self._sym("literal"), value.stop, self._sym("i32")]
            core = [self._sym("range"), start, stop]
            return self._add_expr_metadata(node, core)
        
        # Determine type
        if hasattr(node, 'type_info') and node.type_info and isinstance(node.type_info, PrimitiveType):
            type_str = node.type_info.name
        elif isinstance(value, bool):
            type_str = "bool"
        elif isinstance(value, int):
            type_str = "i32"
        elif isinstance(value, float):
            type_str = "f32"
        elif isinstance(value, str):
            type_str = "str"
        else:
            type_str = "unknown"
        
        # Store bool as symbol to avoid sexpdata's True->() conversion
        val = value
        if isinstance(value, bool):
            val = self._sym("true" if value else "false")
        
        core = [self._sym("literal"), val, self._sym(type_str)]
        return self._add_expr_metadata(node, core)
    
    # === Identifiers ===
    
    def _serialize_IdentifierIR(self, node) -> list:
        """Serialize identifier: (variable "name" :defid [krate index])."""
        core = [self._sym("variable"), node.name]
        if getattr(node, 'defid', None) is not None:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        return self._add_expr_metadata(node, core)

    def _serialize_IndexVarIR(self, node) -> list:
        """Serialize variable index slot: (index-var "name" :defid [...] :range ...)."""
        core = [self._sym("index-var"), node.name]
        if getattr(node, "defid", None) is not None:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        if getattr(node, "range_ir", None) is not None:
            core.extend([self._sym(":range"), self.serialize_to_sexpr(node.range_ir)])
        return self._add_expr_metadata(node, core)

    def _serialize_IndexRestIR(self, node) -> list:
        """Serialize rest index slot: (index-rest "name" :defid [krate index])."""
        core = [self._sym("index-rest"), node.name]
        if getattr(node, "defid", None) is not None:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        return self._add_expr_metadata(node, core)
    
    # === Binary Operations ===
    
    def _serialize_BinaryOpIR(self, node) -> list:
        """Serialize binary operation: (binary-op OP left right). Operator as symbol for canonical format."""
        left = self.serialize_to_sexpr(node.left)
        right = self.serialize_to_sexpr(node.right)
        op_sym = node.operator.value if hasattr(node.operator, "value") else node.operator
        core = [self._sym("binary-op"), self._sym(op_sym), left, right]
        return self._add_expr_metadata(node, core)
    
    # === Unary Operations ===
    
    def _serialize_UnaryOpIR(self, node) -> list:
        """Serialize unary operation: (unary-op OP operand). Operator as symbol."""
        operand = self.serialize_to_sexpr(node.operand)
        op_sym = node.operator.value if hasattr(node.operator, "value") else node.operator
        core = [self._sym("unary-op"), self._sym(op_sym), operand]
        return self._add_expr_metadata(node, core)
    
    # === Array Access ===

    def _serialize_index_slot(self, idx: Any) -> list:
        """Serialize an index slot in rectangular-access. Indices use IndexVarIR, IndexRestIR, or literals only (no IdentifierIR)."""
        from ..ir.nodes import IndexVarIR, IndexRestIR
        if isinstance(idx, IndexVarIR):
            return self.serialize_to_sexpr(idx)  # → (index-var "name" ...)
        if isinstance(idx, IndexRestIR):
            return self.serialize_to_sexpr(idx)  # → (index-rest "name" ...)
        return self.serialize_to_sexpr(idx)

    def _serialize_RectangularAccessIR(self, node) -> list:
        """Serialize array access: (rectangular-access array (indices...)); indices use index-var/index-rest/index, not variable."""
        array = self.serialize_to_sexpr(node.array)
        indices = [self._serialize_index_slot(idx) for idx in (node.indices or [])]
        core = [self._sym("rectangular-access"), array, indices]
        return self._add_expr_metadata(node, core)

    def _serialize_MemberAccessIR(self, node) -> list:
        """Serialize member access (e.g. t.0, arr.shape): (member-access object member)."""
        object_sexpr = self.serialize_to_sexpr(node.object)
        member = getattr(node, "member", "")
        if isinstance(member, int):
            core = [self._sym("member-access"), object_sexpr, member]
        else:
            core = [self._sym("member-access"), object_sexpr, str(member)]
        return self._add_expr_metadata(node, core)
    
    # === Function Calls ===
    
    def _serialize_FunctionCallIR(self, node) -> list:
        """Serialize function call: (function-call (callee <expr>) (args...) :module_path ...)"""
        args = [self.serialize_to_sexpr(arg) for arg in node.arguments]
        callee = self.serialize_to_sexpr(node.callee_expr)
        core = [self._sym("function-call"), [self._sym("callee"), callee], args]
        if getattr(node, 'module_path', None):
            core.extend([self._sym(":module_path"), list(node.module_path)])
        return self._add_expr_metadata(node, core)
    
    def _serialize_BuiltinCallIR(self, node) -> list:
        """Serialize builtin call: (builtin-call "name" (args...) :defid [krate, index])"""
        args = [self.serialize_to_sexpr(arg) for arg in node.args]
        core = [self._sym("builtin-call"), node.builtin_name, args]
        if hasattr(node, 'defid') and node.defid:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        return self._add_expr_metadata(node, core)
    
    # === Cast Expressions ===
    
    def _serialize_CastExpressionIR(self, node) -> list:
        """Serialize cast: (cast expr type)"""
        from ..shared.types import PrimitiveType
        
        expr = self.serialize_to_sexpr(node.expr)
        if isinstance(node.target_type, PrimitiveType):
            ty_sexpr = self._sym(node.target_type.name)
        elif node.target_type is not None:
            ty_sexpr = self._serialize_type(node.target_type)
        else:
            ty_sexpr = self._sym("i32")
        core = [self._sym("cast"), expr, ty_sexpr]
        return self._add_expr_metadata(node, core)

    def _serialize_InterpolatedStringIR(self, node) -> list:
        """Serialize interpolated string: (interpolated-string (part1 part2 ...)); parts are str or expr."""
        parts_sexpr = []
        for p in (node.parts or []):
            if isinstance(p, str):
                parts_sexpr.append(p)
            else:
                parts_sexpr.append(self.serialize_to_sexpr(p))
        core = [self._sym("interpolated-string"), parts_sexpr]
        return self._add_expr_metadata(node, core)
    
    # === Array Literals ===
    
    def _serialize_ArrayLiteralIR(self, node) -> list:
        """Serialize array literal: (array-literal (elements...))"""
        elements = [self.serialize_to_sexpr(elem) for elem in node.elements]
        core = [self._sym("array-literal"), elements]
        return self._add_expr_metadata(node, core)
    
    def _serialize_ArrayComprehensionIR(self, node) -> list:
        """Serialize array comprehension: (array-comprehension body :loop_vars (...) :ranges (...) :constraints (...))"""
        body = self.serialize_to_sexpr(node.body)
        loop_vars_sexpr = [self.serialize_to_sexpr(v) for v in node.loop_vars]
        core = [self._sym("array-comprehension"), body,
                self._sym(":loop_vars"), loop_vars_sexpr,
                self._sym(":ranges"), [self.serialize_to_sexpr(r) for r in node.ranges]]
        if node.constraints:
            core.extend([self._sym(":constraints"), [self.serialize_to_sexpr(c) for c in node.constraints]])
        return self._add_expr_metadata(node, core)
    
    # === Range Expressions ===
    
    def _serialize_RangeIR(self, node) -> list:
        """Serialize range: (range start end [:inclusive true])"""
        start = self.serialize_to_sexpr(node.start) if node.start else [self._sym("nil")]
        end = self.serialize_to_sexpr(node.end) if node.end else [self._sym("nil")]
        core = [self._sym("range"), start, end]
        if getattr(node, 'inclusive', False):
            core.extend([self._sym(":inclusive"), self._sym("true")])
        return self._add_expr_metadata(node, core)
    
    # === Einstein Declarations ===
    
    def _serialize_EinsteinClauseIR(self, node) -> list:
        """Serialize one Einstein clause (indices + value + variable_ranges)."""
        indices = [idx.name if hasattr(idx, 'name') else self.serialize_to_sexpr(idx) for idx in (getattr(node, 'indices', None) or [])]
        value = self.serialize_to_sexpr(node.value) if getattr(node, 'value', None) else [self._sym("nil")]
        out = [self._sym(":indices"), indices, self._sym(":value"), value]
        vr = getattr(node, 'variable_ranges', None) or {}
        var_ranges = [[self._brackets([d.krate, d.index]), self.serialize_to_sexpr(r)] for d, r in vr.items()] if vr else []
        out.extend([self._sym(":variable_ranges"), var_ranges])
        return out

    def _serialize_EinsteinIR(self, node) -> list:
        """Serialize Einstein (rvalue). Name/defid on binding."""
        clauses_sexpr = [self.serialize_to_sexpr(c) for c in (getattr(node, 'clauses', None) or [])]
        core = [self._sym("einstein-value"), self._sym(":clauses"), clauses_sexpr]
        if getattr(node, 'shape', None):
            core.extend([self._sym(":shape"), [self.serialize_to_sexpr(s) for s in node.shape]])
        if getattr(node, 'element_type', None) is not None:
            core.extend([self._sym(":element_type"), self._serialize_type(node.element_type)])
        return core

    def _serialize_LoopStructure(self, loop) -> list:
        """Serialize loop structure: (loop (variable "name" :defid [...]) iterable)."""
        iterable = self.serialize_to_sexpr(loop.iterable)
        var = loop.variable
        if hasattr(var, "accept"):
            var_sexpr = self.serialize_to_sexpr(var)
        else:
            var_sexpr = [self._sym("variable"), str(var)]
        return [self._sym("loop"), var_sexpr, iterable]
    
    def _serialize_BindingIR(self, node) -> list:
        """Serialize binding: (binding name expr type :defid :loc). Single format for all BindingIR."""
        expr = self.serialize_to_sexpr(node.expr)
        typ = self._serialize_type(node.type_info) if getattr(node, 'type_info', None) else [self._sym("nil")]
        core = [self._sym("binding"), node.name, expr, typ]
        if getattr(node, 'defid', None) is not None:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core
    
    # === Lowered IR (replace Einstein/Comprehension/Reduction when lowering succeeds) ===
    
    def _serialize_LoweredEinsteinClauseIR(self, node) -> list:
        """Serialize LoweredEinsteinClauseIR: (lowered-einstein-clause :body ... :loops ...) - per-clause"""
        body = self.serialize_to_sexpr(node.body)
        loops = [self._serialize_LoopStructure(loop) for loop in node.loops]
        loop_var_ranges = []
        for loop in node.loops:
            d = getattr(loop.variable, "defid", None)
            if d is not None:
                loop_var_ranges.append([self._brackets([d.krate, d.index]), self.serialize_to_sexpr(loop.iterable)])
        bindings = [self.serialize_to_sexpr(b) for b in node.bindings]
        guards = [self.serialize_to_sexpr(g.condition) for g in node.guards]
        red_ranges = node.reduction_ranges or {}
        red_defids_str = ",".join(f"{d.krate}:{d.index}" for d in red_ranges.keys())
        red_ranges_sexpr = [[self._brackets([d.krate, d.index]), self._serialize_LoopStructure(loop)]
                           for d, loop in red_ranges.items()]
        indices = getattr(node, "indices", None) or []
        indices_sexpr = [self.serialize_to_sexpr(idx) for idx in indices]
        core = [self._sym("lowered-einstein-clause"),
                self._sym(":body"), body,
                self._sym(":loops"), loops,
                self._sym(":loop_var_ranges"), loop_var_ranges,
                self._sym(":bindings"), bindings,
                self._sym(":guards"), guards,
                self._sym(":reduction_ranges"), red_ranges_sexpr,
                self._sym(":indices"), indices_sexpr]
        if red_defids_str:
            core.extend([self._sym(":reduction_loop_defids"), red_defids_str])
        return core

    def _serialize_LoweredEinsteinIR(self, node) -> list:
        """Serialize LoweredEinsteinIR: (lowered-einstein :shape ... :element_type ... :items (...))"""
        shape = [self.serialize_to_sexpr(s) for s in node.shape] if node.shape else []
        elem_type = self._serialize_type(node.element_type) if node.element_type else [self._sym("nil")]
        items = [self.serialize_to_sexpr(clause) for clause in node.items]
        return [self._sym("lowered-einstein"),
                self._sym(":shape"), shape,
                self._sym(":element_type"), elem_type,
                self._sym(":items"), items]
    
    def _serialize_LoweredComprehensionIR(self, node) -> list:
        """Serialize LoweredComprehensionIR: (lowered-comprehension :body ... :loops ... :bindings ... :guards ...)"""
        body = self.serialize_to_sexpr(node.body)
        loops = [self._serialize_LoopStructure(loop) for loop in node.loops]
        loop_var_ranges = []
        for loop in node.loops:
            d = getattr(loop.variable, "defid", None)
            if d is not None:
                loop_var_ranges.append([self._brackets([d.krate, d.index]), self.serialize_to_sexpr(loop.iterable)])
        bindings = [self.serialize_to_sexpr(b) for b in node.bindings]
        guards = [self.serialize_to_sexpr(g.condition) for g in node.guards]
        core = [self._sym("lowered-comprehension"),
                self._sym(":body"), body,
                self._sym(":loops"), loops,
                self._sym(":loop_var_ranges"), loop_var_ranges,
                self._sym(":bindings"), bindings,
                self._sym(":guards"), guards]
        return self._add_expr_metadata(node, core)
    
    def _serialize_LoweredReductionIR(self, node) -> list:
        """Serialize LoweredReductionIR: (lowered-reduction op :body ... :loops ...)"""
        body = self.serialize_to_sexpr(node.body)
        loops = [self._serialize_LoopStructure(loop) for loop in node.loops]
        loop_var_ranges = []
        for loop in node.loops:
            d = getattr(loop.variable, "defid", None)
            if d is not None:
                loop_var_ranges.append([self._brackets([d.krate, d.index]), self.serialize_to_sexpr(loop.iterable)])
        bindings = [self.serialize_to_sexpr(b) for b in node.bindings]
        guards = [self.serialize_to_sexpr(g.condition) for g in node.guards]
        def _defid_str(loop):
            d = getattr(loop.variable, "defid", None)
            return f"{d.krate}:{d.index}" if d else "?"
        loop_defids_str = ",".join(_defid_str(loop) for loop in node.loops)
        core = [self._sym("lowered-reduction"), node.operation,
                self._sym(":loop_defids"), loop_defids_str,
                self._sym(":loop_var_ranges"), loop_var_ranges,
                self._sym(":body"), body,
                self._sym(":loops"), loops,
                self._sym(":bindings"), bindings,
                self._sym(":guards"), guards]
        return self._add_expr_metadata(node, core)
    
    def _serialize_param(self, p) -> list:
        """Serialize one parameter: (param "name" :defid [krate index] :ty type)."""
        name = getattr(p, 'name', str(p))
        param = [self._sym("param"), name]
        if self.include_location and getattr(p, 'location', None):
            loc = p.location
            param.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        if getattr(p, 'defid', None) is not None:
            param.extend([self._sym(":defid"), self._brackets([p.defid.krate, p.defid.index])])
        if self.include_type_info and getattr(p, 'param_type', None) is not None:
            param.extend([self._sym(":ty"), self._serialize_type(p.param_type)])
        return param
    
    # === Block Expressions ===
    
    def _serialize_BlockExpressionIR(self, node) -> list:
        """Serialize block: (block (statements...) final_expr)"""
        stmts = [self.serialize_to_sexpr(stmt) for stmt in node.statements]
        final = self.serialize_to_sexpr(node.final_expr) if node.final_expr else [self._sym("nil")]
        core = [self._sym("block"), stmts, final]
        return self._add_expr_metadata(node, core)
    
    # === If Expressions ===
    
    def _serialize_IfExpressionIR(self, node) -> list:
        """Serialize if: (if condition then else)"""
        condition = self.serialize_to_sexpr(node.condition)
        then_branch = self.serialize_to_sexpr(node.then_expr)
        else_branch = self.serialize_to_sexpr(node.else_expr) if node.else_expr else [self._sym("nil")]
        core = [self._sym("if"), condition, then_branch, else_branch]
        return self._add_expr_metadata(node, core)
    
    # === Tuple Expressions ===
    
    def _serialize_TupleExpressionIR(self, node) -> list:
        """Serialize tuple: (tuple (elements...))"""
        elements = [self.serialize_to_sexpr(elem) for elem in node.elements]
        core = [self._sym("tuple"), elements]
        return self._add_expr_metadata(node, core)
    
    def _serialize_TupleAccessIR(self, node) -> list:
        """Serialize tuple access: (tuple-access tuple_expr index)"""
        tuple_expr = self.serialize_to_sexpr(node.tuple_expr)
        core = [self._sym("tuple-access"), tuple_expr, node.index]
        return self._add_expr_metadata(node, core)
    
    # === Lambda/Arrow Expressions ===
    
    def _serialize_LambdaIR(self, node) -> list:
        """Serialize lambda: (lambda ((param "x" :defid [...])...) body :defid [...])"""
        params = [self._serialize_param(p) for p in node.parameters]
        body = self.serialize_to_sexpr(node.body)
        core = [self._sym("lambda"), params, body]
        if getattr(node, 'defid', None) is not None:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        return self._add_expr_metadata(node, core)
    
    def _serialize_PipelineExpressionIR(self, node) -> list:
        """Serialize pipeline expression: (pipeline-expression left right operator)"""
        left = self.serialize_to_sexpr(node.left)
        right = self.serialize_to_sexpr(node.right)
        core = [self._sym("pipeline-expression"), left, right, getattr(node, 'operator', '|>')]
        return self._add_expr_metadata(node, core)
    
    # === Reduction Expressions ===
    
    def _serialize_ReductionExpressionIR(self, node) -> list:
        """Serialize reduction: (reduction op (vars...) body :loop_var_ranges ...)"""
        body = self.serialize_to_sexpr(node.body)
        core = [self._sym("reduction"), node.operation, node.loop_var_names, body]
        lvr = getattr(node, 'loop_var_ranges', None) or {}
        if lvr:
            loop_ranges = [[self._brackets([d.krate, d.index]), self.serialize_to_sexpr(r)] for d, r in lvr.items()]
            core.extend([self._sym(":loop_var_ranges"), loop_ranges])
        return self._add_expr_metadata(node, core)

    def _serialize_WhereExpressionIR(self, node) -> list:
        """Serialize where expression: (where-expression expr (constraints...))"""
        expr = self.serialize_to_sexpr(node.expr)
        constraints = [self.serialize_to_sexpr(c) for c in (node.constraints or [])]
        core = [self._sym("where-expression"), expr, constraints]
        return self._add_expr_metadata(node, core)

    def _serialize_MatchExpressionIR(self, node) -> list:
        """Serialize match: (match-expression scrutinee (match-arm pattern body)...)"""
        scrutinee = self.serialize_to_sexpr(node.scrutinee)
        arms = []
        for a in node.arms:
            arms.append([self._sym("match-arm"), self._serialize_pattern(a.pattern), self.serialize_to_sexpr(a.body)])
        core = [self._sym("match-expression"), scrutinee, arms]
        return self._add_expr_metadata(node, core)

    def _serialize_pattern(self, node: Any) -> list:
        """Serialize pattern IR node (LiteralPatternIR, IdentifierPatternIR, etc.)."""
        if node is None:
            return [self._sym("nil")]
        method_name = f"_serialize_{type(node).__name__}"
        method = getattr(self, method_name, None)
        if method is not None:
            return method(node)
        return [self._sym(type(node).__name__), self._sym("...")]

    def _serialize_LiteralPatternIR(self, node) -> list:
        val = node.value
        if isinstance(val, bool):
            val = self._sym("true" if val else "false")
        core = [self._sym("literal-pattern"), val]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_IdentifierPatternIR(self, node) -> list:
        core = [self._sym("identifier-pattern"), node.name]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        if getattr(node, 'defid', None) is not None:
            core.extend([self._sym(":defid"), self._brackets([node.defid.krate, node.defid.index])])
        return core

    def _serialize_WildcardPatternIR(self, node) -> list:
        core = [self._sym("wildcard-pattern")]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_TuplePatternIR(self, node) -> list:
        patterns = [self._serialize_pattern(p) for p in node.patterns]
        core = [self._sym("tuple-pattern"), patterns]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_ArrayPatternIR(self, node) -> list:
        patterns = [self._serialize_pattern(p) for p in node.patterns]
        core = [self._sym("array-pattern"), patterns]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_RestPatternIR(self, node) -> list:
        inner = self._serialize_pattern(node.pattern)
        core = [self._sym("rest-pattern"), inner]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_GuardPatternIR(self, node) -> list:
        inner = self._serialize_pattern(node.inner_pattern)
        guard = self.serialize_to_sexpr(node.guard_expr)
        core = [self._sym("guard-pattern"), inner, guard]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_OrPatternIR(self, node) -> list:
        alts = [self._serialize_pattern(a) for a in node.alternatives]
        core = [self._sym("or-pattern"), alts]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_ConstructorPatternIR(self, node) -> list:
        patterns = [self._serialize_pattern(p) for p in node.patterns]
        core = [self._sym("constructor-pattern"), node.constructor_name, patterns]
        if node.is_struct_literal:
            core.extend([self._sym(":struct"), self._sym("true")])
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_BindingPatternIR(self, node) -> list:
        ident = self._serialize_pattern(node.identifier_pattern)
        inner = self._serialize_pattern(node.inner_pattern)
        core = [self._sym("binding-pattern"), ident, inner]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_RangePatternIR(self, node) -> list:
        core = [self._sym("range-pattern"), node.start, node.end,
                self._sym("inclusive" if node.inclusive else "exclusive")]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        return core

    def _serialize_FunctionValueIR(self, node) -> list:
        """Serialize function value (rvalue). Name/defid on binding; format (function-value (params) body :return_type :loc)."""
        params = [self._serialize_param(p) for p in node.parameters]
        body = self.serialize_to_sexpr(node.body) if node.body else [self._sym("nil")]
        core = [self._sym("function-value"), params, body]
        if self.include_location and getattr(node, 'location', None):
            loc = node.location
            core.extend([self._sym(":loc"), [loc.file, loc.line, loc.column]])
        if self.include_type_info and getattr(node, 'return_type', None) is not None:
            core.extend([self._sym(":return_type"), self._serialize_type(node.return_type)])
        return core

    # === Programs ===
    
    def _serialize_ProgramIR(self, node) -> list:
        """Serialize program: (program :bindings (...) :source_files (...))"""
        bindings_sexpr = [self.serialize_to_sexpr(s) for s in node.statements]
        core = [self._sym("program"), self._sym(":bindings"), bindings_sexpr]
        sf = getattr(node, "source_files", None) or {}
        if sf:
            items = [[k, v] for k, v in sf.items()]
            core.extend([self._sym(":source_files"), items])
        return core


# Convenience function
def to_sexpr(node: IRNode, include_location: bool = False, include_type_info: bool = False) -> str:
    """Alias for serialize_ir"""
    return serialize_ir(node, include_location=include_location, include_type_info=include_type_info)


def _default_loc():
    from ..shared.source_location import SourceLocation
    return SourceLocation(file="", line=0, column=0)


def _sym_val(x: Any) -> str:
    if sexpdata and isinstance(x, sexpdata.Symbol):
        return x.value()
    if isinstance(x, str):
        return x
    return str(x)


def _parse_location(sexpr: Any):
    from ..shared.source_location import SourceLocation
    if sexpr is None:
        return None
    if isinstance(sexpr, list) and len(sexpr) >= 3:
        parts = sexpr
    elif sexpdata and isinstance(sexpr, sexpdata.Brackets):
        parts = getattr(sexpr, "I", list(sexpr))
    else:
        return None
    try:
        file_s = parts[0]
        file_str = str(file_s) if not isinstance(file_s, str) else file_s
        if file_str.startswith('"') and file_str.endswith('"'):
            file_str = file_str.strip('"')
        line = int(parts[1]) if not (sexpdata and isinstance(parts[1], sexpdata.Symbol)) else int(_sym_val(parts[1]))
        col = int(parts[2]) if not (sexpdata and isinstance(parts[2], sexpdata.Symbol)) else int(_sym_val(parts[2]))
        return SourceLocation(file=file_str, line=line, column=col)
    except (TypeError, ValueError):
        return None


def _parse_defid(sexpr: Any):
    from ..shared.defid import DefId
    if sexpr is None:
        return None
    if sexpdata and isinstance(sexpr, sexpdata.Brackets):
        parts = getattr(sexpr, "I", list(sexpr))
    elif isinstance(sexpr, list) and len(sexpr) >= 2:
        parts = sexpr
    else:
        try:
            parts = list(sexpr)
        except Exception:
            return None
        if len(parts) < 2:
            return None
    try:
        krate = int(parts[0]) if not (sexpdata and isinstance(parts[0], sexpdata.Symbol)) else int(_sym_val(parts[0]))
        index = int(parts[1]) if not (sexpdata and isinstance(parts[1], sexpdata.Symbol)) else int(_sym_val(parts[1]))
        return DefId(krate=krate, index=index)
    except (TypeError, ValueError):
        return None


def _plist(tail: list) -> tuple:
    out = {}
    pos = []
    i = 0
    while i < len(tail):
        x = tail[i]
        key = _sym_val(x) if (sexpdata and isinstance(x, sexpdata.Symbol)) else str(x)
        if isinstance(key, str) and key.startswith(":"):
            while i + 1 < len(tail):
                k = _sym_val(tail[i]) if (sexpdata and isinstance(tail[i], sexpdata.Symbol)) else str(tail[i])
                if isinstance(k, str) and k.startswith(":"):
                    out[k] = tail[i + 1]
                    i += 2
                else:
                    break
            break
        pos.append(x)
        i += 1
    return (pos, out)


class IRDeserializer:
    def __init__(self):
        self._loc = _default_loc()

    def _loc_from_opts(self, opts: dict) -> Any:
        loc = _parse_location(opts.get(":loc"))
        return loc if loc is not None else self._loc

    def deserialize(self, sexpr: Any) -> Any:
        if sexpr is None:
            return None
        if isinstance(sexpr, list):
            if not sexpr:
                return None
            tag = _sym_val(sexpr[0])
            if tag == "nil":
                return None
            tail = sexpr[1:]
            method = getattr(self, f"_deserialize_{tag.replace('-', '_')}", None)
            if method is not None:
                return method(tag, tail, sexpr)
            return self._deserialize_expr(tag, tail, sexpr)
        if isinstance(sexpr, (int, float)):
            return sexpr
        if sexpdata and isinstance(sexpr, sexpdata.Symbol):
            v = sexpr.value()
            if v == "true":
                return True
            if v == "false":
                return False
            return v
        if isinstance(sexpr, str):
            return sexpr
        return sexpr

    def _opts_type(self, tail: list, skip: int) -> Any:
        pos, opts = _plist(tail[skip:])
        return self._deserialize_type(opts.get(":inferred_type"))

    def _parse_shape_info_raw(self, raw: Any) -> Optional[Any]:
        if raw is None or not isinstance(raw, list) or not raw:
            return None
        out = []
        for x in raw:
            if sexpdata and isinstance(x, sexpdata.Symbol):
                v = _sym_val(x)
                if v == "?" or v is None:
                    out.append(None)
                else:
                    try:
                        out.append(int(v))
                    except (TypeError, ValueError):
                        out.append(None)
            elif x is None or (isinstance(x, str) and x.strip() == "?"):
                out.append(None)
            elif isinstance(x, int):
                out.append(x)
            elif isinstance(x, float):
                out.append(None)
            else:
                try:
                    xi = int(x)
                    out.append(xi)
                except (TypeError, ValueError):
                    out.append(None)
        return tuple(out)

    def _opts_shape_info(self, tail: list, skip: int) -> Optional[Any]:
        _, opts = _plist(tail[skip:])
        return self._parse_shape_info_raw(opts.get(":shape_info"))

    def _deserialize_literal(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LiteralIR
        from ..shared.types import PrimitiveType
        _, opts = _plist(tail[2:]) if len(tail) > 2 else ([], {})
        loc = self._loc_from_opts(opts)
        val = tail[0]
        type_sym = _sym_val(tail[1]) if len(tail) > 1 else "i32"
        if sexpdata and isinstance(val, sexpdata.Symbol):
            v = val.value()
            val = True if v == "true" else False if v == "false" else v
        elif isinstance(val, list) and val and _sym_val(val[0]) == "literal":
            des = self.deserialize(val)
            val = des.value if hasattr(des, "value") else des
        if isinstance(type_sym, list):
            type_sym = _sym_val(type_sym[1]) if len(type_sym) > 1 else "i32"
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return LiteralIR(value=val, location=loc, type_info=ty)

    def _deserialize_variable(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import IdentifierIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        name = tail[0]
        name = _sym_val(name[0]) if isinstance(name, list) and name else str(name).strip('"')
        opts = _plist(tail[1:])[1]
        defid = _parse_defid(opts.get(":defid"))
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return IdentifierIR(name=name, location=loc, defid=defid, type_info=ty)

    def _deserialize_binary_op(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import BinaryOpIR
        from ..shared.types import BinaryOp
        _, opts = _plist(tail[3:])
        loc = self._loc_from_opts(opts)
        op = _sym_val(tail[0])
        left = self.deserialize(tail[1])
        right = self.deserialize(tail[2])
        ty = self._opts_type(tail, 3)
        op_enum = next((b for b in BinaryOp if b.value == op), op)
        return BinaryOpIR(operator=op_enum, left=left, right=right, location=loc, type_info=ty)

    def _deserialize_unary_op(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import UnaryOpIR
        from ..shared.types import UnaryOp
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        op = _sym_val(tail[0])
        operand = self.deserialize(tail[1])
        ty = self._opts_type(tail, 2)
        op_enum = next((u for u in UnaryOp if u.value == op), op)
        return UnaryOpIR(operator=op_enum, operand=operand, location=loc, type_info=ty)

    def _deserialize_block(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import BlockExpressionIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        stmts_sexpr = tail[0] if tail and isinstance(tail[0], list) else []
        stmts = [self.deserialize(s) for s in stmts_sexpr]
        final = self.deserialize(tail[1]) if len(tail) > 1 else None
        ty = self._opts_type(tail, 2)
        return BlockExpressionIR(statements=stmts, location=loc, final_expr=final, type_info=ty)

    def _deserialize_if(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import IfExpressionIR
        _, opts = _plist(tail[3:])
        loc = self._loc_from_opts(opts)
        cond = self.deserialize(tail[0])
        then_e = self.deserialize(tail[1])
        else_e = self.deserialize(tail[2]) if len(tail) > 2 else None
        ty = self._opts_type(tail, 3)
        return IfExpressionIR(condition=cond, then_expr=then_e, else_expr=else_e, location=loc, type_info=ty)

    def _deserialize_function_call(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import FunctionCallIR, IdentifierIR
        pos, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        callee, args = None, []
        if pos:
            first = pos[0]
            if isinstance(first, list) and first and _sym_val(first[0]) == "callee":
                callee = self.deserialize(first[1]) if len(first) > 1 else None
                if len(pos) > 1 and isinstance(pos[1], list):
                    args = [self.deserialize(a) for a in pos[1]]
            else:
                name = first if isinstance(first, str) else (first.value() if sexpdata and isinstance(first, sexpdata.Symbol) else str(first))
                if isinstance(name, str) and name.startswith('"') and name.endswith('"'):
                    name = name.strip('"')
                if len(pos) > 1 and isinstance(pos[1], list):
                    args = [self.deserialize(a) for a in pos[1]]
                func_defid = _parse_defid(opts.get(":function-defid"))
                callee = IdentifierIR(name=name or "", location=loc, defid=func_defid)
        module_path_raw = opts.get(":module_path")
        module_path = tuple(module_path_raw) if isinstance(module_path_raw, list) else None
        ty = self._deserialize_type(opts.get(":inferred_type"))
        if callee is None:
            callee = IdentifierIR(name="", location=loc, defid=None)
        return FunctionCallIR(callee_expr=callee, location=loc, arguments=args, module_path=module_path, type_info=ty)

    def _deserialize_builtin_call(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import BuiltinCallIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        name = _sym_val(tail[0]) if tail else ""
        args_list = tail[1] if len(tail) > 1 and isinstance(tail[1], list) else []
        args = [self.deserialize(a) for a in args_list]
        opts = _plist(tail[2:])[1]
        defid = _parse_defid(opts.get(":defid"))
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return BuiltinCallIR(builtin_name=name, args=args, location=loc, defid=defid, type_info=ty)

    def _deserialize_array_literal(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import ArrayLiteralIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        elts = tail[0] if tail and isinstance(tail[0], list) else []
        elements = [self.deserialize(e) for e in elts]
        ty = self._opts_type(tail, 1)
        shape_info = self._opts_shape_info(tail, 1)
        return ArrayLiteralIR(elements=elements, location=loc, type_info=ty, shape_info=shape_info)

    def _deserialize_array_comprehension(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import ArrayComprehensionIR, IndexVarIR
        pos, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        body = self.deserialize(pos[0]) if pos else None
        loop_vars_sexpr = opts.get(":loop_vars")
        if isinstance(loop_vars_sexpr, list):
            loop_vars = [self.deserialize(v) for v in loop_vars_sexpr]
        else:
            vars_list = opts.get(":vars") or []
            loop_vars = [IndexVarIR(name=str(n).strip('"') if isinstance(n, str) else str(n), location=loc, defid=None) for n in vars_list]
        ranges_sexpr = opts.get(":ranges") or []
        ranges = [self.deserialize(r) for r in ranges_sexpr] if isinstance(ranges_sexpr, list) else []
        constraints_sexpr = opts.get(":constraints") or []
        constraints = [self.deserialize(c) for c in constraints_sexpr] if isinstance(constraints_sexpr, list) else []
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return ArrayComprehensionIR(body=body, loop_vars=loop_vars, ranges=ranges, constraints=constraints, location=loc, type_info=ty)

    def _deserialize_range(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LiteralIR, RangeIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        start = self.deserialize(tail[0]) if tail else None
        end = self.deserialize(tail[1]) if len(tail) > 1 else None
        inclusive = _sym_val(opts.get(":inclusive", "")) == "true" if ":inclusive" in opts else False
        ty = self._opts_type(tail, 2)
        return RangeIR(start=start or LiteralIR(0, loc), end=end or LiteralIR(0, loc), location=loc,
                       inclusive=inclusive, type_info=ty)

    def _deserialize_rectangular_access(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import RectangularAccessIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        array = self.deserialize(tail[0])
        idx_list = tail[1] if len(tail) > 1 and isinstance(tail[1], list) else []
        indices = [self.deserialize(i) for i in idx_list]
        ty = self._opts_type(tail, 2)
        return RectangularAccessIR(array=array, indices=indices, location=loc, type_info=ty)

    def _name_from_tail(self, tail: list) -> str:
        name = tail[0]
        return _sym_val(name[0]) if isinstance(name, list) and name else str(name).strip('"')

    def _deserialize_index_var(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import IndexVarIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        name = self._name_from_tail(tail)
        opts = _plist(tail[1:])[1]
        defid = _parse_defid(opts.get(":defid"))
        range_ir = self.deserialize(opts.get(":range")) if ":range" in opts else None
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return IndexVarIR(name=name, location=loc, defid=defid, range_ir=range_ir, type_info=ty)

    def _deserialize_index_rest(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import IndexRestIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        name = self._name_from_tail(tail)
        opts = _plist(tail[1:])[1]
        defid = _parse_defid(opts.get(":defid"))
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return IndexRestIR(name=name, location=loc, defid=defid, type_info=ty)

    def _deserialize_member_access(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import MemberAccessIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        obj = self.deserialize(tail[0])
        member = tail[1] if len(tail) > 1 else ""
        member = _sym_val(member[0]) if isinstance(member, list) and member else member
        if isinstance(member, int):
            pass
        elif sexpdata and isinstance(member, sexpdata.Symbol):
            member = _sym_val(member)
        elif isinstance(member, str):
            try:
                member = int(member)
            except ValueError:
                pass
        ty = self._opts_type(tail, 2)
        return MemberAccessIR(object=obj, member=member, location=loc, type_info=ty)

    def _deserialize_cast(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import CastExpressionIR
        from ..shared.types import PrimitiveType
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        expr = self.deserialize(tail[0])
        raw_type = tail[1] if len(tail) > 1 else None
        if isinstance(raw_type, list):
            target_type = self._deserialize_type(raw_type)
        elif raw_type is not None:
            target_type = PrimitiveType(_sym_val(raw_type))
        else:
            target_type = PrimitiveType("i32")
        ty = self._opts_type(tail, 2)
        return CastExpressionIR(expr=expr, target_type=target_type, location=loc, type_info=ty)

    def _deserialize_tuple(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import TupleExpressionIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        elts = tail[0] if tail and isinstance(tail[0], list) else []
        elements = [self.deserialize(e) for e in elts]
        ty = self._opts_type(tail, 1)
        shape_info = self._opts_shape_info(tail, 1)
        return TupleExpressionIR(elements=elements, location=loc, type_info=ty, shape_info=shape_info)

    def _deserialize_tuple_access(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import TupleAccessIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        tup = self.deserialize(tail[0])
        idx = tail[1] if len(tail) > 1 else 0
        if isinstance(idx, list):
            idx = int(_sym_val(idx[0])) if idx else 0
        else:
            idx = int(idx)
        ty = self._opts_type(tail, 2)
        return TupleAccessIR(tuple_expr=tup, index=idx, location=loc, type_info=ty)

    def _deserialize_lambda(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LiteralIR, LambdaIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        params_sexpr = tail[0] if tail and isinstance(tail[0], list) else []
        params = [self._deserialize_param(p) for p in params_sexpr]
        body = self.deserialize(tail[1]) if len(tail) > 1 else None
        body = body if body is not None else LiteralIR(value=None, location=loc)
        ty = self._opts_type(tail, 2)
        return LambdaIR(parameters=params, body=body, location=loc, type_info=ty)

    def _deserialize_pipeline_expression(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import PipelineExpressionIR
        left = self.deserialize(tail[0]) if tail else None
        right = self.deserialize(tail[1]) if len(tail) > 1 else None
        operator = tail[2] if len(tail) > 2 else "|>"
        if hasattr(operator, "value"):
            operator = operator.value()
        elif isinstance(operator, list) and operator and hasattr(operator[0], "value"):
            operator = operator[0].value()
        else:
            operator = str(operator) if operator else "|>"
        _, opts = _plist(tail[3:])
        loc = self._loc_from_opts(opts)
        ty = self._opts_type(tail, 3)
        shape_info = self._opts_shape_info(tail, 3)
        return PipelineExpressionIR(left=left, right=right, operator=operator, location=loc, type_info=ty, shape_info=shape_info)

    def _deserialize_match_expression(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import MatchExpressionIR, MatchArmIR
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        scrutinee = self.deserialize(tail[0]) if tail else None
        arms_sexpr = tail[1] if len(tail) > 1 and isinstance(tail[1], list) else []
        arm_list = [self._deserialize_match_arm_item(a) for a in arms_sexpr if isinstance(a, list) and len(a) >= 3 and _sym_val(a[0]) == "match-arm"]
        arms = [x for x in arm_list if x is not None]
        ty = self._opts_type(tail, 2)
        return MatchExpressionIR(scrutinee=scrutinee, arms=arms, location=loc, type_info=ty)

    def _deserialize_match_arm_item(self, sexpr: list) -> Any:
        from ..ir.nodes import MatchArmIR
        if not isinstance(sexpr, list) or len(sexpr) < 3 or _sym_val(sexpr[0]) != "match-arm":
            return None
        pattern = self.deserialize(sexpr[1])
        body = self.deserialize(sexpr[2])
        return MatchArmIR(pattern=pattern, body=body)

    def _deserialize_match_arm(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import MatchArmIR
        pattern = self.deserialize(tail[0]) if tail else None
        body = self.deserialize(tail[1]) if len(tail) > 1 else None
        return MatchArmIR(pattern=pattern, body=body)

    def _deserialize_literal_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LiteralPatternIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        val = tail[0]
        if sexpdata and isinstance(val, sexpdata.Symbol):
            v = val.value()
            val = True if v == "true" else False if v == "false" else v
        return LiteralPatternIR(value=val, location=loc)

    def _deserialize_identifier_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import IdentifierPatternIR
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        name = _sym_val(tail[0]) if tail else ""
        if isinstance(name, list):
            name = _sym_val(name[0]) if name else ""
        name = str(name).strip('"') if isinstance(name, str) else str(name)
        defid = _parse_defid(opts.get(":defid"))
        return IdentifierPatternIR(name=name, location=loc, defid=defid)

    def _deserialize_wildcard_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import WildcardPatternIR
        _, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        return WildcardPatternIR(location=loc)

    def _deserialize_tuple_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import TuplePatternIR
        patterns_sexpr = tail[0] if tail and isinstance(tail[0], list) else []
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        patterns = [self.deserialize(p) for p in patterns_sexpr]
        return TuplePatternIR(patterns=patterns, location=loc)

    def _deserialize_array_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import ArrayPatternIR
        patterns_sexpr = tail[0] if tail and isinstance(tail[0], list) else []
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        patterns = [self.deserialize(p) for p in patterns_sexpr]
        return ArrayPatternIR(patterns=patterns, location=loc)

    def _deserialize_rest_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import RestPatternIR
        inner = self.deserialize(tail[0]) if tail else None
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        return RestPatternIR(pattern=inner, location=loc)

    def _deserialize_guard_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import GuardPatternIR
        inner = self.deserialize(tail[0]) if tail else None
        guard = self.deserialize(tail[1]) if len(tail) > 1 else None
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        return GuardPatternIR(inner_pattern=inner, guard_expr=guard, location=loc)

    def _deserialize_or_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import OrPatternIR
        alts_raw = tail[0] if tail and isinstance(tail[0], list) else []
        alts = [self.deserialize(a) for a in alts_raw]
        _, opts = _plist(tail[1:])
        loc = self._loc_from_opts(opts)
        return OrPatternIR(alternatives=alts, location=loc)

    def _deserialize_constructor_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import ConstructorPatternIR
        name = str(tail[0]).strip('"') if tail else ""
        patterns_raw = tail[1] if len(tail) > 1 and isinstance(tail[1], list) else []
        patterns = [self.deserialize(p) for p in patterns_raw]
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        is_struct = _sym_val(opts.get(":struct", "")) == "true" if ":struct" in opts else False
        return ConstructorPatternIR(constructor_name=name, patterns=patterns,
                                    is_struct_literal=is_struct, location=loc)

    def _deserialize_binding_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import BindingPatternIR, IdentifierPatternIR
        ident_sexpr = tail[0] if tail else None
        inner = self.deserialize(tail[1]) if len(tail) > 1 else None
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        ident = self.deserialize(ident_sexpr) if ident_sexpr is not None else None
        if not isinstance(ident, IdentifierPatternIR):
            raise ValueError(f"binding-pattern requires identifier-pattern as first element, got {type(ident).__name__}")
        return BindingPatternIR(identifier_pattern=ident, inner_pattern=inner, location=loc)

    def _deserialize_range_pattern(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import RangePatternIR
        start = tail[0] if tail else 0
        end = tail[1] if len(tail) > 1 else 0
        inclusive = _sym_val(tail[2]) == "inclusive" if len(tail) > 2 else True
        _, opts = _plist(tail[3:])
        loc = self._loc_from_opts(opts)
        return RangePatternIR(start=start, end=end, inclusive=inclusive, location=loc)

    def _deserialize_einstein_value(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import EinsteinIR
        from ..shared.source_location import SourceLocation
        _, opts = _plist(tail)
        clauses_sexpr = opts.get(":clauses")
        clauses = [self.deserialize(c) for c in clauses_sexpr] if isinstance(clauses_sexpr, list) else []
        shape_sexpr = opts.get(":shape")
        shape = [self.deserialize(s) for s in shape_sexpr] if isinstance(shape_sexpr, list) else None
        element_type = self._deserialize_type(opts.get(":element_type"))
        loc = self._loc_from_opts(opts) or SourceLocation("", 0, 0)
        return EinsteinIR(clauses=clauses, shape=shape, element_type=element_type, location=loc)

    def _deserialize_binding(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import BindingIR
        if len(tail) < 3:
            return None
        name = tail[0]
        name = str(name).strip('"') if isinstance(name, str) else (_sym_val(name[0]) if isinstance(name, list) and name else "")
        expr = self.deserialize(tail[1])
        typ = self._deserialize_type(tail[2]) if len(tail) > 2 and isinstance(tail[2], list) else None
        _, opts = _plist(tail[3:])
        defid = _parse_defid(opts.get(":defid"))
        loc = self._loc_from_opts(opts)
        return BindingIR(name=name, expr=expr, type_info=typ, location=loc, defid=defid)

    def _deserialize_function_value(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import FunctionValueIR
        if len(tail) < 2:
            return None
        params_sexpr = tail[0] if isinstance(tail[0], list) else []
        params = [self._deserialize_param(p) for p in params_sexpr]
        body = self.deserialize(tail[1]) if len(tail) > 1 else None
        _, opts = _plist(tail[2:])
        loc = self._loc_from_opts(opts)
        return_type = self._deserialize_type(opts.get(":return_type"))
        return FunctionValueIR(parameters=params, body=body, location=loc, return_type=return_type)

    def _deserialize_program(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import ProgramIR
        _, opts = _plist(tail)
        bindings_sexpr = opts.get(":bindings")
        stmts = []
        if isinstance(bindings_sexpr, list):
            for s in bindings_sexpr:
                b = self.deserialize(s)
                if b is not None:
                    stmts.append(b)
        sf = {}
        for item in opts.get(":source_files") or []:
            if isinstance(item, list) and len(item) >= 2:
                sf[item[0]] = item[1]
        return ProgramIR(statements=stmts, source_files=sf, modules=[])

    def _deserialize_loop_structure(self, sexpr: Any) -> Any:
        from ..ir.nodes import LoopStructure
        if not isinstance(sexpr, list) or len(sexpr) < 3 or _sym_val(sexpr[0]) != "loop":
            return None
        var_sexpr = sexpr[1]
        iterable = self.deserialize(sexpr[2])
        variable = self.deserialize(var_sexpr)
        if variable is None or iterable is None:
            return None
        return LoopStructure(variable=variable, iterable=iterable)

    def _deserialize_lowered_reduction(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LoweredReductionIR, GuardCondition, BindingIR
        pos, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        operation = _sym_val(pos[0]) if pos else "sum"
        body = self.deserialize(opts.get(":body"))
        loops_sexpr = opts.get(":loops")
        loops = []
        if isinstance(loops_sexpr, list):
            for s in loops_sexpr:
                if isinstance(s, list) and _sym_val(s[0]) == "loop":
                    loop = self._deserialize_loop_structure(s)
                    if loop is not None:
                        loops.append(loop)
        bindings_sexpr = opts.get(":bindings")
        bindings = []
        if isinstance(bindings_sexpr, list):
            for s in bindings_sexpr:
                b = self.deserialize(s)
                if isinstance(b, BindingIR):
                    bindings.append(b)
        guards_sexpr = opts.get(":guards")
        guards = []
        if isinstance(guards_sexpr, list):
            for g in guards_sexpr:
                cond = self.deserialize(g)
                if cond is not None:
                    guards.append(GuardCondition(condition=cond))
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return LoweredReductionIR(body=body, operation=operation, loops=loops, bindings=bindings, guards=guards, location=loc, type_info=ty)

    def _deserialize_lowered_comprehension(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LoweredComprehensionIR, GuardCondition, BindingIR
        _, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        body = self.deserialize(opts.get(":body"))
        loops_sexpr = opts.get(":loops")
        loops = []
        if isinstance(loops_sexpr, list):
            for s in loops_sexpr:
                if isinstance(s, list) and _sym_val(s[0]) == "loop":
                    loop = self._deserialize_loop_structure(s)
                    if loop is not None:
                        loops.append(loop)
        bindings_sexpr = opts.get(":bindings")
        bindings = []
        if isinstance(bindings_sexpr, list):
            for s in bindings_sexpr:
                b = self.deserialize(s)
                if isinstance(b, BindingIR):
                    bindings.append(b)
        guards_sexpr = opts.get(":guards")
        guards = []
        if isinstance(guards_sexpr, list):
            for g in guards_sexpr:
                cond = self.deserialize(g)
                if cond is not None:
                    guards.append(GuardCondition(condition=cond))
        ty = self._deserialize_type(opts.get(":inferred_type"))
        shape_info = self._parse_shape_info_raw(opts.get(":shape_info"))
        return LoweredComprehensionIR(body=body, loops=loops, bindings=bindings, guards=guards, location=loc, type_info=ty, shape_info=shape_info)

    def _deserialize_lowered_einstein_clause(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LoweredEinsteinClauseIR, GuardCondition, BindingIR
        _, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        body = self.deserialize(opts.get(":body"))
        loops_sexpr = opts.get(":loops")
        loops = []
        if isinstance(loops_sexpr, list):
            for s in loops_sexpr:
                if isinstance(s, list) and _sym_val(s[0]) == "loop":
                    loop = self._deserialize_loop_structure(s)
                    if loop is not None:
                        loops.append(loop)
        bindings_sexpr = opts.get(":bindings")
        bindings = []
        if isinstance(bindings_sexpr, list):
            for s in bindings_sexpr:
                b = self.deserialize(s)
                if isinstance(b, BindingIR):
                    bindings.append(b)
        guards_sexpr = opts.get(":guards")
        guards = []
        if isinstance(guards_sexpr, list):
            for g in guards_sexpr:
                cond = self.deserialize(g)
                if cond is not None:
                    guards.append(GuardCondition(condition=cond))
        red_ranges = {}
        red_sexpr = opts.get(":reduction_ranges")
        if isinstance(red_sexpr, list):
            for pair in red_sexpr:
                if isinstance(pair, list) and len(pair) >= 2:
                    d = _parse_defid(pair[0])
                    if d is not None:
                        loop = self._deserialize_loop_structure(pair[1])
                        if loop is not None:
                            red_ranges[d] = loop
        indices_sexpr = opts.get(":indices")
        indices = [self.deserialize(s) for s in indices_sexpr] if isinstance(indices_sexpr, list) else []
        return LoweredEinsteinClauseIR(body=body, loops=loops, bindings=bindings, guards=guards, reduction_ranges=red_ranges, indices=indices, location=loc)

    def _deserialize_lowered_einstein(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import LoweredEinsteinIR
        _, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        shape_sexpr = opts.get(":shape")
        shape = [self.deserialize(s) for s in shape_sexpr] if isinstance(shape_sexpr, list) else []
        element_type = self._deserialize_type(opts.get(":element_type"))
        items_sexpr = opts.get(":items")
        items = [self.deserialize(s) for s in items_sexpr] if isinstance(items_sexpr, list) else []
        return LoweredEinsteinIR(items=items, shape=shape, element_type=element_type, location=loc)

    def _deserialize_interpolated_string(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import InterpolatedStringIR
        pos, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        parts_sexpr = pos[0] if pos and isinstance(pos[0], list) else []
        parts = []
        for s in parts_sexpr:
            if isinstance(s, str):
                parts.append(s)
            else:
                part = self.deserialize(s)
                if part is not None:
                    parts.append(part)
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return InterpolatedStringIR(parts=parts, location=loc, type_info=ty)

    def _deserialize_where_expression(self, _tag: str, tail: list, _full: list) -> Any:
        from ..ir.nodes import WhereExpressionIR
        pos, opts = _plist(tail)
        loc = self._loc_from_opts(opts)
        expr = self.deserialize(pos[0]) if len(pos) > 0 else None
        constraints_sexpr = pos[1] if len(pos) > 1 and isinstance(pos[1], list) else []
        constraints = [self.deserialize(c) for c in constraints_sexpr] if constraints_sexpr else []
        ty = self._deserialize_type(opts.get(":inferred_type"))
        return WhereExpressionIR(expr=expr, constraints=constraints, location=loc, type_info=ty)

    def _deserialize_expr(self, tag: str, tail: list, full: list) -> Any:
        raise ValueError(f"Unknown IR tag: {tag}")

    def _deserialize_param(self, sexpr: Any) -> Any:
        from ..ir.nodes import ParameterIR
        if not isinstance(sexpr, list) or len(sexpr) < 2:
            return ParameterIR(name="", location=self._loc, defid=None)
        name = sexpr[1]
        if isinstance(name, list):
            name = _sym_val(name[0]) if name else ""
        else:
            name = str(name).strip('"')
        _, opts = _plist(sexpr[2:])
        loc = self._loc_from_opts(opts)
        defid = _parse_defid(opts.get(":defid"))
        param_type = self._deserialize_type(opts.get(":ty"))
        return ParameterIR(name=name, location=loc, param_type=param_type, defid=defid)

    def _deserialize_type(self, sexpr: Any) -> Any:
        if sexpr is None:
            return None
        from ..shared.types import PrimitiveType, RectangularType, UNKNOWN
        if isinstance(sexpr, list):
            tag = _sym_val(sexpr[0]) if sexpr else ""
            if tag == "type" and len(sexpr) >= 2:
                kind = _sym_val(sexpr[1])
                if kind == "unknown":
                    return UNKNOWN
                return PrimitiveType(kind)
            if tag == "rectangular-type" and len(sexpr) >= 2:
                element = self._deserialize_type(sexpr[1])
                if element is None:
                    return None
                shape = None
                is_dynamic_rank = False
                if len(sexpr) >= 3:
                    third = sexpr[2]
                    if isinstance(third, list):
                        dims = []
                        for d in third:
                            if d is None or (sexpdata and isinstance(d, sexpdata.Symbol) and _sym_val(d) == "?"):
                                dims.append(None)
                            elif isinstance(d, int):
                                dims.append(d)
                            else:
                                try:
                                    dims.append(int(d))
                                except (TypeError, ValueError):
                                    dims.append(None)
                        shape = tuple(dims)
                    else:
                        if sexpdata and isinstance(third, sexpdata.Symbol):
                            is_dynamic_rank = _sym_val(third) == "true"
                        elif third is True:
                            is_dynamic_rank = True
                if len(sexpr) >= 4:
                    v = sexpr[3]
                    if sexpdata and isinstance(v, sexpdata.Symbol):
                        is_dynamic_rank = _sym_val(v) == "true"
                    elif v is True:
                        is_dynamic_rank = True
                return RectangularType(element_type=element, shape=shape, is_dynamic_rank=is_dynamic_rank)
            if tag == "function-type" and len(sexpr) >= 3:
                from ..shared.types import FunctionType
                param_sexprs = sexpr[1] if isinstance(sexpr[1], list) else []
                param_types = tuple(self._deserialize_type(p) for p in param_sexprs)
                return_type = self._deserialize_type(sexpr[2])
                return FunctionType(param_types=param_types, return_type=return_type)
            if tag == "tuple-type" and len(sexpr) >= 2:
                from ..shared.types import TupleType
                elem_sexprs = sexpr[1] if isinstance(sexpr[1], list) else []
                element_types = [self._deserialize_type(e) for e in elem_sexprs]
                return TupleType(element_types=element_types)
        return None


def deserialize_ir(sexpr_str: str) -> IRNode:
    """
    Deserialize S-expression string to IR node.
    """
    if not sexpdata:
        raise ImportError("sexpdata is required for IR deserialization")
    parsed = sexpdata.loads(sexpr_str)
    deserializer = IRDeserializer()
    return deserializer.deserialize(parsed)


def load_ir(filepath: str) -> IRNode:
    """
    Load and deserialize IR from file.
    """
    with open(filepath, 'r') as f:
        return deserialize_ir(f.read())


def save_ir(node: IRNode, filepath: str, include_location: bool = False, include_type_info: bool = False) -> None:
    """
    Serialize IR node and save to file.
    
    Args:
        node: IR node to serialize
        filepath: Path to output file
        include_location: Include source location metadata
        include_type_info: Include type information
    """
    sexpr_str = serialize_ir(node, include_location=include_location, include_type_info=include_type_info)
    with open(filepath, 'w') as f:
        f.write(sexpr_str)


def load_ir(filepath: str) -> IRNode:
    """
    Load and deserialize IR from file.
    
    TODO: Implement full deserialization for deserialization.
    For now, this is a stub to maintain API compatibility.
    """
    raise NotImplementedError("IR deserialization not yet implemented")


def dump_ir(node: IRNode, include_location: bool = True, include_type_info: bool = True) -> str:
    """
    Dump IR node as formatted S-expression for debugging.
    
    Args:
        node: IR node to dump
        include_location: Include source location metadata
        include_type_info: Include type information
    
    Returns:
        Formatted S-expression string
    """
    return serialize_ir(node, include_location=include_location, include_type_info=include_type_info)
