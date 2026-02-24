"""NumPy backend core: execute, env scope stack only (no global table)."""

from typing import Dict, Any, Optional, List, Union

from ..backends.base import Backend
from ..ir.nodes import (
    ProgramIR, ExpressionIR, FunctionDefIR, ConstantDefIR,
    LiteralIR, FunctionCallIR, IRVisitor,
)
from ..shared.defid import DefId, Resolver, FIXED_BUILTIN_ORDER, _BUILTIN_CRATE
from ..runtime.environment import ExecutionEnvironment, FunctionValue
from .numpy_helpers import (
    _reject_non_lowered,
    builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
    builtin_shape, builtin_sum, builtin_max, builtin_min,
)


def _register_fixed_builtins(env: ExecutionEnvironment) -> None:
    fns = (
        builtin_assert, builtin_print, builtin_len, builtin_typeof, builtin_array_append,
        builtin_shape, builtin_sum, builtin_max, builtin_min,
    )
    for i, fn in enumerate(fns):
        if i < len(FIXED_BUILTIN_ORDER):
            env.set_value(DefId(krate=_BUILTIN_CRATE, index=i), fn)


def _get_execution_result():
    from ..runtime.runtime import ExecutionResult
    return ExecutionResult


class CoreExecutionMixin:
    """Execute, env scope stack only. No def_table, no builtin_defids; all in env."""

    def __init__(self):
        self.env: ExecutionEnvironment = ExecutionEnvironment()
        _register_fixed_builtins(self.env)
        self.resolver: Optional[Resolver] = None

    def execute(
        self,
        program: ProgramIR,
        inputs: Optional[Dict[str, Any]] = None,
        resolver: Optional[Resolver] = None,
        tcx: Optional[Any] = None,
        *,
        input_by_defid: Optional[Dict[DefId, Any]] = None,
        main_defid: Optional[DefId] = None,
    ) -> Any:
        from ..shared.defid import DefType
        self.resolver = resolver
        self._tcx = tcx
        self.env = ExecutionEnvironment()
        _register_fixed_builtins(self.env)
        for func in program.functions:
            if getattr(func, "defid", None):
                self.env.set_value(func.defid, func, name=getattr(func, 'name', None))
        for mod in getattr(program, "modules", None) or []:
            for func in self._collect_module_functions(mod):
                if getattr(func, "defid", None):
                    self.env.set_value(func.defid, func, name=getattr(func, 'name', None))
        if tcx:
            function_ir_map = getattr(tcx, "function_ir_map", None)
            if function_ir_map:
                for func in function_ir_map.values():
                    if isinstance(func, FunctionDefIR) and getattr(func, "defid", None):
                        self.env.set_value(func.defid, func, name=getattr(func, 'name', None))
        if resolver:
            for defid, (def_type, definition) in resolver._def_registry.items():
                if def_type == DefType.BUILTIN:
                    key = DefId(krate=defid.krate, index=defid.index)
                    self.env.set_value(key, definition)
        if input_by_defid:
            for defid, value in input_by_defid.items():
                self.env.set_value(defid, value)
        try:
            if main_defid:
                main_func = self.env.get_value(main_defid)
                if main_func is not None:
                    result_value = self._call_function(main_func, [])
                    return _get_execution_result()(value=result_value)
            outputs = {}
            if program.statements:
                with self.env.scope():
                    for stmt in program.statements:
                        if stmt is None:
                            raise ValueError("IR statement is None")
                        result_value = stmt.accept(self)
                        binding = getattr(stmt, "_binding", None)
                        variable_defid = getattr(binding, "defid", None) if binding else None
                        if variable_defid is not None:
                            var_name = getattr(binding, "name", None) or getattr(stmt, "name", None)
                            self.env.set_value(variable_defid, result_value, name=var_name)
                            outputs[variable_defid] = result_value
                    for defid, value in self.env.get_current_scope().items():
                        if defid not in outputs:
                            outputs[defid] = value
            return _get_execution_result()(outputs=outputs)
        except Exception as e:
            from ..shared.errors import EinlangSourceError
            if isinstance(e, EinlangSourceError):
                return _get_execution_result()(error=e)
            return _get_execution_result()(error=RuntimeError(str(e)))

    def execute_expression(self, expr: ExpressionIR, env: Dict[DefId, Any]) -> Any:
        with self.env.scope():
            for defid, value in env.items():
                self.env.set_value(defid, value)
            return expr.accept(self)

    def _collect_module_functions(self, mod: Any) -> List[FunctionDefIR]:
        from ..ir.nodes import ModuleIR
        if not isinstance(mod, ModuleIR):
            return []
        result = list(getattr(mod, "functions", None) or [])
        for sub in getattr(mod, "submodules", None) or []:
            result.extend(self._collect_module_functions(sub))
        return result

    def _call_function(self, func_def: Union[FunctionDefIR, Any], args: List[Any]) -> Any:
        expected = len(func_def.parameters)
        actual = len(args)
        if actual != expected:
            raise RuntimeError(f"Function (name log: {getattr(func_def, 'name', '<lambda>')}) expects {expected} argument(s), got {actual}")
        with self.env.scope():
            for param, arg_value in zip(func_def.parameters, args):
                if param.defid is None:
                    raise RuntimeError(f"Parameter has no defid; cannot bind. Name (log): {getattr(param, 'name', '?')}")
                self.env.set_value(param.defid, arg_value, name=getattr(param, 'name', None))
            return func_def.body.accept(self)

    def codegen(self, program: ProgramIR) -> str:
        return "# NumPy code generation not yet implemented"

    def visit_program(self, node: ProgramIR) -> Any:
        results = []
        for stmt in node.statements:
            results.append(stmt.accept(self))
        return results[-1] if results else None

    def visit_module(self, node: Any) -> Any:
        raise NotImplementedError("Module execution not yet implemented")

    def visit_function_def(self, node: FunctionDefIR) -> Any:
        if node.defid:
            self.env.set_value(node.defid, node, name=node.name)
        return None

    def visit_constant_def(self, node: ConstantDefIR) -> Any:
        value = node.value.accept(self)
        if node.defid:
            self.env.set_value(node.defid, value, name=node.name)
        return value

    def visit_literal_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_identifier_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_wildcard_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_tuple_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_array_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_rest_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_guard_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_or_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_constructor_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_binding_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
    def visit_range_pattern(self, node: Any) -> Any:
        raise NotImplementedError("Patterns are matched, not executed")
