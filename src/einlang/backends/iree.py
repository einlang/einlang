"""IREE-backed CPU execution with NumPy fallback."""

from __future__ import annotations

import hashlib
import importlib
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..ir.nodes import (
    ArrayLiteralIR,
    BinaryOpIR,
    BindingIR,
    BlockExpressionIR,
    CastExpressionIR,
    IdentifierIR,
    LiteralIR,
    UnaryOpIR,
    is_function_binding,
)
from ..shared.types import BinaryOp, PrimitiveType, RectangularType, UnaryOp
from .numpy import NumPyBackend


class _IREEUnsupported(RuntimeError):
    """Raised when a function cannot be lowered to the supported IREE subset."""


class _IREEUnavailable(RuntimeError):
    """Raised when the optional IREE dependency is not installed."""


@dataclass(frozen=True)
class _ValueSpec:
    shape: Tuple[int, ...]
    numpy_dtype_name: str
    mlir_type: str
    is_scalar: bool

    @property
    def signature_key(self) -> Tuple[Tuple[int, ...], str]:
        return (self.shape, self.numpy_dtype_name)


@dataclass(frozen=True)
class _CompiledFunction:
    entry_name: str
    arg_specs: Tuple[_ValueSpec, ...]
    result_spec: _ValueSpec
    returns_python_scalar: bool
    callable_obj: Any
    mlir_module: str


class IREEBackend(NumPyBackend):
    """
    Hybrid CPU backend.

    Pure, supported tensor/scalar functions are compiled through IREE. Everything
    else runs through the existing NumPy interpreter so language coverage stays
    intact while the IREE surface grows.
    """

    _FALLBACK_SENTINEL = object()

    def __init__(self):
        super().__init__()
        self._iree_module_cache: Dict[Any, Any] = {}
        self._iree_import_cache: Optional[Tuple[Any, Any]] = None

    def _call_function(self, func_def: Any, args: List[Any]) -> Any:
        if not isinstance(func_def, BindingIR) or not is_function_binding(func_def) or func_def.defid is None:
            return super()._call_function(func_def, args)
        if not self._is_structurally_eligible(func_def):
            return super()._call_function(func_def, args)

        try:
            arg_specs, prepared_args = self._prepare_argument_specs(func_def, args)
        except _IREEUnsupported:
            return super()._call_function(func_def, args)

        cache_key = (func_def.defid, tuple(spec.signature_key for spec in arg_specs))
        cached = self._iree_module_cache.get(cache_key)
        if cached is self._FALLBACK_SENTINEL:
            return super()._call_function(func_def, args)

        if cached is None:
            preview_result = super()._call_function(func_def, args)
            try:
                cached = self._compile_function(func_def, arg_specs, preview_result)
            except _IREEUnsupported:
                self._iree_module_cache[cache_key] = self._FALLBACK_SENTINEL
                return preview_result
            self._iree_module_cache[cache_key] = cached

        try:
            return self._invoke_compiled(cached, prepared_args)
        except _IREEUnavailable:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"IREE execution failed for function '{func_def.name}': {exc}"
            ) from exc

    def codegen(self, program: Any) -> str:
        return "# IREE backend lowers supported functions lazily at runtime"

    def _is_structurally_eligible(self, func_def: BindingIR) -> bool:
        local_defids = {
            param.defid for param in func_def.parameters if getattr(param, "defid", None) is not None
        }
        try:
            return self._expr_is_supported(func_def.body, local_defids)
        except _IREEUnsupported:
            return False

    def _expr_is_supported(self, expr: Any, local_defids: set) -> bool:
        if expr is None:
            raise _IREEUnsupported("missing expression")
        if isinstance(expr, LiteralIR):
            self._infer_literal_spec(expr)
            return True
        if isinstance(expr, IdentifierIR):
            if expr.defid in local_defids:
                return True
            value = self.env.get_value(expr.defid) if expr.defid is not None else None
            self._value_to_spec(value)
            return True
        if isinstance(expr, BinaryOpIR):
            if expr.operator not in {
                BinaryOp.ADD,
                BinaryOp.SUB,
                BinaryOp.MUL,
                BinaryOp.DIV,
            }:
                raise _IREEUnsupported(f"unsupported binary operator: {expr.operator}")
            return self._expr_is_supported(expr.left, local_defids) and self._expr_is_supported(expr.right, local_defids)
        if isinstance(expr, UnaryOpIR):
            if expr.operator not in {UnaryOp.NEG, UnaryOp.POS}:
                raise _IREEUnsupported(f"unsupported unary operator: {expr.operator}")
            return self._expr_is_supported(expr.operand, local_defids)
        if isinstance(expr, CastExpressionIR):
            self._declared_numpy_dtype(expr.target_type)
            return self._expr_is_supported(expr.expr, local_defids)
        if isinstance(expr, ArrayLiteralIR):
            self._constant_value_from_expr(expr)
            return True
        if isinstance(expr, BlockExpressionIR):
            scoped = set(local_defids)
            for stmt in expr.statements or ():
                if not isinstance(stmt, BindingIR) or is_function_binding(stmt) or stmt.defid is None:
                    raise _IREEUnsupported("unsupported block statement")
                self._expr_is_supported(stmt.expr, scoped)
                scoped.add(stmt.defid)
            if expr.final_expr is None:
                raise _IREEUnsupported("block expression missing final value")
            return self._expr_is_supported(expr.final_expr, scoped)
        raise _IREEUnsupported(f"unsupported expression node: {type(expr).__name__}")

    def _compile_function(
        self,
        func_def: BindingIR,
        arg_specs: Sequence[_ValueSpec],
        preview_result: Any,
    ) -> _CompiledFunction:
        result_spec, returns_python_scalar = self._infer_result_spec(
            preview_result,
            declared_type=getattr(func_def, "return_type", None) or getattr(func_def.body, "type_info", None),
        )
        entry_name = self._entry_name(func_def, arg_specs, result_spec)
        mlir_module = self._emit_mlir_module(func_def, entry_name, arg_specs, result_spec)
        compile_tools, runtime_mod = self._import_iree_modules()
        try:
            try:
                vmfb = compile_tools.compile_str(
                    mlir_module,
                    target_backends=["llvm-cpu"],
                    input_type="auto",
                )
            except TypeError:
                vmfb = compile_tools.compile_str(
                    mlir_module,
                    target_backends=["llvm-cpu"],
                )
        except Exception as exc:
            raise RuntimeError(
                f"IREE compilation failed for function '{func_def.name}': {exc}"
            ) from exc
        callable_obj = self._load_entry_callable(runtime_mod, vmfb, entry_name)
        return _CompiledFunction(
            entry_name=entry_name,
            arg_specs=tuple(arg_specs),
            result_spec=result_spec,
            returns_python_scalar=returns_python_scalar,
            callable_obj=callable_obj,
            mlir_module=mlir_module,
        )

    def _prepare_argument_specs(
        self,
        func_def: BindingIR,
        args: Sequence[Any],
    ) -> Tuple[Tuple[_ValueSpec, ...], Tuple[np.ndarray, ...]]:
        params = tuple(func_def.parameters or ())
        if len(args) != len(params):
            raise _IREEUnsupported("arity mismatch while preparing IREE call")
        specs: List[_ValueSpec] = []
        prepared: List[np.ndarray] = []
        for param, arg in zip(params, args):
            array_value = self._prepare_runtime_value(arg, getattr(param, "param_type", None))
            specs.append(self._value_to_spec(array_value))
            prepared.append(array_value)
        return tuple(specs), tuple(prepared)

    def _prepare_runtime_value(self, value: Any, declared_type: Optional[Any]) -> np.ndarray:
        declared_dtype = self._declared_numpy_dtype(declared_type)
        try:
            array_value = np.asarray(value, dtype=declared_dtype) if declared_dtype is not None else np.asarray(value)
        except Exception as exc:
            raise _IREEUnsupported(f"unsupported runtime value: {exc}") from exc
        if array_value.dtype.kind in {"O", "U", "S"}:
            raise _IREEUnsupported(f"unsupported dtype for IREE lowering: {array_value.dtype}")
        return array_value

    def _declared_numpy_dtype(self, type_obj: Optional[Any]) -> Optional[np.dtype]:
        if type_obj is None:
            return None
        converter = getattr(self, "_type_info_to_numpy_dtype", None)
        if callable(converter):
            try:
                dtype = converter(type_obj)
            except Exception:
                dtype = None
            if dtype is not None:
                return np.dtype(dtype)
        if isinstance(type_obj, PrimitiveType):
            return self._primitive_numpy_dtype(type_obj.name)
        if isinstance(type_obj, RectangularType):
            return self._declared_numpy_dtype(type_obj.element_type)
        return None

    @staticmethod
    def _primitive_numpy_dtype(name: str) -> Optional[np.dtype]:
        mapping = {
            "bool": np.dtype(np.bool_),
            "i8": np.dtype(np.int8),
            "i32": np.dtype(np.int32),
            "i64": np.dtype(np.int64),
            "f16": np.dtype(np.float16),
            "bf16": np.dtype(np.float16),
            "f32": np.dtype(np.float32),
            "f64": np.dtype(np.float64),
        }
        return mapping.get(name)

    def _value_to_spec(self, value: Any) -> _ValueSpec:
        array_value = value if isinstance(value, np.ndarray) else np.asarray(value)
        if array_value.dtype.kind in {"O", "U", "S"}:
            raise _IREEUnsupported(f"unsupported dtype for IREE lowering: {array_value.dtype}")
        dtype = np.dtype(array_value.dtype)
        mlir_elem_type = self._mlir_element_type(dtype)
        shape = tuple(int(dim) for dim in array_value.shape)
        if shape:
            dims = "x".join(str(dim) for dim in shape)
            mlir_type = f"tensor<{dims}x{mlir_elem_type}>"
        else:
            mlir_type = f"tensor<{mlir_elem_type}>"
        return _ValueSpec(
            shape=shape,
            numpy_dtype_name=dtype.name,
            mlir_type=mlir_type,
            is_scalar=(array_value.ndim == 0),
        )

    @staticmethod
    def _mlir_element_type(dtype: np.dtype) -> str:
        dtype = np.dtype(dtype)
        mapping = {
            "bool": "i1",
            "int8": "i8",
            "int16": "i16",
            "int32": "i32",
            "int64": "i64",
            "float16": "f16",
            "float32": "f32",
            "float64": "f64",
        }
        try:
            return mapping[dtype.name]
        except KeyError as exc:
            raise _IREEUnsupported(f"unsupported dtype for IREE lowering: {dtype}") from exc

    def _infer_result_spec(
        self,
        preview_result: Any,
        declared_type: Optional[Any] = None,
    ) -> Tuple[_ValueSpec, bool]:
        declared_dtype = self._declared_numpy_dtype(declared_type)
        if isinstance(preview_result, np.ndarray):
            array_value = (
                np.asarray(preview_result, dtype=declared_dtype)
                if declared_dtype is not None
                else np.asarray(preview_result)
            )
            return self._value_to_spec(array_value), False
        if np.isscalar(preview_result) or isinstance(preview_result, np.generic):
            array_value = (
                np.asarray(preview_result, dtype=declared_dtype)
                if declared_dtype is not None
                else np.asarray(preview_result)
            )
            return self._value_to_spec(array_value), True
        raise _IREEUnsupported(f"unsupported IREE return value: {type(preview_result).__name__}")

    def _emit_mlir_module(
        self,
        func_def: BindingIR,
        entry_name: str,
        arg_specs: Sequence[_ValueSpec],
        result_spec: _ValueSpec,
    ) -> str:
        params = []
        env: Dict[Any, Tuple[str, _ValueSpec]] = {}
        for index, (param, spec) in enumerate(zip(func_def.parameters, arg_specs)):
            ssa_name = f"%arg{index}"
            params.append(f"{ssa_name}: {spec.mlir_type}")
            if param.defid is not None:
                env[param.defid] = (ssa_name, spec)
        builder = _MLIRBuilder()
        result_ref, lowered_spec = self._lower_expr(func_def.body, env, builder)
        if lowered_spec != result_spec:
            raise _IREEUnsupported("lowered result signature does not match inferred runtime result")
        lines = [
            "module {",
            f"  func.func @{entry_name}({', '.join(params)}) -> {result_spec.mlir_type} {{",
        ]
        lines.extend(builder.lines)
        lines.append(f"    return {result_ref} : {result_spec.mlir_type}")
        lines.append("  }")
        lines.append("}")
        return "\n".join(lines)

    def _lower_expr(
        self,
        expr: Any,
        env: Mapping[Any, Tuple[str, _ValueSpec]],
        builder: "_MLIRBuilder",
    ) -> Tuple[str, _ValueSpec]:
        if isinstance(expr, LiteralIR):
            return self._emit_constant(expr.value, self._infer_literal_spec(expr), builder)
        if isinstance(expr, IdentifierIR):
            if expr.defid in env:
                return env[expr.defid]
            if expr.defid is None:
                raise _IREEUnsupported("identifier without defid")
            value = self.env.get_value(expr.defid)
            return self._emit_constant(value, self._value_to_spec(value), builder)
        if isinstance(expr, BinaryOpIR):
            lhs_ref, lhs_spec = self._lower_expr(expr.left, env, builder)
            rhs_ref, rhs_spec = self._lower_expr(expr.right, env, builder)
            if lhs_spec != rhs_spec:
                raise _IREEUnsupported("binary op requires matching shapes and dtypes")
            op_name = {
                BinaryOp.ADD: "add",
                BinaryOp.SUB: "subtract",
                BinaryOp.MUL: "multiply",
                BinaryOp.DIV: "divide",
            }.get(expr.operator)
            if op_name is None:
                raise _IREEUnsupported(f"unsupported binary operator: {expr.operator}")
            result_ref = builder.new_value()
            builder.emit(
                f"{result_ref} = stablehlo.{op_name} {lhs_ref}, {rhs_ref} : {lhs_spec.mlir_type}"
            )
            return result_ref, lhs_spec
        if isinstance(expr, UnaryOpIR):
            operand_ref, operand_spec = self._lower_expr(expr.operand, env, builder)
            if expr.operator == UnaryOp.POS:
                return operand_ref, operand_spec
            if expr.operator != UnaryOp.NEG:
                raise _IREEUnsupported(f"unsupported unary operator: {expr.operator}")
            result_ref = builder.new_value()
            builder.emit(f"{result_ref} = stablehlo.negate {operand_ref} : {operand_spec.mlir_type}")
            return result_ref, operand_spec
        if isinstance(expr, CastExpressionIR):
            input_ref, input_spec = self._lower_expr(expr.expr, env, builder)
            cast_dtype = self._declared_numpy_dtype(expr.target_type)
            if cast_dtype is None:
                raise _IREEUnsupported("unsupported cast target")
            target_spec = self._value_spec_with_dtype(input_spec, cast_dtype)
            result_ref = builder.new_value()
            builder.emit(
                f"{result_ref} = stablehlo.convert {input_ref} : ({input_spec.mlir_type}) -> {target_spec.mlir_type}"
            )
            return result_ref, target_spec
        if isinstance(expr, ArrayLiteralIR):
            constant_value = self._constant_value_from_expr(expr)
            return self._emit_constant(constant_value, self._value_to_spec(constant_value), builder)
        if isinstance(expr, BlockExpressionIR):
            scoped_env = dict(env)
            for stmt in expr.statements or ():
                if not isinstance(stmt, BindingIR) or stmt.defid is None or is_function_binding(stmt):
                    raise _IREEUnsupported("unsupported block statement")
                value_ref, value_spec = self._lower_expr(stmt.expr, scoped_env, builder)
                scoped_env[stmt.defid] = (value_ref, value_spec)
            if expr.final_expr is None:
                raise _IREEUnsupported("block expression missing final value")
            return self._lower_expr(expr.final_expr, scoped_env, builder)
        raise _IREEUnsupported(f"unsupported expression node: {type(expr).__name__}")

    def _infer_literal_spec(self, expr: LiteralIR) -> _ValueSpec:
        declared_dtype = self._declared_numpy_dtype(expr.type_info)
        if declared_dtype is not None:
            return self._value_to_spec(np.asarray(expr.value, dtype=declared_dtype))
        value = expr.value
        if isinstance(value, bool):
            return self._value_to_spec(np.asarray(value, dtype=np.bool_))
        if isinstance(value, int):
            return self._value_to_spec(np.asarray(value, dtype=np.int32))
        if isinstance(value, float):
            return self._value_to_spec(np.asarray(value, dtype=np.float32))
        raise _IREEUnsupported(f"unsupported literal value: {value!r}")

    def _constant_value_from_expr(self, expr: Any) -> np.ndarray:
        if isinstance(expr, LiteralIR):
            spec = self._infer_literal_spec(expr)
            return np.asarray(expr.value, dtype=np.dtype(spec.numpy_dtype_name))
        if isinstance(expr, ArrayLiteralIR):
            elements = [self._constant_value_from_expr(element) for element in expr.elements]
            if not elements:
                return np.asarray([], dtype=np.int32)
            dtype = np.result_type(*[element.dtype for element in elements])
            normalized = [np.asarray(element, dtype=dtype) for element in elements]
            return np.asarray(normalized, dtype=dtype)
        raise _IREEUnsupported("array literals may only contain literal constants in IREE mode")

    def _emit_constant(
        self,
        value: Any,
        spec: _ValueSpec,
        builder: "_MLIRBuilder",
    ) -> Tuple[str, _ValueSpec]:
        constant_ref = builder.new_value()
        constant_value = np.asarray(value, dtype=np.dtype(spec.numpy_dtype_name))
        builder.emit(
            f"{constant_ref} = stablehlo.constant dense<{self._format_dense_value(constant_value)}> : {spec.mlir_type}"
        )
        return constant_ref, spec

    def _value_spec_with_dtype(self, original: _ValueSpec, dtype: np.dtype) -> _ValueSpec:
        dtype = np.dtype(dtype)
        elem_type = self._mlir_element_type(dtype)
        if original.shape:
            dims = "x".join(str(dim) for dim in original.shape)
            mlir_type = f"tensor<{dims}x{elem_type}>"
        else:
            mlir_type = f"tensor<{elem_type}>"
        return _ValueSpec(
            shape=original.shape,
            numpy_dtype_name=dtype.name,
            mlir_type=mlir_type,
            is_scalar=original.is_scalar,
        )

    def _invoke_compiled(self, compiled: _CompiledFunction, args: Sequence[np.ndarray]) -> Any:
        result = compiled.callable_obj(*args)
        host_value = self._materialize_runtime_value(result)
        if compiled.returns_python_scalar:
            array_value = np.asarray(host_value, dtype=np.dtype(compiled.result_spec.numpy_dtype_name))
            return array_value.item()
        return np.asarray(host_value, dtype=np.dtype(compiled.result_spec.numpy_dtype_name))

    def _materialize_runtime_value(self, value: Any) -> Any:
        current = value
        if isinstance(current, tuple) and len(current) == 1:
            current = current[0]
        if hasattr(current, "to_host") and callable(current.to_host):
            current = current.to_host()
        elif hasattr(current, "numpy") and callable(current.numpy):
            current = current.numpy()
        return current

    def _import_iree_modules(self) -> Tuple[Any, Any]:
        if self._iree_import_cache is not None:
            return self._iree_import_cache
        try:
            compile_tools = importlib.import_module("iree.compiler.tools")
            runtime_mod = importlib.import_module("iree.runtime")
        except ImportError as exc:
            raise _IREEUnavailable(
                "IREE backend requires optional dependencies "
                "`iree-base-compiler` and `iree-base-runtime`."
            ) from exc
        self._iree_import_cache = (compile_tools, runtime_mod)
        return self._iree_import_cache

    def _load_entry_callable(self, runtime_mod: Any, vmfb: Any, entry_name: str) -> Any:
        loader = getattr(runtime_mod, "load_vm_flatbuffer", None)
        if callable(loader):
            try:
                loaded = loader(vmfb, driver="local-task")
            except TypeError:
                loaded = loader(vmfb)
            entry = self._lookup_entry_callable(loaded, entry_name)
            if entry is not None:
                return entry

        config_cls = getattr(runtime_mod, "Config", None)
        context_cls = getattr(runtime_mod, "SystemContext", None)
        vm_module_cls = getattr(runtime_mod, "VmModule", None)
        if config_cls is None or context_cls is None or vm_module_cls is None:
            raise RuntimeError("Unable to find a supported IREE Python runtime entrypoint")

        config = config_cls("local-task")
        from_flatbuffer = getattr(vm_module_cls, "from_flatbuffer", None)
        if from_flatbuffer is None:
            raise RuntimeError("IREE VmModule.from_flatbuffer is unavailable")
        vm_module = from_flatbuffer(config.vm_instance, vmfb)
        context = context_cls(config=config)
        add_vm_module = getattr(context, "add_vm_module", None)
        if callable(add_vm_module):
            add_vm_module(vm_module)

        entry = self._lookup_entry_callable(context, entry_name)
        if entry is None:
            raise RuntimeError(f"IREE entrypoint '{entry_name}' was not found in the loaded module")
        return entry

    def _lookup_entry_callable(self, container: Any, entry_name: str) -> Optional[Any]:
        if container is None:
            return None
        try:
            candidate = container[entry_name]
        except Exception:
            candidate = None
        if callable(candidate):
            return candidate
        direct = getattr(container, entry_name, None)
        if callable(direct):
            return direct
        for attr_name in ("module", "main", "modules"):
            nested = getattr(container, attr_name, None)
            if nested is None or nested is container:
                continue
            nested_hit = self._lookup_entry_callable(nested, entry_name)
            if nested_hit is not None:
                return nested_hit
        if isinstance(container, dict):
            nested = container.get(entry_name)
            if callable(nested):
                return nested
        return None

    def _entry_name(
        self,
        func_def: BindingIR,
        arg_specs: Sequence[_ValueSpec],
        result_spec: _ValueSpec,
    ) -> str:
        base_name = re.sub(r"[^0-9A-Za-z_]+", "_", func_def.name or "fn").strip("_") or "fn"
        signature = repr(
            (
                [(spec.shape, spec.numpy_dtype_name) for spec in arg_specs],
                (result_spec.shape, result_spec.numpy_dtype_name),
            )
        ).encode("utf-8")
        digest = hashlib.sha1(signature).hexdigest()[:12]
        return f"ein_{base_name}_{digest}"

    def _format_dense_value(self, value: np.ndarray) -> str:
        if value.ndim == 0:
            return self._format_scalar(value.item())
        return "[" + ", ".join(self._format_dense_value(np.asarray(item)) for item in value) + "]"

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if isinstance(value, (np.bool_, bool)):
            return "true" if bool(value) else "false"
        if isinstance(value, (np.integer, int)):
            return str(int(value))
        if isinstance(value, (np.floating, float)):
            numeric = float(value)
            if math.isnan(numeric) or math.isinf(numeric):
                raise _IREEUnsupported("NaN/Inf constants are not supported in the current IREE backend")
            text = repr(numeric)
            if "." not in text and "e" not in text.lower():
                text += ".0"
            return text
        raise _IREEUnsupported(f"unsupported scalar literal for dense constant: {value!r}")


class _MLIRBuilder:
    """Small helper for deterministic SSA emission."""

    def __init__(self):
        self.lines: List[str] = []
        self._counter = 0

    def new_value(self) -> str:
        name = f"%{self._counter}"
        self._counter += 1
        return name

    def emit(self, line: str) -> None:
        self.lines.append(f"    {line}")
