import ast
import json
import re
import sys
import types

import numpy as np

from einlang.runtime.runtime import EinlangRuntime
from tests.test_utils import compile_and_execute


def _numpy_dtype_from_mlir_tensor(mlir_type: str):
    inner = mlir_type[len("tensor<"):-1]
    elem = inner.split("x")[-1]
    mapping = {
        "i1": np.bool_,
        "i8": np.int8,
        "i16": np.int16,
        "i32": np.int32,
        "i64": np.int64,
        "f16": np.float16,
        "f32": np.float32,
        "f64": np.float64,
    }
    return mapping[elem]


def _parse_dense_literal(raw: str):
    text = raw.strip()
    if text == "true":
        return True
    if text == "false":
        return False
    try:
        return ast.literal_eval(text.replace("true", "True").replace("false", "False"))
    except Exception:
        if "." in text or "e" in text.lower():
            return float(text)
        return int(text)


def _evaluate_fake_mlir(mlir_module: str, *args):
    env = {f"%arg{i}": np.asarray(arg) for i, arg in enumerate(args)}
    lines = [line.strip() for line in mlir_module.splitlines() if line.strip()]
    for line in lines:
        if line.startswith("%") and "stablehlo.constant" in line:
            match = re.match(r"(%\d+) = stablehlo\.constant dense<(.*)> : (tensor<.*>)", line)
            assert match is not None, line
            ssa_name, dense_literal, mlir_type = match.groups()
            env[ssa_name] = np.asarray(_parse_dense_literal(dense_literal), dtype=_numpy_dtype_from_mlir_tensor(mlir_type))
            continue
        if line.startswith("%") and "stablehlo.convert" in line:
            match = re.match(r"(%\d+) = stablehlo\.convert (%[\w\d]+) : \((tensor<.*>)\) -> (tensor<.*>)", line)
            assert match is not None, line
            ssa_name, operand_name, _old_type, new_type = match.groups()
            env[ssa_name] = np.asarray(env[operand_name], dtype=_numpy_dtype_from_mlir_tensor(new_type))
            continue
        if line.startswith("%") and "stablehlo.negate" in line:
            match = re.match(r"(%\d+) = stablehlo\.negate (%[\w\d]+) : (tensor<.*>)", line)
            assert match is not None, line
            ssa_name, operand_name, _mlir_type = match.groups()
            env[ssa_name] = -np.asarray(env[operand_name])
            continue
        if line.startswith("%") and "stablehlo." in line:
            match = re.match(r"(%\d+) = stablehlo\.(add|subtract|multiply|divide) (%[\w\d]+), (%[\w\d]+) : (tensor<.*>)", line)
            assert match is not None, line
            ssa_name, op_name, lhs_name, rhs_name, _mlir_type = match.groups()
            lhs = np.asarray(env[lhs_name])
            rhs = np.asarray(env[rhs_name])
            if op_name == "add":
                env[ssa_name] = lhs + rhs
            elif op_name == "subtract":
                env[ssa_name] = lhs - rhs
            elif op_name == "multiply":
                env[ssa_name] = lhs * rhs
            else:
                env[ssa_name] = lhs / rhs
            continue
        if line.startswith("return "):
            match = re.match(r"return (%[\w\d]+) : (tensor<.*>)", line)
            assert match is not None, line
            return env[match.group(1)]
    raise AssertionError("No return found in fake MLIR module")


def _install_fake_iree(monkeypatch):
    state = {"compiled_modules": [], "loaded_entries": []}

    iree_pkg = types.ModuleType("iree")
    compiler_pkg = types.ModuleType("iree.compiler")
    tools_mod = types.ModuleType("iree.compiler.tools")
    runtime_mod = types.ModuleType("iree.runtime")

    def compile_str(mlir_module, target_backends=None, input_type=None):
        match = re.search(r"func.func @([^(]+)\(", mlir_module)
        assert match is not None, mlir_module
        entry_name = match.group(1)
        state["compiled_modules"].append(mlir_module)
        payload = {"entry_name": entry_name, "mlir_module": mlir_module}
        return json.dumps(payload).encode("utf-8")

    class FakeModule:
        def __init__(self, entry_name, mlir_module):
            self._entry_name = entry_name
            self._mlir_module = mlir_module

        def __getitem__(self, key):
            if key != self._entry_name:
                raise KeyError(key)

            def _invoke(*args):
                return _evaluate_fake_mlir(self._mlir_module, *args)

            return _invoke

    def load_vm_flatbuffer(vmfb, driver="local-task"):
        payload = json.loads(vmfb.decode("utf-8"))
        state["loaded_entries"].append((payload["entry_name"], driver))
        return FakeModule(payload["entry_name"], payload["mlir_module"])

    tools_mod.compile_str = compile_str
    runtime_mod.load_vm_flatbuffer = load_vm_flatbuffer
    iree_pkg.compiler = compiler_pkg
    iree_pkg.runtime = runtime_mod
    compiler_pkg.tools = tools_mod

    monkeypatch.setitem(sys.modules, "iree", iree_pkg)
    monkeypatch.setitem(sys.modules, "iree.compiler", compiler_pkg)
    monkeypatch.setitem(sys.modules, "iree.compiler.tools", tools_mod)
    monkeypatch.setitem(sys.modules, "iree.runtime", runtime_mod)
    return state


def test_runtime_selects_iree_backend():
    runtime = EinlangRuntime(backend="iree")
    assert runtime.backend.__class__.__name__ == "IREEBackend"


def test_iree_backend_compiles_scalar_function_with_mocked_runtime(compiler, monkeypatch):
    state = _install_fake_iree(monkeypatch)
    runtime = EinlangRuntime(backend="iree")
    source = """
fn add_ints(a: i32, b: i32) -> i32 { a + b }
let result = add_ints(20, 22);
"""
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    assert result.outputs["result"] == 42
    assert len(state["compiled_modules"]) == 1
    assert "stablehlo.add" in state["compiled_modules"][0]


def test_iree_backend_compiles_tensor_function_with_mocked_runtime(compiler, monkeypatch):
    state = _install_fake_iree(monkeypatch)
    runtime = EinlangRuntime(backend="iree")
    source = """
fn add_vec(a: [i32; 3], b: [i32; 3]) -> [i32; 3] { a + b }
let result = add_vec([1, 2, 3], [4, 5, 6]);
"""
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    np.testing.assert_array_equal(result.outputs["result"], np.array([5, 7, 9], dtype=np.int32))
    assert len(state["compiled_modules"]) == 1
    assert "tensor<3xi32>" in state["compiled_modules"][0]


def test_iree_backend_falls_back_for_unsupported_function_body(compiler, monkeypatch):
    state = _install_fake_iree(monkeypatch)
    runtime = EinlangRuntime(backend="iree")
    source = """
fn count_elems(x: [i32; 3]) -> i32 { len(x) }
let result = count_elems([1, 2, 3]);
"""
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    assert result.outputs["result"] == 3
    assert state["compiled_modules"] == []


def test_iree_backend_reports_missing_optional_dependency(compiler, monkeypatch):
    import einlang.backends.iree as iree_backend_module

    real_import_module = iree_backend_module.importlib.import_module

    def _missing_iree(name, package=None):
        if name.startswith("iree"):
            raise ImportError("missing fake iree")
        return real_import_module(name, package)

    monkeypatch.setattr(iree_backend_module.importlib, "import_module", _missing_iree)

    runtime = EinlangRuntime(backend="iree")
    source = """
fn add_ints(a: i32, b: i32) -> i32 { a + b }
let result = add_ints(1, 2);
"""
    result = compile_and_execute(source, compiler, runtime)
    assert not result.success
    assert result.error is not None
    assert "iree-base-compiler" in result.error
    assert "iree-base-runtime" in result.error
