"""
Minimal repro tests for specialized-function failures.

Failure patterns:
- "Function not found (DefId: ...)" when call is rewritten to specialized DefId not in def_table.
- "Function not found (DefId: 0:2393)" when calling std::ml::topk: resolver gives use-site DefId but
  we only register the stdlib definition DefId (Rust: use definition DefId at call sites).
- "Non-lowered IR at runtime: ArrayComprehensionIR" when generic path still executed.

Minimal stdlib DefId repro: test_minimal_stdlib_topk_defid_repro
  - Single call std::ml::topk(arr_2d, k, axis); fails at runtime with DefId not in def_table.
Real repro with user impl: test_minimal_user_module_topk_defid_repro
  - User module (source_overlay) defines topk; main uses 'use my_module::topk' and calls topk(...).
  - Fails with Function not found (DefId: 0:0); definition lowered as 0:1 (same use-site vs definition bug).
ArrayComprehensionIR fail pattern as user impl: test_minimal_user_module_array_comprehension_repro
  - User module with sort_descending + topk_else_branch (3-way-if then sort_descending(unsorted));
    same pattern as std::ml::topk path that hits ArrayComprehensionIR. With fixes, passes (regression).

Smallest ArrayComprehensionIR repro: test_specialized_topk_three_way_unsorted_repro
  - Generic with comprehension whose body is a 3-way nested if, then call to another generic.
  - 2-way if in comprehension (test_specialized_topk_mid_branch_only_repro) passes.

Minimal failure repro (minimal_specialized_call_not_rewritten_repro):
  - Generic callee with one comprehension. Generic caller: let with comprehension (if-body), then call callee.
  - In mini-program type pass the let's type is UNKNOWN so the call is not rewritten to specialized defid;
    at runtime we execute generic callee -> ArrayComprehensionIR.
"""

import pytest
import numpy as np
from tests.test_utils import compile_and_execute


def _assert_vector_output(result, output_name: str, expected):
    assert result.success, f"Execution failed: {result.errors}"
    actual = np.asarray(result.outputs[output_name], dtype=np.float64)
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float64), rtol=1e-6)


def _assert_pair_outputs(result, value_name: str, index_name: str, expected_values):
    assert result.success, f"Execution failed: {result.errors}"
    vals = np.asarray(result.outputs[value_name], dtype=np.float64)
    idxs = np.asarray(result.outputs[index_name], dtype=np.int32)
    assert vals.shape == tuple(np.asarray(expected_values).shape)
    assert idxs.shape == vals.shape
    np.testing.assert_allclose(vals, np.asarray(expected_values, dtype=np.float64), rtol=1e-6)


def test_user_module_defid(compiler, runtime):
    """
    Real repro of DefId mismatch with user impl only (no stdlib topk): topk defined in
    user module (source_overlay), main does 'use my_module::topk' and calls topk(...).
    Call site gets use-site DefId 0:0, definition is lowered as 0:1 -> Function not found
    (DefId: 0:0). Same bug as std::ml::topk; fix: use definition DefId at call sites.
    """
    main_source = """
use my_module::topk;
let arr = [[1.0, 5.0, 3.0, 2.0, 4.0]];
let (vals, idxs) = topk(arr, 2, 1);
"""
    my_module_source = """
fn argmax_1d(row) {
    let n = len(row);
    [i | i in 0..n, len([1 | j in 0..n, row[i] >= row[j]]) == n][0]
}
pub fn topk(X, k, axis) {
    let row = [X[0, i] | i in 0..len(X[0])];
    let n = len(row);
    let idx0 = argmax_1d(row);
    let idx1 = [i | i in 0..n, i != idx0, len([1 | j in 0..n, j != idx0, row[i] >= row[j]]) == n - 1][0];
    ([[row[idx0], row[idx1]]], [[idx0, idx1]])
}
"""
    source_overlay = {("my_module",): my_module_source}
    result = compile_and_execute(
        main_source,
        compiler,
        runtime,
        source_file="main.ein",
        source_overlay=source_overlay,
    )
    _assert_pair_outputs(result, "vals", "idxs", [[5.0, 4.0]])


def test_user_module_einstein(compiler, runtime):
    """
    Use user module that mirrors std::ml::topk: Einstein let with call to helper that has
    array comprehension (let values_work[i,j] = row_values(X,i,N,k)[j]; row_values has
    let row = [X[row_idx,c] | c in 0..n_cols]). Same structure as std::ml::topk; if calls
    inside Einstein RHS are not rewritten we get ArrayComprehensionIR at runtime.
    Main: use my_module::topk_2d; call topk_2d(arr, 2, 1).
    """
    main_source = """
use my_module::topk_2d;
let arr = [[1.0, 5.0, 3.0, 2.0, 4.0]];
let (vals, idxs) = topk_2d(arr, 2, 1);
"""
    my_module_source = """
fn row_values(arr_2d, row_idx, n_cols, k) {
    let row = [arr_2d[row_idx, c] | c in 0..n_cols];
    if k <= 0 { [] } else {
        let n = len(row);
        let idx0 = [i | i in 0..n, len([1 | j in 0..n, row[i] >= row[j]]) == n][0];
        let idx1 = [i | i in 0..n, i != idx0, len([1 | j in 0..n, j != idx0, row[i] >= row[j]]) == n - 1][0];
        [row[idx0], row[idx1]]
    }
}
fn row_indices(arr_2d, row_idx, n_cols, k) {
    let row = [arr_2d[row_idx, c] | c in 0..n_cols];
    if k <= 0 { [] } else {
        let n = len(row);
        let idx0 = [i | i in 0..n, len([1 | j in 0..n, row[i] >= row[j]]) == n][0];
        let idx1 = [i | i in 0..n, i != idx0, len([1 | j in 0..n, j != idx0, row[i] >= row[j]]) == n - 1][0];
        [idx0, idx1]
    }
}
pub fn topk_2d(X, k, axis) {
    let M = len(X);
    let N = len(X[0]);
    let values_work[i in 0..M, j in 0..k] = row_values(X, i, N, k)[j];
    let indices_work[i in 0..M, j in 0..k] = row_indices(X, i, N, k)[j];
    (values_work, indices_work)
}
"""
    source_overlay = {("my_module",): my_module_source}
    result = compile_and_execute(
        main_source,
        compiler,
        runtime,
        source_file="main.ein",
        source_overlay=source_overlay,
    )
    _assert_pair_outputs(result, "vals", "idxs", [[5.0, 4.0]])


def test_user_module_array_comp(compiler, runtime):
    """
    Extract ML selection fail pattern (ArrayComprehensionIR at runtime) as user impl.
    Same pattern as std::ml::topk: generic with comprehension + inner call to another generic
    (3-way-if comprehension then sort_descending(unsorted)). When inner call is not rewritten
    (arg type UNKNOWN in mini-program type pass) the generic path runs -> ArrayComprehensionIR.
    User module has sort_descending and topk_else_branch; main calls topk_else_branch(arr, 2).
    With type_inference UNKNOWN fallback and module_path lowering this passes (regression test).
    """
    main_source = """
use my_module::topk_else_branch;
let arr = [3.0, 1.0, 4.0, 1.0, 5.0];
let top2 = topk_else_branch(arr, 2);
"""
    my_module_source = """
fn sort_descending(arr) {
    if len(arr) <= 1 { arr } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) { pivot }
            else { if i < len(sorted_larger) + 1 + len(equal) { equal[i - len(sorted_larger) - 1] }
                   else { sorted_smaller[i - len(sorted_larger) - 1 - len(equal)] } }
        };
        result
    }
}
pub fn topk_else_branch(arr, k) {
    if k == 0 { [] }
    else {
        if len(arr) <= k { sort_descending(arr) }
        else {
            let pivot = arr[len(arr) / 2];
            let larger = [arr[i] | i in 0..len(arr), arr[i] > pivot];
            let equal = [arr[i] | i in 0..len(arr), arr[i] == pivot];
            let smaller = [arr[i] | i in 0..len(arr), arr[i] < pivot];
            let larger_count = len(larger);
            let equal_count = len(equal);
            let needed_smaller = k - larger_count - equal_count;
            let smaller_top = topk_else_branch(smaller, needed_smaller);
            let unsorted = [
                if i < larger_count { larger[i] }
                else { if i < larger_count + equal_count { equal[i - larger_count] } else { smaller_top[i - larger_count - equal_count] } }
                | i in 0..k
            ];
            sort_descending(unsorted)
        }
    }
}
"""
    source_overlay = {("my_module",): my_module_source}
    result = compile_and_execute(
        main_source,
        compiler,
        runtime,
        source_file="main.ein",
        source_overlay=source_overlay,
    )
    _assert_vector_output(result, "top2", [5.0, 4.0])


def test_stdlib_defid(compiler, runtime):
    """
    Minimal repro for 'Function not found (DefId: 0:2393)' when calling std::ml::topk.
    Call site gets a resolver DefId (e.g. use-site binding) but backend def_table only has
    the stdlib definition DefId. Fix: at lowering use definition DefId for codegen (Rust pattern).
    """
    source = """
let arr = [[1.0, 5.0, 3.0, 2.0, 4.0]];
let (vals, idxs) = std::ml::topk(arr, 2, 1);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_pair_outputs(result, "vals", "idxs", [[5.0, 4.0]])


@pytest.mark.parametrize(
    ("source", "output_name", "expected"),
    [
        pytest.param(
            """
            fn copy_arr(x) { [x[i] | i in 0..len(x)] }
            let a = [1.0, 2.0, 3.0];
            let b = copy_arr(a);
            """,
            "b",
            [1.0, 2.0, 3.0],
            id="specialized-comprehension-minimal",
        ),
        pytest.param(
            """
            fn sort_descending(arr) {
                if len(arr) <= 1 { arr } else {
                    let pivot = arr[0];
                    let rest = [arr[i] | i in 1..len(arr)];
                    let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
                    let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
                    let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
                    let sorted_larger = sort_descending(larger);
                    let sorted_smaller = sort_descending(smaller);
                    let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
                    let result[i in 0..total_len] = if i < len(sorted_larger) {
                        sorted_larger[i]
                    } else {
                        if i == len(sorted_larger) { pivot }
                        else { if i < len(sorted_larger) + 1 + len(equal) { equal[i - len(sorted_larger) - 1] }
                               else { sorted_smaller[i - len(sorted_larger) - 1 - len(equal)] } }
                    };
                    result
                }
            }
            fn topk_else_branch(arr, k) {
                if k == 0 { [] }
                else {
                    if len(arr) <= k { sort_descending(arr) }
                    else {
                        let pivot = arr[len(arr) / 2];
                        let larger = [arr[i] | i in 0..len(arr), arr[i] > pivot];
                        let equal = [arr[i] | i in 0..len(arr), arr[i] == pivot];
                        let smaller = [arr[i] | i in 0..len(arr), arr[i] < pivot];
                        let larger_count = len(larger);
                        let equal_count = len(equal);
                        let needed_smaller = k - larger_count - equal_count;
                        let smaller_top = topk_else_branch(smaller, needed_smaller);
                        let unsorted = [
                            if i < larger_count { larger[i] }
                            else { if i < larger_count + equal_count { equal[i - larger_count] } else { smaller_top[i - larger_count - equal_count] } }
                            | i in 0..k
                        ];
                        sort_descending(unsorted)
                    }
                }
            }
            let arr = [3.0, 1.0, 4.0, 1.0, 5.0];
            let top2 = topk_else_branch(arr, 2);
            """,
            "top2",
            [5.0, 4.0],
            id="specialized-call-not-rewritten",
        ),
        pytest.param(
            """
            fn double_elems(x) {
                let result[i in 0..len(x)] = x[i] * 2;
                result
            }
            let a = [1.0, 2.0, 3.0];
            let b = double_elems(a);
            """,
            "b",
            [2.0, 4.0, 6.0],
            id="specialized-einstein-decl",
        ),
        pytest.param(
            """
            fn inner(x) { [x[i] | i in 0..len(x)] }
            fn outer(x) { inner(x) }
            let a = [1.0, 2.0, 3.0];
            let b = outer(a);
            """,
            "b",
            [1.0, 2.0, 3.0],
            id="specialized-inner-call",
        ),
        pytest.param(
            """
            fn maybe_copy(x) {
                if len(x) <= 1 {
                    x
                } else {
                    [x[i] | i in 0..len(x)]
                }
            }
            let a = [1.0, 2.0, 3.0];
            let b = maybe_copy(a);
            """,
            "b",
            [1.0, 2.0, 3.0],
            id="specialized-conditional-comprehension",
        ),
        pytest.param(
            """
            fn concat_two(a, b) {
                let n = len(a) + len(b);
                [if i < len(a) { a[i] } else { b[i - len(a)] } | i in 0..n]
            }
            let x = [1.0, 2.0];
            let y = [3.0];
            let z = concat_two(x, y);
            """,
            "z",
            [1.0, 2.0, 3.0],
            id="specialized-comprehension-with-if-body",
        ),
        pytest.param(
            """
            fn concat_two(a, b) {
                let n = len(a) + len(b);
                [if i < len(a) { a[i] } else { b[i - len(a)] } | i in 0..n]
            }
            fn id_arr(x) { [x[i] | i in 0..len(x)] }
            let x = [1.0, 2.0];
            let y = [3.0];
            let z = concat_two(x, y);
            let w = id_arr(z);
            """,
            "w",
            [1.0, 2.0, 3.0],
            id="specialized-build-then-call-generic",
        ),
        pytest.param(
            """
            fn concat_two(a, b) {
                let n = len(a) + len(b);
                [if i < len(a) { a[i] } else { b[i - len(a)] } | i in 0..n]
            }
            fn double_elems(x) {
                let result[i in 0..len(x)] = x[i] * 2;
                result
            }
            let x = [1.0, 2.0];
            let y = [3.0];
            let z = concat_two(x, y);
            let w = double_elems(z);
            """,
            "w",
            [2.0, 4.0, 6.0],
            id="specialized-build-then-einstein-callee",
        ),
    ],
)
def test_vec(compiler, runtime, source, output_name, expected):
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, output_name, expected)


def test_comp_then_call_same_fn(compiler, runtime):
    """Repro: one generic that builds array (comprehension with if) and passes to another generic in same fn."""
    source = """
fn double_elems(x) {
    let result[i in 0..len(x)] = x[i] * 2;
    result
}
fn concat_then_double(a, b) {
    let n = len(a) + len(b);
    let z = [if i < len(a) { a[i] } else { b[i - len(a)] } | i in 0..n];
    double_elems(z)
}
let x = [1.0, 2.0];
let y = [3.0];
let w = concat_then_double(x, y);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "w", [2.0, 4.0, 6.0])


def test_topk_mid_branch(compiler, runtime):
    """Repro: topk path that builds unsorted (comprehension with 2-way if) then sort_descending(unsorted)."""
    source = """
fn sort_descending(arr) {
    if len(arr) <= 1 {
        arr
    } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) {
                pivot
            } else {
                if i < len(sorted_larger) + 1 + len(equal) {
                    equal[i - len(sorted_larger) - 1]
                } else {
                    sorted_smaller[i - len(sorted_larger) - 1 - len(equal)]
                }
            }
        };
        result
    }
}
fn topk_mid_only(arr, k) {
    let pivot = arr[len(arr) / 2];
    let larger = [arr[i] | i in 0..len(arr), arr[i] > pivot];
    let equal = [arr[i] | i in 0..len(arr), arr[i] == pivot];
    let larger_count = len(larger);
    let needed_equal = k - larger_count;
    let equal_subset = [equal[i] | i in 0..needed_equal];
    let unsorted = [if i < larger_count { larger[i] } else { equal_subset[i - larger_count] } | i in 0..k];
    sort_descending(unsorted)
}
let arr = [5.0, 1.0, 4.0, 2.0, 3.0];
let top2 = topk_mid_only(arr, 2);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "top2", [5.0, 4.0])


def test_topk_recursive_sort(compiler, runtime):
    """Repro: generic that recurses and also calls sort_descending (like full topk_extract)."""
    source = """
fn sort_descending(arr) {
    if len(arr) <= 1 {
        arr
    } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) {
                pivot
            } else {
                if i < len(sorted_larger) + 1 + len(equal) {
                    equal[i - len(sorted_larger) - 1]
                } else {
                    sorted_smaller[i - len(sorted_larger) - 1 - len(equal)]
                }
            }
        };
        result
    }
}
fn topk_simple(arr, k) {
    if k == 0 {
        []
    } else {
        if len(arr) <= k {
            sort_descending(arr)
        } else {
            let pivot = arr[0];
            let larger = [arr[i] | i in 1..len(arr), arr[i] > pivot];
            if k <= len(larger) {
                topk_simple(larger, k)
            } else {
                sort_descending(arr)
            }
        }
    }
}
let arr = [3.0, 1.0, 4.0, 1.0, 5.0];
let top2 = topk_simple(arr, 2);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "top2", [5.0, 4.0])


def test_topk_three_way_unsorted(compiler, runtime):
    """Repro: topk branch with 3-way if in comprehension then sort_descending (full topk else branch)."""
    source = """
fn sort_descending(arr) {
    if len(arr) <= 1 {
        arr
    } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) {
                pivot
            } else {
                if i < len(sorted_larger) + 1 + len(equal) {
                    equal[i - len(sorted_larger) - 1]
                } else {
                    sorted_smaller[i - len(sorted_larger) - 1 - len(equal)]
                }
            }
        };
        result
    }
}
fn topk_else_branch(arr, k) {
    if k == 0 {
        []
    } else {
        if len(arr) <= k {
            sort_descending(arr)
        } else {
            let pivot = arr[len(arr) / 2];
            let larger = [arr[i] | i in 0..len(arr), arr[i] > pivot];
            let equal = [arr[i] | i in 0..len(arr), arr[i] == pivot];
            let smaller = [arr[i] | i in 0..len(arr), arr[i] < pivot];
            let larger_count = len(larger);
            let equal_count = len(equal);
            let needed_smaller = k - larger_count - equal_count;
            let smaller_top = topk_else_branch(smaller, needed_smaller);
            let unsorted = [
                if i < larger_count {
                    larger[i]
                } else {
                    if i < larger_count + equal_count {
                        equal[i - larger_count]
                    } else {
                        smaller_top[i - larger_count - equal_count]
                    }
                }
                | i in 0..k
            ];
            sort_descending(unsorted)
        }
    }
}
let arr = [3.0, 1.0, 4.0, 1.0, 5.0];
let top2 = topk_else_branch(arr, 2);
"""
    result = compile_and_execute(source, compiler, runtime)
    assert result.success, f"Execution failed: {result.errors}"
    out = np.asarray(result.outputs["top2"])
    assert len(out) == 2
    assert out[0] == 5.0 and out[1] == 4.0


def test_sort_desc_only(compiler, runtime):
    """Repro: only sort_descending (recursive + comprehensions + Einstein decl), no topk_extract."""
    source = """
fn sort_descending(arr) {
    if len(arr) <= 1 {
        arr
    } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) {
                pivot
            } else {
                if i < len(sorted_larger) + 1 + len(equal) {
                    equal[i - len(sorted_larger) - 1]
                } else {
                    sorted_smaller[i - len(sorted_larger) - 1 - len(equal)]
                }
            }
        };
        result
    }
}
let arr = [3.0, 1.0, 4.0];
let top = sort_descending(arr);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "top", [4.0, 3.0, 1.0])


def test_topk_calls_sort(compiler, runtime):
    """Minimal repro: topk_extract that only calls sort_descending (k>=len path)."""
    source = """
fn sort_descending(arr) {
    if len(arr) <= 1 {
        arr
    } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) {
                pivot
            } else {
                if i < len(sorted_larger) + 1 + len(equal) {
                    equal[i - len(sorted_larger) - 1]
                } else {
                    sorted_smaller[i - len(sorted_larger) - 1 - len(equal)]
                }
            }
        };
        result
    }
}
fn topk_extract(arr, k) {
    if len(arr) <= k {
        sort_descending(arr)
    } else {
        []
    }
}
let arr = [3.0, 1.0, 4.0];
let top = topk_extract(arr, 10);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "top", [4.0, 3.0, 1.0])


def test_recursive_comp(compiler, runtime):
    """Repro: generic fn with comprehension that calls itself (like sort_descending)."""
    source = """
fn rev(x) {
    if len(x) <= 1 {
        x
    } else {
        let rest = [x[i] | i in 1..len(x)];
        let reversed_rest = rev(rest);
        let n = len(reversed_rest) + 1;
        let result[i in 0..n] = if i == n - 1 { x[0] } else { reversed_rest[i] };
        result
    }
}
let a = [1.0, 2.0, 3.0];
let b = rev(a);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "b", [3.0, 2.0, 1.0])


def test_topk_extract(compiler, runtime):
    """Minimal repro: std::array::topk_extract (specialized body calls sort_descending)."""
    source = """
use std::array;
let arr = [3.0, 1.0, 4.0, 1.0, 5.0];
let top2 = std::array::topk_extract(arr, 2);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "top2", [5.0, 4.0])


def test_topk_user_impl(compiler, runtime):
    """Same as topk_extract repro but with topk_extract implemented in user code (no stdlib)."""
    source = """
fn sort_descending(arr) {
    if len(arr) <= 1 {
        arr
    } else {
        let pivot = arr[0];
        let rest = [arr[i] | i in 1..len(arr)];
        let larger = [rest[i] | i in 0..len(rest), rest[i] > pivot];
        let equal = [rest[i] | i in 0..len(rest), rest[i] == pivot];
        let smaller = [rest[i] | i in 0..len(rest), rest[i] < pivot];
        let sorted_larger = sort_descending(larger);
        let sorted_smaller = sort_descending(smaller);
        let total_len = len(sorted_larger) + 1 + len(equal) + len(sorted_smaller);
        let result[i in 0..total_len] = if i < len(sorted_larger) {
            sorted_larger[i]
        } else {
            if i == len(sorted_larger) {
                pivot
            } else {
                if i < len(sorted_larger) + 1 + len(equal) {
                    equal[i - len(sorted_larger) - 1]
                } else {
                    sorted_smaller[i - len(sorted_larger) - 1 - len(equal)]
                }
            }
        };
        result
    }
}
fn topk_extract(arr, k) {
    if k == 0 {
        []
    } else {
        if len(arr) <= k {
            sort_descending(arr)
        } else {
            let pivot = arr[len(arr) / 2];
            let larger = [arr[i] | i in 0..len(arr), arr[i] > pivot];
            let equal = [arr[i] | i in 0..len(arr), arr[i] == pivot];
            let smaller = [arr[i] | i in 0..len(arr), arr[i] < pivot];
            let larger_count = len(larger);
            let equal_count = len(equal);
            if k <= larger_count {
                topk_extract(larger, k)
            } else {
                if k <= larger_count + equal_count {
                    let needed_equal = k - larger_count;
                    let equal_subset = [equal[i] | i in 0..needed_equal];
                    let unsorted = [if i < larger_count { larger[i] } else { equal_subset[i - larger_count] } | i in 0..k];
                    sort_descending(unsorted)
                } else {
                    let needed_smaller = k - larger_count - equal_count;
                    let smaller_top = topk_extract(smaller, needed_smaller);
                    let unsorted = [
                        if i < larger_count {
                            larger[i]
                        } else {
                            if i < larger_count + equal_count {
                                equal[i - larger_count]
                            } else {
                                smaller_top[i - larger_count - equal_count]
                            }
                        }
                        | i in 0..k
                    ];
                    sort_descending(unsorted)
                }
            }
        }
    }
}
let arr = [3.0, 1.0, 4.0, 1.0, 5.0];
let top2 = topk_extract(arr, 2);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_vector_output(result, "top2", [5.0, 4.0])


def test_ml_topk(compiler, runtime):
    """Minimal repro: std::ml::topk (specialized DefId must be in def_table)."""
    source = """
use std::ml;
let x = [[1.0, 5.0, 3.0], [4.0, 1.0, 6.0]];
let (vals, idxs) = std::ml::topk(x, 2, 1);
"""
    result = compile_and_execute(source, compiler, runtime)
    _assert_pair_outputs(result, "vals", "idxs", [[5.0, 3.0], [6.0, 4.0]])
