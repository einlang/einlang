# Built-ins vs C-callable: learning from Julia and MATLAB

Design note: what should be a **built-in** (language primitive) vs what can be provided by **C (or native) libraries** and how to expose them. Informed by Julia and MATLAB.

---

## 1. Julia

### Built-in vs stdlib in Julia

Julia’s layering is:

| Layer | What it is | Examples | How you use it |
|-------|------------|----------|----------------|
| **Core** | Implemented in **C**; the minimal runtime. Types and operations the compiler and runtime need to bootstrap. “Implementation detail” users rarely touch; `Base` re-exports much of it. | `Core.Int`, `Core.Intrinsics`, `Core.Builtins` (e.g. `memoryrefget`), fundamental types and intrinsics | Usually via `Base` (e.g. `Int`, `println`). Direct `Core.*` for low-level/unstable APIs. |
| **Base** | The **standard library** users think of as “built-in”: written mostly in Julia, built on Core. Broad, stable, documented. Preloaded in every session. | I/O (`print`, `read`), collections (`length`, `getindex`), math (`sin`, `exp`), types (`String`, `Array`), etc. | No `using` needed; names are in the default namespace. |
| **Stdlib packages** | **Standard library packages**: shipped with Julia, but separate namespaces. Load with `using`. Functionality has been moved here from Base over time. | `LinearAlgebra`, `Random`, `Dates`, `SparseArrays`, `Distributed`, etc. | `using LinearAlgebra` then `eigen(A)`, etc. |
| **C libraries** | External shared libraries. | libc, libm, custom `.so`/`.dylib` | `@ccall library.function_name(args...)::returntype`. No glue; JIT emits a direct native call. |

So in Julia terminology: **“built-in”** usually means “in Core or exposed by Base without a `using`”; **“stdlib”** means Base (the default library) plus the bundled packages you load with `using`. Core is the only part that is literally C-builtins + intrinsics; Base and stdlib packages are Julia code (and Base wraps many C calls).

### Three layers (implementation view)

| Layer | What it is | Examples | How you use it |
|-------|------------|----------|----------------|
| **Intrinsics** | LLVM/compiler primitives; operate on unboxed data, statically typed. | `Core.Intrinsics.*`, LLVM intrinsics via `llvmcall` | Compiler lowers to IR; not first-class user callables. |
| **Built-ins** | First-class Julia functions implemented in C inside the runtime (`builtins.c`). | Many exposed in `Base`: I/O, type utilities, core data operations | Call by name like any other function; compiler/runtime know the symbol. |
| **C libraries** | External shared libraries. | libc, libm, libglib, custom `.so`/`.dylib` | `@ccall library.function_name(args...)::returntype` (or `ccall(...)`). No glue code; JIT generates a direct native call. |

### Design choices

- **No boilerplate**: `@ccall` from the REPL; no codegen or wrapper step.
- **Zero overhead**: ccall compiles to the same machine call as from C.
- **Type mapping**: Arguments are converted via `cconvert` / `unsafe_convert`; you must match C types exactly (no headers; you declare types in Julia).
- **Standard library as wrapper**: Many “built-in-like” APIs (e.g. `getenv`) are **Julia functions** that wrap `@ccall` and turn C error conventions into Julia exceptions.
- **Math**: Julia often implements math (e.g. trig) in Julia for reproducibility and range reduction; LLVM intrinsics/libm are used where appropriate but are not the only option.

**Takeaway for Einlang:**  
- Keep a **small, fixed set of true built-ins** (language primitives with DefId in a builtin crate).  
- Everything else can be **stdlib or user code** that eventually calls C/native via a single, explicit mechanism (e.g. `@ccall`-style or a dedicated “call C” construct).  
- Don’t overload “built-in” with “anything we ship in stdlib”; built-ins are the minimal set the compiler and runtime treat specially.

---

## 2. MATLAB

### Three layers

| Layer | What it is | Examples | How you use it |
|-------|------------|----------|----------------|
| **Built-in functions** | Pre-compiled C (and similar) written and optimized by MathWorks; no user-visible source. | `svd`, `eig`, `fft`, `qr`, `lu`, many core ops | Call by name; look like M-file functions but are not. |
| **M-files** | User or MathWorks MATLAB source (`.m`). | Most of the “library” and user code | Normal scripts/functions; call by name. |
| **MEX files** | C/C++/Fortran compiled to a binary callable from MATLAB (gateway `mexFunction`). | User-written native code, or wrappers to C libs | Build with `mex`; then call the MEX function by name like a built-in. |

### Is exp (and sin, tanh, …) built-in or stdlib?

- **Julia:** **Stdlib (Base).** `exp`, `sin`, `tanh`, etc. live in **Base** and are implemented in **Julia** (e.g. `base/special/exp.jl`). They are not in Core (C builtins). So Julia treats math functions as part of the standard library, not as language primitives. Users still call `exp(x)` with no `using` because Base is preloaded.
- **MATLAB:** **Built-in.** `exp`, `sin`, and similar math functions are **built-in**: they are part of the MATLAB executable (implemented in C/compiled code by MathWorks), not M-files. You can call `builtin("exp", x)` to invoke the built-in even when `exp` is overloaded. So MATLAB treats exp as a built-in.

So the two languages differ: Julia puts exp in the **stdlib** (Base, implemented in Julia); MATLAB puts exp in the **built-in** set (part of the executable). Einlang’s choice to have exp in **stdlib** (`std::math::exp` → `python::numpy::exp`) is aligned with Julia’s approach (stdlib, not core primitive).

### Design choices

- **Built-ins dominate performance**: Core math/linear algebra are built-ins; converting user code to MEX only helps if that user code dominates, not if it’s mostly calling built-ins.
- **MEX for extension**: When you need C/C++/Fortran (speed, existing libs), you write a MEX file; from the user’s perspective it’s “a function,” same as built-in or M-file.
- **Single call surface**: Built-in, M-file, and MEX are all invoked the same way (by name); the language doesn’t force you to think about which layer you’re calling.

**Takeaway for Einlang:**  
- **Built-in** = small set of primitives the compiler/runtime know by DefId (like `assert`, `print`, `len`, `shape`, `sum`, `max`, `min`, etc.).  
- **Stdlib** = Einlang (or host) code that can call out to C/NumPy/etc.; to the user it’s “a function,” but it’s not a language built-in.  
- Optionally, a **single “native call” mechanism** (e.g. “call this C/lib symbol with this signature”) keeps the boundary clear: everything beyond built-ins can be implemented via that mechanism or via stdlib that uses it.

---

## 3. Einlang alignment

### Current state

- **Builtins (DefId in BUILTIN_CRATE):** Fixed set in `FIXED_BUILTIN_ORDER`: `assert`, `print`, `len`, `typeof`, `array_append`, `shape`, `sum`, `max`, `min`. The NumPy backend registers all of these; builtin calls are dispatched by DefId.
- **exp, tanh (and similar math):** **Stdlib only.** They are not in `FIXED_BUILTIN_ORDER`. Use `use std::math::exp::exp;` and `use std::math::hyperbolic::{tanh};` (or re-exports from `std::math`). Implementations live in `stdlib/math/exp.ein`, `stdlib/math/hyperbolic.ein`, etc., and call `python::numpy::exp`, `python::numpy::tanh`, etc.

### Recommended split (Julia + MATLAB style)

| Category | Meaning | Examples | Implemented as |
|----------|--------|----------|----------------|
| **Language built-ins** | Primitives the compiler and runtime always recognize (DefId in builtin crate). Small set. | `assert`, `print`, `len`, `typeof`, `shape`, `array_append`, `sum`, `max`, `min` | Backend registers a callable per DefId; no user override. |
| **Stdlib** | Normal modules/functions; may call Python/NumPy/C under the hood. | `std::math::exp`, `std::math::tanh`, `std::math::ln`, `python::numpy::*` | Einlang code + FFI or host interop; not DefId-builtins. |
| **C / native (future)** | Explicit “call C/lib” mechanism. | Custom libs, libm, BLAS, etc. | Single construct (e.g. `@ccall`-like or `extern "C"` block) with type mapping; no ad-hoc builtins for every C function. |

### Principles

1. **Built-in = minimal and fixed**  
   Only what the language and passes (e.g. autodiff, shape) need to treat as primitives. Everything else is stdlib or user code.

2. **One C-call path**  
   Don’t add a new built-in for every C function. Add one mechanism (e.g. `@ccall lib.symbol(args...)::rettype`) and use it from stdlib or user code.

3. **Stdlib can wrap C**  
   Like Julia’s `getenv` or MATLAB’s M-file wrappers: user-facing API is Einlang; implementation can be “call C/NumPy” under the hood.

4. **exp, tanh, and similar math**  
   Stdlib only. Use `std::math::exp`, `std::math::hyperbolic::tanh`, etc. No builtin slot; keeps the builtin set small and aligns with Julia (Base/stdlib, not Core).

### What about exp?

**exp (and tanh) are stdlib-only.** They are not in `FIXED_BUILTIN_ORDER`. Call them via `use std::math::exp::exp;` and `use std::math::hyperbolic::{tanh};` (or re-exports from `std::math`). The stdlib implementations (`stdlib/math/exp.ein`, `stdlib/math/hyperbolic.ein`) call `python::numpy::exp` and `python::numpy::tanh`. Execution and autodiff (diff rules or differentiating through the function body) use the stdlib path. Bare `exp(x)` with no `use` is undefined and will report “Undefined function 'exp'” (or similar); users must import from stdlib.

---

## 4. Summary

- **Julia:** Intrinsics (compiler) vs built-ins (runtime C) vs `@ccall` (external C); stdlib wraps C and converts errors.  
- **MATLAB:** Built-ins (opaque C) vs M-files vs MEX (your C); same call surface for all.  
- **Einlang:** Keep a small, fixed set of true built-ins; treat “can be implemented by C/NumPy” as stdlib or a single C-call mechanism, not as an ever-growing builtin list.

This keeps the language core small and makes “built-in vs from C” a clear design boundary rather than an implementation detail.
