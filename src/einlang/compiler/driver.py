"""
Compiler Driver

Rust Pattern: rustc_driver::driver
Reference: COMPILER_FLOW_DESIGN.md
"""

from typing import Optional
from pathlib import Path
from ..passes.base import TyCtxt, PassManager, BasePass
from ..shared.source_location import SourceLocation
from ..ir.nodes import ProgramIR
from ..shared.errors import ErrorReporter
from ..frontend.parser import Parser, ParseError
from ..passes.ast_to_ir import ASTToIRLoweringPass
from ..passes.type_inference import TypeInferencePass
from ..passes.range_analysis import RangeAnalysisPass
from ..analysis.module_system import ModuleSystem


def _location_is_meaningful(loc: Optional[SourceLocation]) -> bool:
    if loc is None:
        return False
    if loc.line <= 0:
        return False
    fn = (loc.file or "").strip()
    if not fn or fn in ("<generated>", "<unknown>"):
        return False
    return True


def _same_user_file(loc_file: str, entry_file: str) -> bool:
    if loc_file == entry_file:
        return True
    try:
        a, b = Path(loc_file), Path(entry_file)
        if a.is_absolute() and b.is_absolute():
            try:
                if a.resolve() == b.resolve():
                    return True
            except OSError:
                pass
        return a.name == b.name and bool(a.name)
    except Exception:
        return False


def _first_span_from_program(ir: Optional[ProgramIR], entry_file: str) -> Optional[SourceLocation]:
    if ir is None:
        return None
    entry_match: Optional[SourceLocation] = None
    any_match: Optional[SourceLocation] = None
    for stmt in ir.statements or []:
        loc = getattr(stmt, "location", None)
        if not _location_is_meaningful(loc):
            continue
        assert loc is not None
        if _same_user_file(loc.file, entry_file):
            if entry_match is None:
                entry_match = loc
        elif any_match is None:
            any_match = loc
    return entry_match or any_match


def span_for_uncaught_compile_exception(
    exception: BaseException,
    ir: Optional[ProgramIR],
    entry_file: str,
) -> SourceLocation:
    """Best-effort source span when a pass raises instead of using the error reporter."""
    from ..shared.errors import EinlangError

    if isinstance(exception, EinlangError) and _location_is_meaningful(exception.location):
        assert exception.location is not None
        return exception.location
    span = _first_span_from_program(ir, entry_file)
    if span is not None:
        return span
    return SourceLocation(
        file=entry_file,
        line=1,
        column=1,
        end_line=1,
        end_column=1,
    )


def _report_uncaught_exception_as_diagnostic(
    tcx: TyCtxt,
    exception: BaseException,
    ir: Optional[ProgramIR],
    entry_file: str,
    pass_name: Optional[str] = None,
) -> None:
    if tcx.reporter.has_errors():
        return
    loc = span_for_uncaught_compile_exception(exception, ir, entry_file)
    help_txt: Optional[str] = None
    if isinstance(exception, RecursionError):
        help_txt = (
            "Python recursion limit exceeded inside the compiler (often autodiff or deep IR traversal). "
            "Try simplifying the program; for local debugging you can raise sys.setrecursionlimit."
        )
    note = f"while running compiler pass `{pass_name}`" if pass_name else None
    tcx.reporter.report_error(str(exception), loc, note=note, help=help_txt)


class CompilationResult:
    """Compilation result (Rust: crate output; Python: __file__ = entry source path)."""
    def __init__(
        self,
        ir: Optional[ProgramIR] = None,
        tcx: Optional[TyCtxt] = None,
        success: bool = False,
        entry_source_file: Optional[str] = None,
    ):
        self.ir = ir
        self.tcx = tcx
        self.success = success
        self.entry_source_file = entry_source_file

    def has_errors(self) -> bool:
        """True if compilation reported errors."""
        if self.tcx and self.tcx.reporter:
            return self.tcx.reporter.has_errors()
        return not self.success

    def get_errors(self) -> list:
        """Get compilation errors (legacy API)"""
        if self.tcx and self.tcx.reporter:
            if self.tcx.reporter.has_errors():
                return [self.tcx.reporter.format_all_errors()]
        return []

class CompilerDriver:
    """
    Compiler driver (Rust naming: rustc_driver::driver).
    
    Rust Pattern: rustc_driver::driver
    
    Implementation Alignment: Follows Rust's compiler driver:
    - Orchestrates all compiler phases
    - Manages pass execution
    - Handles errors
    - Returns compilation result
    
    Reference: `rustc_driver::driver` for compiler orchestration
    """
    
    def __init__(self):
        self.pass_manager = PassManager()
        self.parser = Parser()
        self._register_passes()
    
    def _register_passes(self) -> None:
        """
        Register all passes in aligned order.
        
        Pass order:
        1. ModulePass (name resolution)
        2. EinsteinGroupingPass
        3. ConstraintClassifierPass
        4. RestPatternPreprocessingPass
        5. RangeAnalysisPass
        6. UnifiedShapeAnalysisPass
        7. TypeAnalysisPass
        8. PreAutodiffPruningPass
        9. CastValidationPass
        10. PipelineTypeValidationPass
        11. ExhaustivenessPass
        
        Alignment:
        - NameResolutionPass runs on AST before lowering (manual call)
        - ASTToIRLoweringPass converts AST to IR
        - Then follow order on IR
        """
        # 0. AST to IR lowering
        self.pass_manager.register_pass(ASTToIRLoweringPass)
        # Note: Einstein grouping runs on IR via EinsteinDeclarationGroupingPass (registered below)

        # 1. Einstein declaration grouping (analysis)
        from ..passes.einstein_grouping import EinsteinDeclarationGroupingPass
        self.pass_manager.register_pass(EinsteinDeclarationGroupingPass)

        # 2. Constraint classification (analysis)
        from ..passes.constraint_classifier import ConstraintClassifierPass
        self.pass_manager.register_pass(ConstraintClassifierPass)
        
        # 3. Rest pattern preprocessing (expands ..batch to batch.0 early)
        from ..passes.rest_pattern_preprocessing import RestPatternPreprocessingPass
        self.pass_manager.register_pass(RestPatternPreprocessingPass)

        # 3b. Coordinate grounding and selection shorthand expansion
        from ..passes.coordinate_analysis import CoordinateGroundingPass
        self.pass_manager.register_pass(CoordinateGroundingPass)
        
        # 4. Range analysis (infers ranges for loop variables)
        # CRITICAL: Must come before shape analysis (shape needs ranges for offsets)
        self.pass_manager.register_pass(RangeAnalysisPass)
        
        # 5. Shape analysis (uses ranges to compute output dimensions)
        from ..passes.shape_analysis import UnifiedShapeAnalysisPass
        self.pass_manager.register_pass(UnifiedShapeAnalysisPass)
        
        # 6. Type inference (Type analysis runs after shape)
        # This allows type inference to use shape information
        self.pass_manager.register_pass(TypeInferencePass)

        # 7. Canonicalize generic extremum-selection patterns to SelectAtArgmaxIR
        from ..passes.extremum_selection_canonicalization import (
            ExtremumSelectionCanonicalizationPass,
        )
        self.pass_manager.register_pass(ExtremumSelectionCanonicalizationPass)

        # 8. Pre-autodiff pruning (shape/rank branch pruning only)
        from ..passes.pre_autodiff_pruning import PreAutodiffPruningPass
        self.pass_manager.register_pass(PreAutodiffPruningPass)

        # 9. Autodiff (high-level EinsteinIR only; before lowering)
        from ..passes.autodiff import AutodiffPass
        self.pass_manager.register_pass(AutodiffPass)

        # 10. Lower autodiff request nodes to ordinary IR
        from ..passes.autodiff import AutodiffRequestLoweringPass
        self.pass_manager.register_pass(AutodiffRequestLoweringPass)

        # 11. Post-autodiff pruning (shape/rank branch pruning only)
        from ..passes.pre_autodiff_pruning import PostAutodiffPruningPass
        self.pass_manager.register_pass(PostAutodiffPruningPass)

        # 12. Autodiff leak check (fail if @ artifacts survive)
        from ..passes.autodiff_leak_check import AutodiffLeakCheckPass
        self.pass_manager.register_pass(AutodiffLeakCheckPass)

        # 13. Einstein lowering (lower Einstein declarations to loops)
        from ..passes.einstein_lowering import EinsteinLoweringPass
        self.pass_manager.register_pass(EinsteinLoweringPass)

        # 14. Recurrence ordering and lowering
        from ..passes.recurrence_order import RecurrenceOrderPass
        self.pass_manager.register_pass(RecurrenceOrderPass)

        # 15. Compiler-owned lowered execution metadata for backend hot paths
        from ..passes.lowered_execution_facts import LoweredExecutionFactsPass
        self.pass_manager.register_pass(LoweredExecutionFactsPass)

        # 16. Validation passes
        from ..passes.cast_validation import CastValidationPass
        self.pass_manager.register_pass(CastValidationPass)
        
        from ..passes.pipeline_validation import PipelineTypeValidationPass
        self.pass_manager.register_pass(PipelineTypeValidationPass)
        
        from ..passes.exhaustiveness import ExhaustivenessPass
        self.pass_manager.register_pass(ExhaustivenessPass)
        
        # 16. IR validation (validation)
        from ..passes.ir_validation import IRValidationPass
        self.pass_manager.register_pass(IRValidationPass)
        
        # 9. Optimizations (validation)

    def _run_name_resolution_pass(self, ast, tcx: TyCtxt):
        """Run the name resolution pass on AST (its own pass; allocates DefIds, resolves names). Returns AST with DefIds attached."""
        from ..passes.name_resolution import NameResolutionPass
        pass_instance = NameResolutionPass()
        return pass_instance.run(ast, tcx)

    def compile(
        self,
        source: str,
        source_file: str = "main.ein",
        root_path: Optional[Path] = None,
        stdlib_root: Optional[Path] = None,
        stop_after_pass: Optional[str] = None,
        source_overlay: Optional[dict] = None,
    ) -> CompilationResult:
        """
        Compile source code.
        
        Rust Pattern: rustc_driver::driver::compile_input()
        
        Phases:
        1. Parsing (source → AST)
        2. Name Resolution (DefId allocation)
        3. IR Lowering (AST → IR)
        4. Type Inference (on IR)
        5. Const Folding (on IR)
        6. Optimization (on IR)
        7. Codegen (IR → Backend)
        
        Args:
            stop_after_pass: Optional pass name to stop after (e.g., "TypeInferencePass")
                           Useful for IR inspection before optimization passes
        """
        tcx = TyCtxt()
        tcx.source_files[source_file] = source
        # In-memory module sources (avoid I/O on critical path when possible)
        tcx.source_overlay = source_overlay if source_overlay is not None else {}
        ir: Optional[ProgramIR] = None
        try:
            # Phase 1: Parsing (source → AST)
            ast = self.parser.parse(source, source_file)
            
            if root_path is None:
                root_path = Path.cwd()
            module_system = ModuleSystem(root_path, tcx.resolver, stdlib_root=stdlib_root)
            tcx.module_system = module_system
            tcx.discovered_modules = {}
            tcx.source_files[source_file] = source

            # Merge @fn into fn at AST so one DefId per function; no separate DiffRuleDef after this.
            from ..passes.merge_diff_rules import merge_diff_rules_into_functions
            merge_diff_rules_into_functions(ast)

            ast = self._run_name_resolution_pass(ast, tcx)
            if tcx.reporter.has_errors():
                return CompilationResult(success=False, tcx=tcx, entry_source_file=source_file)
            
            # Phase 3-7: Passes (all on IR after lowering)
            from ..passes.ast_to_ir import ASTToIRLoweringPass
            lowering_pass = ASTToIRLoweringPass()
            try:
                ir = lowering_pass.run(ast, tcx)
            except Exception as e:
                _report_uncaught_exception_as_diagnostic(
                    tcx, e, None, source_file, pass_name="ASTToIRLoweringPass"
                )
                return CompilationResult(
                    success=False, ir=None, tcx=tcx, entry_source_file=source_file
                )

            # Check if we should stop after lowering
            if stop_after_pass == 'ASTToIRLoweringPass':
                return CompilationResult(success=True, ir=ir, tcx=tcx, entry_source_file=source_file)

            # Run remaining passes on IR
            # Filter out ASTToIRLoweringPass from pass manager (already run)
            remaining_passes = [
                p for p in self.pass_manager.ordered_passes()
                if p != ASTToIRLoweringPass
            ]
            
            # Run remaining passes using pass manager (handles dependencies automatically)
            # Design Pattern: Use pass manager for dependency resolution (no manual isinstance checks)
            for pass_class in remaining_passes:
                pass_instance = pass_class()
                try:
                    ir = pass_instance.run(ir, tcx)
                except Exception as e:
                    _report_uncaught_exception_as_diagnostic(
                        tcx, e, ir, source_file, pass_name=pass_class.__name__
                    )
                    return CompilationResult(
                        success=False, ir=ir, tcx=tcx, entry_source_file=source_file
                    )

                # Stop after specified pass if requested
                if stop_after_pass and pass_class.__name__ == stop_after_pass:
                    break

            # Check for errors
            if tcx.reporter.has_errors():
                return CompilationResult(
                    success=False, ir=ir, tcx=tcx, entry_source_file=source_file
                )
            
            function_ir_map = getattr(tcx, 'function_ir_map', None) or {}
            func_set = set(ir.functions)
            for f in function_ir_map.values():
                if f is not None and f not in func_set:
                    ir.statements.append(f)
                    ir.bindings.append(f)
                    func_set.add(f)
            
            from ..passes.tree_shaking import tree_shake
            ir = tree_shake(ir)
            return CompilationResult(ir=ir, tcx=tcx, success=True, entry_source_file=source_file)
        
        except ParseError as e:
            from ..shared.source_location import SourceLocation
            if e.location is not None:
                loc = e.location
                span = SourceLocation(
                    file=loc.file,
                    line=loc.line,
                    column=loc.column,
                    end_line=loc.line,
                    end_column=loc.column + 1,
                )
            else:
                span = SourceLocation(
                    file=source_file, line=1, column=1,
                    end_line=1, end_column=1,
                )
            tcx.reporter.report_error(e.message, span)
            return CompilationResult(
                success=False, ir=ir, tcx=tcx, entry_source_file=source_file
            )
        except Exception as e:
            _report_uncaught_exception_as_diagnostic(tcx, e, ir, source_file, pass_name=None)
            return CompilationResult(
                success=False, ir=ir, tcx=tcx, entry_source_file=source_file
            )
