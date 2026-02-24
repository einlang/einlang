"""
Compiler Driver

Rust Pattern: rustc_driver::driver
Reference: COMPILER_FLOW_DESIGN.md
"""

import os
from typing import Optional
from pathlib import Path
from ..passes.base import TyCtxt, PassManager, BasePass
from ..ir.nodes import ProgramIR
from ..shared.errors import ErrorReporter
from ..frontend.parser import Parser, ParseError
from ..passes.ast_to_ir import ASTToIRLoweringPass
from ..passes.type_inference import TypeInferencePass
from ..passes.range_analysis import RangeAnalysisPass
from ..analysis.module_system import ModuleSystem

class CompilationResult:
    """Compilation result"""
    def __init__(
        self,
        ir: Optional[ProgramIR] = None,
        tcx: Optional[TyCtxt] = None,
        success: bool = False
    ):
        self.ir = ir
        self.tcx = tcx
        self.success = success

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
        8. CastValidationPass
        9. PipelineTypeValidationPass
        10. ExhaustivenessPass
        
        Alignment:
        - NameResolutionPass runs on AST before lowering (manual call)
        - ASTToIRLoweringPass converts AST to IR
        - Then follow order on IR
        """
        # 0. AST to IR lowering
        self.pass_manager.register_pass(ASTToIRLoweringPass)
        # Note: Einstein grouping runs on AST as part of NameResolutionPass
        
        # 2. Rest pattern preprocessing (expands ..batch to batch.0 early)
        from ..passes.rest_pattern_preprocessing import RestPatternPreprocessingPass
        self.pass_manager.register_pass(RestPatternPreprocessingPass)
        
        # 3. Range analysis (infers ranges for loop variables)
        # CRITICAL: Must come before shape analysis (shape needs ranges for offsets)
        self.pass_manager.register_pass(RangeAnalysisPass)
        
        # 4. Shape analysis (uses ranges to compute output dimensions)
        from ..passes.shape_analysis import UnifiedShapeAnalysisPass
        self.pass_manager.register_pass(UnifiedShapeAnalysisPass)
        
        # 5. Type inference (Type analysis runs after shape)
        # This allows type inference to use shape information
        self.pass_manager.register_pass(TypeInferencePass)
        
        # 6. Einstein lowering (lower Einstein declarations to loops)
        # After type inference so it can access type_info for element_type
        from ..passes.einstein_lowering import EinsteinLoweringPass
        self.pass_manager.register_pass(EinsteinLoweringPass)
        
        # 7. Validation passes
        from ..passes.cast_validation import CastValidationPass
        self.pass_manager.register_pass(CastValidationPass)
        
        from ..passes.pipeline_validation import PipelineTypeValidationPass
        self.pass_manager.register_pass(PipelineTypeValidationPass)
        
        from ..passes.exhaustiveness import ExhaustivenessPass
        self.pass_manager.register_pass(ExhaustivenessPass)
        
        # 8. IR validation (validation)
        from ..passes.ir_validation import IRValidationPass
        self.pass_manager.register_pass(IRValidationPass)
        
        # 9. Optimizations (validation)
        from ..passes.arrow_optimization import ArrowOptimizationPass
        self.pass_manager.register_pass(ArrowOptimizationPass)

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

        try:
            # Phase 1: Parsing (source → AST)
            ast = self.parser.parse(source, source_file)
            
            if root_path is None:
                root_path = Path.cwd()
            module_system = ModuleSystem(root_path, tcx.resolver)
            tcx.module_system = module_system
            tcx.discovered_modules = {}
            tcx.source_files[source_file] = source

            ast = self._run_name_resolution_pass(ast, tcx)
            if tcx.reporter.has_errors():
                return CompilationResult(success=False, tcx=tcx)
            
            # Phase 3-7: Passes (all on IR after lowering)
            from ..passes.ast_to_ir import ASTToIRLoweringPass
            lowering_pass = ASTToIRLoweringPass()
            ir = lowering_pass.run(ast, tcx)

            if os.environ.get("EINLANG_DUMP_IR_PER_PASS"):
                from ..ir.serialization import serialize_ir
                ir_dump_dir = Path("ir_dump")
                ir_dump_dir.mkdir(parents=True, exist_ok=True)
                try:
                    (ir_dump_dir / "after_ast_to_ir_lowering.sexpr").write_text(serialize_ir(ir), encoding="utf-8")
                except Exception as e:
                    import traceback
                    traceback.print_exc()

            # Check if we should stop after lowering
            if stop_after_pass == 'ASTToIRLoweringPass':
                return CompilationResult(success=True, ir=ir, tcx=tcx)

            dump_ir_per_pass = os.environ.get("EINLANG_DUMP_IR_PER_PASS", "")
            if dump_ir_per_pass:
                from ..ir.serialization import serialize_ir
                dump_dir = Path("ir_dumps")
                dump_dir.mkdir(parents=True, exist_ok=True)
                try:
                    (dump_dir / "00_after_ASTToIRLoweringPass.sexpr").write_text(serialize_ir(ir), encoding="utf-8")
                except Exception as e:
                    import traceback
                    traceback.print_exc()

            # Run remaining passes on IR
            # Filter out ASTToIRLoweringPass from pass manager (already run)
            remaining_passes = [
                p for p in self.pass_manager.passes 
                if p != ASTToIRLoweringPass
            ]
            
            # Run remaining passes using pass manager (handles dependencies automatically)
            # Design Pattern: Use pass manager for dependency resolution (no manual isinstance checks)
            dump_ir_per_pass = os.environ.get("EINLANG_DUMP_IR_PER_PASS", "")
            dump_dir = Path("ir_dumps") if dump_ir_per_pass else None
            pass_index = 1
            for pass_class in remaining_passes:
                pass_instance = pass_class()
                try:
                    ir = pass_instance.run(ir, tcx)
                except RecursionError as e:
                    raise

                if dump_dir is not None:
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    if pass_index == 1:
                        readme = dump_dir / "README.txt"
                        if not readme.exists():
                            readme.write_text(
                                "IR S-expr dumps per pass (EINLANG_DUMP_IR_PER_PASS=1).\n"
                                "00 = after ASTToIRLoweringPass, 01 = after RestPatternPreprocessingPass, etc.\n"
                                "Reductions include :loop_var_ranges only when non-empty.\n"
                                "Compare dumps to see which pass stops setting loop_var_ranges on specialized functions.\n",
                                encoding="utf-8",
                            )
                    from ..ir.serialization import serialize_ir
                    try:
                        out_path = dump_dir / f"{pass_index:02d}_after_{pass_class.__name__}.sexpr"
                        out_path.write_text(serialize_ir(ir), encoding="utf-8")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                if pass_class.__name__ == "EinsteinLoweringPass" and os.environ.get("EINLANG_DUMP_IR_AFTER_EINSTEIN_LOWERING", ""):
                    from ..ir.serialization import serialize_ir
                    from ..ir.nodes import ProgramIR
                    dump_dir = Path("ir_dumps")
                    dump_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        func_map = getattr(tcx, "function_ir_map", None) or {}
                        extra = [f for f in func_map.values() if f is not None and f not in ir.functions]
                        combined = ProgramIR(
                            modules=getattr(ir, "modules", []) or [],
                            functions=list(ir.functions) + extra,
                            constants=getattr(ir, "constants", []) or [],
                            statements=getattr(ir, "statements", []) or [],
                            source_files=getattr(ir, "source_files", {}) or {},
                            defid_to_name=getattr(ir, "defid_to_name", None),
                        )
                        (dump_dir / "after_einstein_lowering.sexpr").write_text(serialize_ir(combined), encoding="utf-8")
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                pass_index += 1

                # Stop after specified pass if requested
                if stop_after_pass and pass_class.__name__ == stop_after_pass:
                    break

            # Check for errors
            if tcx.reporter.has_errors():
                return CompilationResult(success=False, tcx=tcx)
            
            function_ir_map = getattr(tcx, 'function_ir_map', None) or {}
            func_set = {id(f) for f in ir.functions}
            for f in function_ir_map.values():
                if f is not None and id(f) not in func_set:
                    ir.functions.append(f)
                    func_set.add(id(f))
            
            from ..passes.tree_shaking import tree_shake
            ir = tree_shake(ir)

            if os.environ.get("EINLANG_DUMP_FINAL_IR"):
                from ..ir.serialization import serialize_ir
                ir_dump_dir = Path("ir_dump")
                ir_dump_dir.mkdir(parents=True, exist_ok=True)
                try:
                    (ir_dump_dir / "final_ir.sexpr").write_text(serialize_ir(ir), encoding="utf-8")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
            return CompilationResult(ir=ir, tcx=tcx, success=True)
        
        except ParseError as e:
            from ..shared.source_location import SourceLocation
            if e.location is not None:
                loc = e.location
                span = SourceLocation(
                    file=getattr(loc, "file", source_file),
                    line=getattr(loc, "line", 1),
                    column=getattr(loc, "column", 1),
                    end_line=getattr(loc, "line", 1),
                    end_column=getattr(loc, "column", 1) + 1,
                )
            else:
                span = SourceLocation(
                    file=source_file, line=1, column=1,
                    end_line=1, end_column=1,
                )
            tcx.reporter.report_error(e.message, span)
            return CompilationResult(success=False, tcx=tcx)
        except Exception as e:
            if not tcx.reporter.has_errors():
                from ..shared.source_location import SourceLocation
                location = SourceLocation(
                    file=source_file,
                    line=1,
                    column=1,
                    end_line=1,
                    end_column=1
                )
                tcx.reporter.report_error(
                    str(e),
                    location
                )
            return CompilationResult(success=False, tcx=tcx)

