"""
Base Pass System

Rust Pattern: rustc_mir::transform::MirPass
Reference: PASS_SYSTEM_DESIGN.md
"""

from abc import ABC, abstractmethod
from typing import List, Type, TypeVar, Generic, Any, Optional, Dict, Tuple
from ..ir.nodes import ProgramIR


T = TypeVar('T')


class TyCtxt:
    """
    Type context - single source of truth for all compiler state (Rust naming: rustc_middle::ty::TyCtxt).
    
    Rust Pattern: rustc_middle::ty::TyCtxt
    
    Implementation Alignment: Follows Rust's `rustc_middle::ty::TyCtxt` pattern:
    - Single source of truth for all compiler state
    - Analysis results stored here (not in passes)
    - Immutable data structures
    - Query-based access (we use explicit dependencies for)
    
    Scoped storage: DefId and name maps live on TyCtxt (compilation scope), not global dicts.
    Resolver allocates DefIds and writes into these maps; all lookups go through tcx.
    
    Reference: `rustc_middle::ty::TyCtxt` is the single context in Rust compiler
    
    Note: Using Rust naming - "TyCtxt" instead of "Context" to match rustc
    """
    
    def __init__(self):
        from ..shared.defid import Resolver, DefId, DefType
        from ..shared.errors import ErrorReporter
        
        self.def_registry: Dict[DefId, Tuple[DefType, Any]] = {}
        self.symbol_table: Dict[Tuple[Tuple[str, ...], str, DefType], DefId] = {}
        self.alias_table: Dict[str, Tuple[str, ...]] = {}
        
        self.resolver: Resolver = Resolver(self)
        
        self._analysis_results: Dict[Type['BasePass'], Any] = {}
        
        # Error reporter
        self.reporter: ErrorReporter = ErrorReporter({})
        
        # Source information
        self.source_files: Dict[str, str] = {}
        
        # Module system (for lazy loading / tree-shaking)
        self.module_system: Optional[Any] = None  # ModuleSystem instance
        self.discovered_modules: Dict[Any, Any] = {}  # Discovered module paths (not loaded yet)
    
    def get_definition(self, defid: Any) -> Optional[Tuple[Any, Any]]:
        return self.def_registry.get(defid)
    
    def get_analysis(self, pass_class: Type['BasePass']) -> Any:
        """Get analysis results from a pass"""
        if pass_class not in self._analysis_results:
            raise RuntimeError(f"Analysis {pass_class} not available")
        return self._analysis_results[pass_class]
    
    def set_analysis(self, pass_class: Type['BasePass'], results: Any) -> None:
        """Store analysis results"""
        self._analysis_results[pass_class] = results


class BasePass(ABC):
    """
    Base class for all passes (all operate on IR).
    
    Rust Pattern: rustc_mir::transform::MirPass
    
    Implementation Alignment: Follows Rust's pass interface:
    - All passes operate on IR (not AST)
    - Explicit dependencies via `requires`
    - Pass results stored in TyCtxt (not in pass)
    - Immutable IR (passes return new IR)
    
    Reference: `rustc_mir::transform::MirPass` for pass interface
    """
    requires: List[Type['BasePass']] = []  # Dependencies (empty by default)
    
    @abstractmethod
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Run pass on IR.
        
        Rust Pattern: rustc_mir::transform::MirPass::run()
        
        Returns: New IR (immutable - passes create new IR nodes)
        """
        raise NotImplementedError


class PassManager:
    """
    Pass manager with dependency resolution.
    
    Rust Pattern: rustc driver with pass scheduling
    
    Implementation Alignment: Follows Rust's pass scheduling:
    - Automatic dependency resolution (topological sort)
    - Passes run in dependency order
    - Single TyCtxt shared across all passes
    
    Reference: Rust driver pass scheduling
    """
    
    def __init__(self):
        self.passes: List[Type[BasePass]] = []
        self._dependency_graph: dict[Type[BasePass], set[Type[BasePass]]] = {}
    
    def register_pass(self, pass_class: Type[BasePass]) -> None:
        """Register a pass"""
        self.passes.append(pass_class)
        self._dependency_graph[pass_class] = set(pass_class.requires)
    
    def run_all(self, ir: ProgramIR, tcx: TyCtxt, dump_ir: bool = False) -> ProgramIR:
        """
        Run all passes in dependency order.
        
        Rust Pattern: rustc driver schedules passes based on dependencies
        
        Args:
            ir: Input IR
            tcx: Type context
            dump_ir: If True, dump IR S-expression after each pass for debugging
        """
        sorted_passes = self._topological_sort()
        
        for pass_class in sorted_passes:
            pass_instance = pass_class()
            pass_name = pass_class.__name__
            
            ir = pass_instance.run(ir, tcx)
            
            # Dump IR S-expression after pass (for debugging)
            if dump_ir:
                try:
                    from ..ir.serialization import IRSerializer
                    serializer = IRSerializer(include_type_info=False)
                    ir_sexpr = serializer.serialize(ir)
                    
                    # Find first Einstein declaration to show
                    einstein_start = ir_sexpr.find('(einstein-declaration')
                    if einstein_start >= 0:
                        einstein_end = min(einstein_start + 1500, len(ir_sexpr))
                        snippet = ir_sexpr[einstein_start:einstein_end]
                    else:
                        snippet = ir_sexpr[:1000] if len(ir_sexpr) > 1000 else ir_sexpr
                    
                    print(f"\n{'='*80}")
                    print(f"After {pass_name}:")
                    print(f"{'='*80}")
                    print(snippet)
                    print(f"{'='*80}\n")
                except Exception as e:
                    print(f"Warning: Could not dump IR after {pass_name}: {e}")
        
        return ir
    
    def _topological_sort(self) -> List[Type[BasePass]]:
        """Topological sort of passes by dependencies"""
        in_degree = {p: len(self._dependency_graph[p]) for p in self.passes}
        queue = [p for p, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            pass_class = queue.pop(0)
            result.append(pass_class)
            
            for other_pass in self.passes:
                if pass_class in self._dependency_graph[other_pass]:
                    in_degree[other_pass] -= 1
                    if in_degree[other_pass] == 0:
                        queue.append(other_pass)
        
        if len(result) != len(self.passes):
            raise RuntimeError("Circular dependency detected in passes")
        
        return result

