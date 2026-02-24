"""
Arrow Graph Optimization Pass

Rust Pattern: rustc_mir::transform::MirPass (optimization)
Reference: PASS_SYSTEM_DESIGN.md

Optimization passes for arrow expressions (computation graphs).

Three optimization phases:
1. Fusion: Detect and fuse compatible sequential operations (conv + batch_norm + relu)
2. Parallelization: Identify parallel execution opportunities
3. Memory optimization: Tensor lifetime analysis and memory reuse
"""

import logging
from typing import List, Dict, Set, Optional, Any, Tuple
from ..passes.base import BasePass, TyCtxt
from ..passes.type_inference import TypeInferencePass
from ..ir.nodes import (
    ProgramIR, ExpressionIR, ArrowExpressionIR, FunctionCallIR,
    IRVisitor, IRNode, IdentifierIR, FunctionDefIR, ConstantDefIR,
    EinsteinDeclarationIR
)

logger = logging.getLogger("einlang.passes.arrow_optimization")


class FusionOpportunity:
    """Represents a fusion opportunity in sequential arrow chain"""
    def __init__(self, start_idx: int, end_idx: int, components: List[ExpressionIR], reason: str):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.components = components
        self.reason = reason


class ParallelOpportunity:
    """Represents a parallel execution opportunity"""
    def __init__(self, components: List[ExpressionIR], dependencies: Dict[int, Set[int]]):
        self.components = components
        self.dependencies = dependencies  # component_idx -> set of dependent indices


class TensorLifetime:
    """Represents tensor lifetime information"""
    def __init__(self, tensor_id: str, created_at: int, last_used_at: int, size: Optional[int] = None):
        self.tensor_id = tensor_id
        self.created_at = created_at
        self.last_used_at = last_used_at
        self.size = size  # Estimated size in bytes
    
    @property
    def lifetime(self) -> int:
        return self.last_used_at - self.created_at


class ArrowOptimizationVisitor(IRVisitor[ExpressionIR]):
    """
    Visitor to analyze and optimize arrow expressions.
    
    Performs:
    1. Fusion analysis for sequential arrows
    2. Parallelization analysis for parallel/fanout arrows
    3. Memory lifetime analysis
    """
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.fusion_opportunities: List[FusionOpportunity] = []
        self.parallel_opportunities: List[ParallelOpportunity] = []
        self.tensor_lifetimes: List[TensorLifetime] = []
        self.component_index = 0
    
    def visit_program(self, node: ProgramIR) -> ExpressionIR:
        """Visit program - not used for arrow optimization (expressions only)"""
        from ..ir.nodes import LiteralIR
        from ..shared.source_location import SourceLocation
        return LiteralIR(value=None, location=SourceLocation("", 0, 0, 0, 0))
    
    def visit_arrow_expression(self, node: ArrowExpressionIR) -> ExpressionIR:
        """Analyze and optimize arrow expression"""
        operator = node.operator
        
        # Phase 1: Fusion (for sequential arrows)
        if operator == ">>>":
            optimized = self._optimize_sequential(node)
            if optimized is not None:
                return optimized
        
        # Phase 2: Parallelization (for parallel and fanout arrows)
        elif operator in ("***", "&&&"):
            optimized = self._optimize_parallel(node)
            if optimized is not None:
                return optimized
        
        # Phase 3: Memory optimization (for all arrow types)
        self._analyze_memory_lifetimes(node)
        
        # Recursively optimize components
        optimized_components = [comp.accept(self) for comp in node.components]
        
        # If components changed, create new arrow expression
        if optimized_components != node.components:
            return ArrowExpressionIR(
                components=optimized_components,
                operator=operator,
                location=node.location,
                type_info=node.type_info,
                shape_info=node.shape_info
            )
        
        return node
    
    def _optimize_sequential(self, node: ArrowExpressionIR) -> Optional[ExpressionIR]:
        """Optimize sequential arrow chain through fusion"""
        if len(node.components) < 2:
            return None
        
        # Detect fusion opportunities
        fusion_ops = self._detect_fusion_opportunities(node.components)
        
        if not fusion_ops:
            return None
        
        # Apply fusion (merge compatible operations)
        optimized_components = []
        i = 0
        fused_count = 0
        
        while i < len(node.components):
            # Check if this component is part of a fusion opportunity
            fused = False
            for fusion_op in fusion_ops:
                if fusion_op.start_idx == i:
                    # Create fused operation
                    fused_comp = self._create_fused_component(fusion_op)
                    optimized_components.append(fused_comp)
                    i = fusion_op.end_idx + 1
                    fused = True
                    fused_count += 1
                    break
            
            if not fused:
                optimized_components.append(node.components[i])
                i += 1
        
        if fused_count > 0:
            logger.debug(f"Fused {fused_count} sequential arrow operations")
            return ArrowExpressionIR(
                components=optimized_components,
                operator=node.operator,
                location=node.location,
                type_info=node.type_info,
                shape_info=node.shape_info
            )
        
        return None
    
    def _detect_fusion_opportunities(self, components: List[ExpressionIR]) -> List[FusionOpportunity]:
        """Detect fusion opportunities in sequential chain"""
        opportunities = []
        
        # Pattern 1: conv >>> batch_norm >>> activation
        # Pattern 2: linear >>> activation
        # Pattern 3: batch_norm >>> activation
        
        i = 0
        while i < len(components) - 1:
            current = components[i]
            next_comp = components[i + 1]
            
            # Check if current and next are function calls
            if isinstance(current, FunctionCallIR) and isinstance(next_comp, FunctionCallIR):
                # Pattern: conv/batch_norm followed by activation
                if self._is_activation_function(next_comp.function_name):
                    # Check if previous is conv or batch_norm
                    if i > 0 and isinstance(components[i-1], FunctionCallIR):
                        prev = components[i-1]
                        if self._is_conv_or_linear(prev.function_name) or self._is_batch_norm(prev.function_name):
                            # Found: conv/batch_norm >>> activation
                            opportunities.append(FusionOpportunity(
                                start_idx=i-1,
                                end_idx=i+1,
                                components=components[i-1:i+2],
                                reason="conv/batch_norm + activation fusion"
                            ))
                            i += 2
                            continue
                
                # Pattern: batch_norm followed by activation
                if self._is_batch_norm(current.function_name) and self._is_activation_function(next_comp.function_name):
                    opportunities.append(FusionOpportunity(
                        start_idx=i,
                        end_idx=i+1,
                        components=components[i:i+2],
                        reason="batch_norm + activation fusion"
                    ))
                    i += 2
                    continue
            
            i += 1
        
        return opportunities
    
    def _is_activation_function(self, name: str) -> bool:
        """Check if function is an activation function (fusible)"""
        activation_functions = {
            'relu', 'sigmoid', 'tanh', 'gelu', 'swish', 'elu',
            'leaky_relu', 'prelu', 'softmax', 'softplus'
        }
        return name.lower() in activation_functions
    
    def _is_conv_or_linear(self, name: str) -> bool:
        """Check if function is convolution or linear layer"""
        conv_linear_functions = {
            'conv', 'conv2d', 'conv1d', 'conv3d',
            'linear', 'dense', 'fully_connected'
        }
        return name.lower() in conv_linear_functions
    
    def _is_batch_norm(self, name: str) -> bool:
        """Check if function is batch normalization"""
        batch_norm_functions = {
            'batch_norm', 'batchnorm', 'batch_normalization',
            'layer_norm', 'layernorm', 'layer_normalization'
        }
        return name.lower() in batch_norm_functions
    
    def _create_fused_component(self, fusion_op: FusionOpportunity) -> ExpressionIR:
        """Create a fused component from fusion opportunity"""
        # For now, create a function call to a fused operation
        # In a full implementation, this would create a specialized fused kernel
        # We'll mark it for backend optimization by creating a special function call
        
        # Get the first component as base
        first_comp = fusion_op.components[0]
        
        if isinstance(first_comp, FunctionCallIR):
            # Create fused function call
            fused_name = f"fused_{first_comp.function_name}"
            # Combine arguments from all components
            all_args = []
            for comp in fusion_op.components:
                if isinstance(comp, FunctionCallIR):
                    all_args.extend(comp.arguments)
            
            return FunctionCallIR(
                function_name=fused_name,
                function_defid=first_comp.function_defid,  # Use first component's DefId
                arguments=all_args,
                module_path=first_comp.module_path,  # Preserve module_path for Python module calls
                location=first_comp.location,
                type_info=first_comp.type_info,
                shape_info=first_comp.shape_info
            )
        
        # Fallback: return first component
        return first_comp
    
    def _optimize_parallel(self, node: ArrowExpressionIR) -> Optional[ExpressionIR]:
        """Optimize parallel/fanout arrows through parallelization analysis"""
        if len(node.components) < 2:
            return None
        
        # Analyze dependencies
        dependencies = self._analyze_dependencies(node.components)
        
        # Identify independent operations
        independent_groups = self._group_independent_operations(node.components, dependencies)
        
        if len(independent_groups) > 1:
            # Multiple independent groups - can be parallelized
            parallel_op = ParallelOpportunity(node.components, dependencies)
            self.parallel_opportunities.append(parallel_op)
            
            # Mark for parallel execution (add metadata to IR node)
            # We'll store this in shape_info or add a new field
            # For now, we'll create an optimized version with parallel hints
            logger.debug(f"Identified {len(independent_groups)} independent operation groups for parallelization")
            
            # Return node with parallelization metadata
            # Note: In a full implementation, this would transform the IR to use parallel execution
            # For now, we'll just mark it and let the backend handle it
            return node
        
        return None
    
    def _analyze_dependencies(self, components: List[ExpressionIR]) -> Dict[int, Set[int]]:
        """Analyze data dependencies between components"""
        dependencies: Dict[int, Set[int]] = {i: set() for i in range(len(components))}
        
        # Simple dependency analysis: if component i's output is used by component j, j depends on i
        # In a full implementation, this would do proper dataflow analysis
        
        for i, comp_i in enumerate(components):
            # Get output identifiers from comp_i
            output_ids = self._get_output_identifiers(comp_i)
            
            for j, comp_j in enumerate(components):
                if i >= j:
                    continue  # Only check forward dependencies
                
                # Get input identifiers from comp_j
                input_ids = self._get_input_identifiers(comp_j)
                
                # If there's overlap, j depends on i
                if output_ids & input_ids:
                    dependencies[j].add(i)
        
        return dependencies
    
    def _get_output_identifiers(self, expr: ExpressionIR) -> Set[str]:
        """Extract output identifier names from expression"""
        # Simple implementation: extract function names and identifiers
        ids = set()
        
        if isinstance(expr, FunctionCallIR):
            ids.add(expr.function_name)
            # Also check arguments for identifiers
            for arg in expr.arguments:
                if isinstance(arg, IdentifierIR):
                    ids.add(arg.name)
        
        return ids
    
    def _get_input_identifiers(self, expr: ExpressionIR) -> Set[str]:
        """Extract input identifier names from expression"""
        # Similar to output identifiers
        return self._get_output_identifiers(expr)
    
    def _group_independent_operations(self, components: List[ExpressionIR], 
                                     dependencies: Dict[int, Set[int]]) -> List[List[int]]:
        """Group operations into independent sets"""
        # Simple grouping: operations with no dependencies on each other
        groups: List[List[int]] = []
        assigned = set()
        
        for i in range(len(components)):
            if i in assigned:
                continue
            
            # Start new group
            group = [i]
            assigned.add(i)
            
            # Find other operations that don't depend on this one and vice versa
            for j in range(i + 1, len(components)):
                if j in assigned:
                    continue
                
                # Check if i and j are independent (no dependencies between them)
                if i not in dependencies.get(j, set()) and j not in dependencies.get(i, set()):
                    group.append(j)
                    assigned.add(j)
            
            groups.append(group)
        
        return groups
    
    def _analyze_memory_lifetimes(self, node: ArrowExpressionIR) -> None:
        """Analyze tensor lifetimes for memory optimization"""
        # Track when tensors are created and last used
        tensor_creations: Dict[str, int] = {}
        tensor_last_use: Dict[str, int] = {}
        
        for idx, comp in enumerate(node.components):
            # Identify tensor outputs
            if isinstance(comp, FunctionCallIR) and comp.type_info:
                # Estimate tensor size from type_info
                size = self._estimate_tensor_size(comp.type_info)
                
                # Create tensor ID from function name and index
                tensor_id = f"{comp.function_name}_{idx}"
                
                if tensor_id not in tensor_creations:
                    tensor_creations[tensor_id] = idx
                tensor_last_use[tensor_id] = idx
        
        # Create lifetime records
        for tensor_id, created_at in tensor_creations.items():
            last_used = tensor_last_use.get(tensor_id, created_at)
            size = None  # Would be estimated from type_info
            
            lifetime = TensorLifetime(
                tensor_id=tensor_id,
                created_at=created_at,
                last_used_at=last_used,
                size=size
            )
            self.tensor_lifetimes.append(lifetime)
    
    def _estimate_tensor_size(self, type_info: Any) -> Optional[int]:
        """Estimate tensor size in bytes from type_info"""
        # Placeholder: would analyze type_info to estimate size
        # For now, return None
        return None
    
    # Default visitor methods (delegate to children)
    def visit_literal(self, node) -> ExpressionIR:
        return node
    
    def visit_identifier(self, node) -> ExpressionIR:
        return node
    
    def visit_binary_op(self, node) -> ExpressionIR:
        return node
    
    def visit_unary_op(self, node) -> ExpressionIR:
        return node
    
    def visit_function_call(self, node) -> ExpressionIR:
        return node
    
    def visit_builtin_call(self, node) -> ExpressionIR:
        return node
    
    def visit_rectangular_access(self, node) -> ExpressionIR:
        return node
    
    def visit_jagged_access(self, node) -> ExpressionIR:
        return node
    
    def visit_array_literal(self, node) -> ExpressionIR:
        return node
    
    def visit_tuple_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_reduction_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_block_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_if_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_match_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_literal_pattern(self, node) -> Any:
        return node
    
    def visit_identifier_pattern(self, node) -> Any:
        return node
    
    def visit_wildcard_pattern(self, node) -> Any:
        return node
    
    def visit_tuple_pattern(self, node) -> Any:
        return node
    
    def visit_rest_pattern(self, node) -> Any:
        return node
    
    def visit_array_pattern(self, node) -> Any:
        return node
    
    def visit_guard_pattern(self, node) -> Any:
        return node
    
    def visit_cast_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_try_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_lambda(self, node) -> ExpressionIR:
        return node
    
    def visit_range(self, node) -> ExpressionIR:
        return node
    
    def visit_array_comprehension(self, node) -> ExpressionIR:
        return node
    
    def visit_pipeline_expression(self, node) -> ExpressionIR:
        return node
    
    def visit_tuple_access(self, node) -> ExpressionIR:
        return node
    
    def visit_interpolated_string(self, node) -> ExpressionIR:
        return node
    
    def visit_function_ref(self, node) -> ExpressionIR:
        return node
    
    def visit_member_access(self, node) -> ExpressionIR:
        return node
    
    # Required visitor methods for IRVisitor interface
    def visit_function_def(self, node) -> ExpressionIR:
        # ArrowOptimizationVisitor operates on expressions, not function definitions
        # This should not be called, but required by interface
        raise NotImplementedError("ArrowOptimizationVisitor should not visit function definitions")
    
    def visit_constant_def(self, node) -> ExpressionIR:
        # ArrowOptimizationVisitor operates on expressions, not constant definitions
        # This should not be called, but required by interface
        raise NotImplementedError("ArrowOptimizationVisitor should not visit constant definitions")
    
    def visit_einstein_declaration(self, node) -> ExpressionIR:
        # ArrowOptimizationVisitor operates on expressions, not Einstein declarations
        # This should not be called, but required by interface
        raise NotImplementedError("ArrowOptimizationVisitor should not visit Einstein declarations")
    
    def visit_module(self, node) -> ExpressionIR:
        # ArrowOptimizationVisitor operates on expressions, not modules
        # This should not be called, but required by interface
        raise NotImplementedError("ArrowOptimizationVisitor should not visit modules")
    
    def visit_where_expression(self, node) -> ExpressionIR:
        from ..ir.nodes import WhereExpressionIR
        new_constraints = [c.accept(self) for c in node.constraints]
        return WhereExpressionIR(
            expr=node.expr.accept(self),
            constraints=new_constraints,
            location=node.location,
            type_info=node.type_info,
            shape_info=node.shape_info
        )


    def visit_variable_declaration(self, node) -> ExpressionIR:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None

class ArrowOptimizationInPlaceVisitor(IRVisitor[None]):
    """
    Visitor that optimizes arrow expressions in place.
    Uses ArrowOptimizationVisitor for expression optimization.
    """
    
    def __init__(self, tcx: TyCtxt):
        self.tcx = tcx
        self.optimizer = ArrowOptimizationVisitor(tcx)
        # Share state with optimizer
        self.fusion_opportunities = self.optimizer.fusion_opportunities
        self.parallel_opportunities = self.optimizer.parallel_opportunities
        self.tensor_lifetimes = self.optimizer.tensor_lifetimes
    
    def visit_program(self, node: ProgramIR) -> None:
        """Visit program and optimize all functions, constants, and statements in place."""
        # Optimize all functions in place
        for func in node.functions:
            func.accept(self)
        
        # Optimize all constants in place
        for const in node.constants:
            const.accept(self)
        
        # Optimize top-level statements in place - use visitor pattern
        for stmt in node.statements:
            stmt.accept(self)
    
    def visit_function_def(self, node: FunctionDefIR) -> None:
        """Optimize function body in place"""
        optimized_body = node.body.accept(self.optimizer)
        if optimized_body is not node.body:
            node.body = optimized_body
    
    def visit_constant_def(self, node: ConstantDefIR) -> None:
        """Optimize constant value in place"""
        optimized_value = node.value.accept(self.optimizer)
        if optimized_value is not node.value:
            node.value = optimized_value
    
    def visit_einstein_declaration(self, node: EinsteinDeclarationIR) -> None:
        """Optimize Einstein declaration value in place"""
        optimized_value = node.value.accept(self.optimizer)
        if optimized_value is not node.value:
            node.value = optimized_value
    
    # Required visitor methods (void visitor, no-op for expressions)
    def visit_literal(self, node) -> None:
        pass
    
    def visit_identifier(self, node) -> None:
        pass
    
    def visit_binary_op(self, node) -> None:
        pass
    
    def visit_function_call(self, node) -> None:
        pass
    
    def visit_unary_op(self, node) -> None:
        pass
    
    def visit_rectangular_access(self, node) -> None:
        pass
    
    def visit_jagged_access(self, node) -> None:
        pass
    
    def visit_block_expression(self, node) -> None:
        pass
    
    def visit_if_expression(self, node) -> None:
        pass
    
    def visit_lambda(self, node) -> None:
        pass
    
    def visit_range(self, node) -> None:
        pass
    
    def visit_array_comprehension(self, node) -> None:
        pass
    
    def visit_array_literal(self, node) -> None:
        pass
    
    def visit_tuple_expression(self, node) -> None:
        pass
    
    def visit_tuple_access(self, node) -> None:
        pass
    
    def visit_interpolated_string(self, node) -> None:
        pass
    
    def visit_cast_expression(self, node) -> None:
        pass
    
    def visit_member_access(self, node) -> None:
        pass
    
    def visit_try_expression(self, node) -> None:
        pass
    
    def visit_match_expression(self, node) -> None:
        pass
    
    def visit_reduction_expression(self, node) -> None:
        pass
    
    def visit_where_expression(self, node) -> None:
        pass
    
    def visit_arrow_expression(self, node) -> None:
        pass
    
    def visit_pipeline_expression(self, node) -> None:
        pass
    
    def visit_builtin_call(self, node) -> None:
        pass
    
    def visit_function_ref(self, node) -> None:
        pass
    
    def visit_literal_pattern(self, node) -> None:
        pass
    
    def visit_identifier_pattern(self, node) -> None:
        pass
    
    def visit_wildcard_pattern(self, node) -> None:
        pass
    
    def visit_tuple_pattern(self, node) -> None:
        pass
    
    def visit_array_pattern(self, node) -> None:
        pass
    
    def visit_rest_pattern(self, node) -> None:
        pass
    
    def visit_guard_pattern(self, node) -> None:
        pass
    
    def visit_module(self, node) -> None:
        pass



    def visit_variable_declaration(self, node) -> Any:
        """Visit variable declaration - recurse into value"""
        if hasattr(node, 'value') and node.value:
            return node.value.accept(self)
        return None
class ArrowOptimizationPass(BasePass):
    """
    Optimization pass for arrow expressions.
    
    Three optimization phases:
    1. Fusion: Merge compatible sequential operations
    2. Parallelization: Schedule parallel operations
    3. Memory optimization: Reuse tensor memory
    """
    requires = [TypeInferencePass]  # Needs type information for optimization
    
    def run(self, ir: ProgramIR, tcx: TyCtxt) -> ProgramIR:
        """
        Optimize arrow expressions in IR.
        
        Rust Pattern: Optimization pass transforms IR in place
        """
        logger.debug("Starting arrow optimization")
        
        # Create visitor that modifies in place
        visitor = ArrowOptimizationInPlaceVisitor(tcx)
        
        # Use visitor pattern: ir.accept(visitor) modifies in place
        ir.accept(visitor)
        
        # Log optimization results
        if visitor.fusion_opportunities:
            logger.debug(f"Found {len(visitor.fusion_opportunities)} fusion opportunities")
        if visitor.parallel_opportunities:
            logger.debug(f"Found {len(visitor.parallel_opportunities)} parallelization opportunities")
        if visitor.tensor_lifetimes:
            logger.debug(f"Analyzed {len(visitor.tensor_lifetimes)} tensor lifetimes")
        
        # Store optimization results in TyCtxt
        tcx.set_analysis(ArrowOptimizationPass, {
            'fusion_opportunities': visitor.fusion_opportunities,
            'parallel_opportunities': visitor.parallel_opportunities,
            'tensor_lifetimes': visitor.tensor_lifetimes
        })
        
        # Return same IR object (modified in place)
        return ir

