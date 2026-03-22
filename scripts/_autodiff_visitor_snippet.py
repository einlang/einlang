# Temporary: content spliced into autodiff.py by scripts/apply_autodiff_visitor_patch.py
SUBSTITUTION_AND_HELPERS = r'''
class _Rewriter(IRVisitor[ExpressionIR]):
    """Default tree rewrite: rebuild compound nodes via ``child.accept(self)``."""

    def __init__(self, loc: SourceLocation) -> None:
        self._loc = loc

    def _rw_wc(self, wc: Any) -> Any:
        if wc is None:
            return None
        cc = getattr(wc, "constraints", None)
        if not cc:
            return wc
        return WhereClauseIR(constraints=[c.accept(self) for c in cc], location=self._loc)

    def _rw_lr(self, lr: Optional[Dict]) -> Optional[Dict]:
        if not lr:
            return lr
        out: Dict = {}
        for k, v in lr.items():
            if isinstance(v, RangeIR):
                out[k] = RangeIR(
                    start=v.start.accept(self),
                    end=v.end.accept(self),
                    location=v.location or self._loc,
                    type_info=v.type_info,
                )
            else:
                out[k] = v
        return out

    def visit_literal(self, n: LiteralIR) -> ExpressionIR:
        return n

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        return n

    def visit_index_var(self, n: IndexVarIR) -> ExpressionIR:
        return n

    def visit_index_rest(self, n: IndexRestIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binary_op(self, n: BinaryOpIR) -> ExpressionIR:
        loc = n.location or self._loc
        return BinaryOpIR(
            n.operator,
            n.left.accept(self),
            n.right.accept(self),
            loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_unary_op(self, n: UnaryOpIR) -> ExpressionIR:
        return UnaryOpIR(
            n.operator,
            n.operand.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_function_call(self, n: FunctionCallIR) -> ExpressionIR:
        return FunctionCallIR(
            callee_expr=n.callee_expr,
            location=n.location or self._loc,
            arguments=[a.accept(self) for a in (n.arguments or [])],
            module_path=getattr(n, "module_path", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_rectangular_access(self, n: RectangularAccessIR) -> ExpressionIR:
        return RectangularAccessIR(
            n.array.accept(self),
            [i.accept(self) for i in (n.indices or [])],
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_jagged_access(self, n: JaggedAccessIR) -> ExpressionIR:
        return n

    def visit_block_expression(self, n: BlockExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        ns: List[Any] = []
        for s in n.statements or []:
            if isinstance(s, BindingIR) and s.expr is not None:
                ns.append(
                    BindingIR(
                        name=s.name,
                        expr=s.expr.accept(self),
                        location=s.location,
                        defid=s.defid,
                        type_info=_ti(s),
                    )
                )
            else:
                ns.append(s)
        nf = n.final_expr.accept(self) if n.final_expr is not None else None
        return BlockExpressionIR(ns, loc, nf, type_info=_ti(n), shape_info=_si(n))

    def visit_if_expression(self, n: IfExpressionIR) -> ExpressionIR:
        loc = n.location or self._loc
        return IfExpressionIR(
            condition=n.condition.accept(self),
            then_expr=n.then_expr.accept(self),
            location=loc,
            else_expr=n.else_expr.accept(self) if n.else_expr is not None else None,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_lambda(self, n: LambdaIR) -> ExpressionIR:
        return n

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        return DifferentialIR(
            n.operand.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_range(self, n: RangeIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_comprehension(self, n: ArrayComprehensionIR) -> ExpressionIR:
        return n

    def visit_module(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_literal(self, n: ArrayLiteralIR) -> ExpressionIR:
        return n

    def visit_tuple_expression(self, n: TupleExpressionIR) -> ExpressionIR:
        return n

    def visit_tuple_access(self, n: TupleAccessIR) -> ExpressionIR:
        return n

    def visit_interpolated_string(self, n: InterpolatedStringIR) -> ExpressionIR:
        return n

    def visit_cast_expression(self, n: CastExpressionIR) -> ExpressionIR:
        return CastExpressionIR(
            n.expr.accept(self),
            n.target_type,
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_member_access(self, n: MemberAccessIR) -> ExpressionIR:
        return n

    def visit_try_expression(self, n: TryExpressionIR) -> ExpressionIR:
        return n

    def visit_match_expression(self, n: MatchExpressionIR) -> ExpressionIR:
        return n

    def visit_reduction_expression(self, n: ReductionExpressionIR) -> ExpressionIR:
        return ReductionExpressionIR(
            n.operation,
            n.loop_vars,
            n.body.accept(self),
            n.location or self._loc,
            where_clause=self._rw_wc(n.where_clause),
            loop_var_ranges=self._rw_lr(n.loop_var_ranges),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_builtin_call(self, n: BuiltinCallIR) -> ExpressionIR:
        return BuiltinCallIR(
            n.builtin_name,
            [a.accept(self) for a in (n.args or [])],
            n.location or self._loc,
            defid=getattr(n, "defid", None),
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_where_expression(self, n: WhereExpressionIR) -> ExpressionIR:
        return n

    def visit_pipeline_expression(self, n: PipelineExpressionIR) -> ExpressionIR:
        return n

    def visit_literal_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_identifier_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_wildcard_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_tuple_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_array_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_rest_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_guard_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_or_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_constructor_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binding_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_range_pattern(self, n: Any) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_function_value(self, n: FunctionValueIR) -> ExpressionIR:
        return n

    def visit_einstein(self, n: EinsteinIR) -> ExpressionIR:
        nc: List[EinsteinClauseIR] = []
        for c in n.clauses or []:
            nc.append(
                EinsteinClauseIR(
                    indices=[i.accept(self) for i in (c.indices or [])],
                    value=c.value.accept(self) if c.value is not None else None,
                    location=c.location,
                    where_clause=self._rw_wc(c.where_clause),
                    variable_ranges=dict(self._rw_lr(c.variable_ranges) or {}),
                )
            )
        return EinsteinIR(
            clauses=nc,
            shape=n.shape,
            element_type=n.element_type,
            location=n.location,
            type_info=_ti(n),
            shape_info=_si(n),
        )

    def visit_einstein_clause(self, n: EinsteinClauseIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_binding(self, n: BindingIR) -> ExpressionIR:
        if n.expr is None:
            return cast(ExpressionIR, n)
        return n.expr.accept(self)

    def visit_program(self, n: ProgramIR) -> ExpressionIR:
        return cast(ExpressionIR, n)

    def visit_select_at_argmax(self, n: SelectAtArgmaxIR) -> ExpressionIR:
        loc = n.location or self._loc
        pb = n.primal_body.accept(self) if n.primal_body is not None else None
        db = n.diff_body.accept(self) if n.diff_body is not None else None
        return SelectAtArgmaxIR(
            pb,
            db,
            n.loop_vars,
            loop_var_ranges=self._rw_lr(n.loop_var_ranges),
            location=loc,
            type_info=_ti(n),
            shape_info=_si(n),
            use_argmin=n.use_argmin,
        )

    def visit_lowered_reduction(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_select_at_argmax(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_comprehension(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_einstein_clause(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_einstein(self, n: Any) -> ExpressionIR:
        return n

    def visit_lowered_recurrence(self, n: Any) -> ExpressionIR:
        return n


class _SubstVisitor(_Rewriter):
    def __init__(self, m: Dict[DefId, ExpressionIR], loc: SourceLocation) -> None:
        super().__init__(loc)
        self._m = m

    def visit_identifier(self, n: IdentifierIR) -> ExpressionIR:
        if n.defid is not None and n.defid in self._m:
            return self._m[n.defid]
        return n


class _SubstDiffsVisitor(_SubstVisitor):
    def __init__(
        self,
        pm: Dict[DefId, ExpressionIR],
        dm: Dict[DefId, ExpressionIR],
        loc: SourceLocation,
    ) -> None:
        super().__init__(pm, loc)
        self._dm = dm

    def visit_differential(self, n: DifferentialIR) -> ExpressionIR:
        op = n.operand
        if isinstance(op, IdentifierIR) and op.defid is not None and op.defid in self._dm:
            return self._dm[op.defid]
        return DifferentialIR(
            op.accept(self),
            n.location or self._loc,
            type_info=_ti(n),
            shape_info=_si(n),
        )


def _sub(expr: ExpressionIR, m: Dict[DefId, ExpressionIR], loc: SourceLocation) -> ExpressionIR:
    """Replace IdentifierIR whose defid ∈ m."""
    if expr is None:
        return expr  # type: ignore[return-value]
    return expr.accept(_SubstVisitor(m, loc))


_substitute = _sub


def _sub_wd(
    expr: ExpressionIR,
    pm: Dict[DefId, ExpressionIR],
    dm: Dict[DefId, ExpressionIR],
    loc: SourceLocation,
) -> ExpressionIR:
    """Substitute identifiers AND DifferentialIR(@param) nodes."""
    if expr is None:
        return expr  # type: ignore[return-value]
    return expr.accept(_SubstDiffsVisitor(pm, dm, loc))


_substitute_with_diffs = _sub_wd
'''
