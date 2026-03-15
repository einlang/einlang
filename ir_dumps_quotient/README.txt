IR S-expr dumps per pass (EINLANG_DUMP_IR_PER_PASS=1).
00 = after ASTToIRLoweringPass, 01 = after RestPatternPreprocessingPass, etc.
Reductions include :loop_var_ranges only when non-empty.
Compare dumps to see which pass stops setting loop_var_ranges on specialized functions.
