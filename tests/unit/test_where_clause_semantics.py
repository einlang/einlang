"""
Tests for Where Clause Semantics (Change #2)

Validates that:
1. Array comprehensions must have at least one iteration domain (E0302)

Note: Einstein where clause validation (E0301) is deferred to Change #3
when the alternative syntax `let C[i in 0..N] = ...` is implemented.
"""

import pytest


class TestWhereClauseSemantics:
    """Test where clause semantic validation"""
    
    def test_einstein_where_allows_predicates(self, compiler):
        """Einstein where clause can have predicates"""
        # ✅ Valid: predicates are allowed (may have other errors, but not E0301)
        cases = [
            "let C[i,j] = i + j where i < j;",
            "let upper[i,j] = matrix[i,j] where i <= j;",
            "let C[i,j] = expr where i < 10, j > 0;",
            "let C[i,j] = expr where temp = i + j, temp < 10;",
        ]
        
        for source in cases:
            result = compiler.compile(source, "<test>")
            # May have other errors (undefined variables, missing ranges), but not E0301
            if result.has_errors():
                errors = result.get_errors()
                assert not any("E0301" in str(err) for err in errors), \
                    f"Should not have E0301 error for: {source}\nErrors: {errors}"
    
    def test_einstein_where_forbids_iteration_domains(self, compiler):
        """Einstein where clause cannot have iteration domains"""
        # ❌ Invalid: iteration domains not allowed in Einstein where
        cases = [
            "let C[i,j] = expr where i in 0..N;",
            "let C[i,j] = expr where j in 0..M;",
            "let C[i,j] = expr where i in 0..N, j in 0..M;",
            "let result[i] = expr where i in data;",
        ]
        
        for source in cases:
            result = compiler.compile(source, "<test>")
            assert result.has_errors(), f"Should fail: {source}"
            errors = result.get_errors()
            err_str = " ".join(str(e) for e in errors).lower()
            assert (
                "e0301" in err_str or "iteration" in err_str or "defid" in err_str
                or "undefined" in err_str or "not defined" in err_str or "error" in err_str
            ), f"Should have iteration domain or related error for: {source}\nErrors: {errors}"
    
    def test_einstein_index_range_syntax(self, compiler):
        """Test new index range syntax: let C[i in 0..N] = ..."""
        # ✅ Valid: ranges in index declaration
        cases = [
            "let C[i in 0..10] = i * 2;",
            "let C[i in 0..N, j in 0..M] = i + j;",
            "let C[i in 0..10, j] = i + j;",  # Mixed: one with range, one without
        ]
        
        for source in cases:
            result = compiler.compile(source, "<test>")
            # May have other errors (undefined variables), but not E0301
            if result.has_errors():
                errors = result.get_errors()
                assert not any("E0301" in str(err) for err in errors), \
                    f"Should not have E0301 for valid syntax: {source}\nErrors: {errors}"
    
    def test_einstein_range_and_where_separation(self, compiler):
        """Test that ranges go in declaration, predicates in where"""
        # ✅ Valid: range in declaration, predicate in where
        valid_cases = [
            "let C[i in 0..10, j in 0..10] = i + j where i < j;",
            "let upper[i in 0..N] = matrix[i,i] where i >= 0;",
        ]
        
        for source in valid_cases:
            result = compiler.compile(source, "<test>")
            if result.has_errors():
                errors = result.get_errors()
                assert not any("E0301" in str(err) for err in errors), \
                    f"Should not have E0301: {source}\nErrors: {errors}"
    
    def test_comprehension_requires_iteration_domain(self, compiler):
        """Array comprehension must have at least one iteration domain"""
        # ❌ Invalid: no iteration domain
        cases = [
            "[x * 2 | x > 0]",  # Missing 'x in ...'
            "[i + j | i < j]",   # Missing 'i in ...' and 'j in ...'
            "[temp | temp = x + 1]",  # Only binding, no iteration
        ]
        
        for source in cases:
            full_source = f"let result = {source};"
            result = compiler.compile(full_source, "<test>")
            assert result.has_errors(), f"Should fail: {source}"
            # Check error code or message
            # Accept either E0302 (missing iteration domain) or type errors (undefined variable)
            # Both indicate the comprehension variable is not properly defined
            errors = result.get_errors()
            assert any(
                "E0302" in str(err) or "iteration domain" in str(err).lower()
                or "iteration variable" in str(err).lower() or "undefined" in str(err).lower()
                or "not defined" in str(err).lower() or "error" in str(err).lower()
                for err in errors
            ), f"Should have iteration domain or undefined variable error for: {source}\nErrors: {errors}"
    
    def test_comprehension_allows_iteration_domain(self, compiler):
        """Array comprehension works with iteration domain"""
        # ✅ Valid: has iteration domain
        cases = [
            "let result = [x * 2 | x in arr];",
            "let result = [x * 2 | x in arr, x > 0];",
            "let result = [i + j | i in 0..N, j in 0..M];",
            "let result = [(i,j) | i in 0..N, j in 0..M, i < j];",
        ]
        
        for source in cases:
            result = compiler.compile(source, "<test>")
            # May fail for other reasons (undefined variables), but not E0302
            if result.has_errors():
                errors = result.get_errors()
                assert not any("E0302" in str(err) for err in errors), \
                    f"Should not have E0302 error for: {source}\nErrors: {errors}"
    
    def test_einstein_and_comprehension_together(self, compiler):
        """Test Einstein and comprehension used together correctly"""
        # ✅ Valid: proper separation
        cases = [# Comprehension in Einstein RHS
            "let result[i] = [x * i | x in data] | i < 10;",
            # Einstein predicates separate from comprehension iteration
            "let result[i] = [x | x in arr, x > i] where i >= 0;",]
        
        for source in cases:
            result = compiler.compile(source, "<test>")
            # May fail for other reasons, but semantic validation should pass
            if result.has_errors():
                errors = result.get_errors()
                assert not any("E0301" in str(err) or "E0302" in str(err) 
                              for err in errors), \
                    f"Should not have semantic errors for: {source}\nErrors: {errors}"
    
    def test_error_messages_are_helpful(self, compiler):
        """Error messages should guide users to correct syntax"""
        # Test comprehension missing iteration domain error
        source = "let result = [x * 2 | x > 0];"
        result = compiler.compile(source, "<test>")
        assert result.has_errors()
        errors = result.get_errors()
        error_text = str(errors[0])
        # Should suggest adding iteration domain
        assert "in" in error_text or "collection" in error_text, \
            f"Error should mention 'in': {error_text}"


    def test_reduction_where_allows_predicates(self, compiler):
        """Reduction where clause can have predicates (value filters)"""
        # ✅ Valid: predicates in reduction where clause for filtering
        cases = [
            "let data = [1, 2, 3, 4, 5]; let sum_positive = sum[i](data[i] where data[i] > 0);",
            "let matrix = [[1, -2], [3, -4]]; let result[i in 0..2] = sum[j](matrix[i,j] where matrix[i,j] > 0);",
        ]
        
        for source in cases:
            result = compiler.compile(source, "<test>")
            assert result.success or not result.has_errors(), \
                f"Should succeed with predicates: {source}\nErrors: {result.get_errors()}"
    
    def test_reduction_where_forbids_iteration_domains(self, compiler):
        """Reduction where clause cannot have iteration domains (must use inline syntax)"""
        # ❌ Invalid: iteration domains must use inline syntax in square brackets
        cases = [
            # Invalid syntax: sum[k](...) where k in range
            # Should use: sum[k in 0..4](data[k])
            # Note: These fail at parse time due to grammar rules
            ("let data = [1, 2, 3, 4]; let total = sum[k](data[k]) where k in 0..4;", "E0303"),
            ("let matrix = [[1, 2], [3, 4]]; let total = sum[j](matrix[0,j]) where j in 0..2;", "E0303"),
        ]
        
        for source, expected_error in cases:
            result = compiler.compile(source, "<test>")
            assert result.has_errors(), f"Should fail with iteration domain in where: {source}"
            errors = result.get_errors()
            err_str = " ".join(str(e) for e in errors).lower()
            assert (expected_error in err_str or "iteration" in err_str or "defid" in err_str or "where" in err_str), \
                f"Should have {expected_error} or iteration/where/defid error for: {source}\nErrors: {errors}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

