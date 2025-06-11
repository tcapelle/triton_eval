import pytest
import os
import tempfile
from unittest.mock import patch, MagicMock
from triton_eval.language_checks import (
    detect_lang, strip_noise, quick_check, language_reward, 
    _get_model, _FASTTEXT_MODEL
)


class TestStripNoise:
    """Test suite for the strip_noise function."""

    def test_strip_latex(self):
        """Test removal of LaTeX expressions."""
        text = "This is $x^2 + y^2 = z^2$ and also \\[E = mc^2\\]"
        result = strip_noise(text)
        assert result == "This is and also"

    def test_strip_code_blocks(self):
        """Test removal of fenced code blocks."""
        text = "Here is some code ```python\nprint('hello')\n``` and more text"
        result = strip_noise(text)
        assert result == "Here is some code and more text"

    def test_strip_begin_end_blocks(self):
        """Test removal of LaTeX begin/end blocks."""
        text = "Text \\begin{equation} x = y \\end{equation} more text"
        result = strip_noise(text)
        assert result == "Text more text"

    def test_normalize_whitespace(self):
        """Test normalization of multiple whitespaces."""
        text = "Text   with    multiple\n\n\nspaces"
        result = strip_noise(text)
        assert result == "Text with multiple spaces"

    def test_empty_string(self):
        """Test with empty string."""
        result = strip_noise("")
        assert result == ""

    def test_complex_mixed_noise(self):
        """Test with complex mixed LaTeX and code."""
        text = """
        Here is some text $\\alpha + \\beta$ and code:
        ```python
        def func():
            return x + y
        ```
        And more \\begin{align} x &= y \\\\ z &= w \\end{align} text.
        """
        result = strip_noise(text)
        assert "alpha" not in result
        assert "def func" not in result
        assert "python" not in result
        assert "align" not in result
        assert "Here is some text" in result
        assert "And more" in result
        assert "text." in result


class TestQuickCheck:
    """Test suite for the quick_check function."""

    def test_latin_text_passes(self):
        """Test that Latin text passes quick check."""
        assert quick_check("Hello world, this is English text!") == True
        assert quick_check("Bonjour, comment allez-vous?") == True
        assert quick_check("Hola, ¿cómo estás?") == True

    def test_cyrillic_text_fails(self):
        """Test that Cyrillic text fails quick check."""
        assert quick_check("Привет мир") == False
        assert quick_check("Hello Привет") == False

    def test_cjk_text_fails(self):
        """Test that CJK text fails quick check."""
        assert quick_check("你好世界") == False
        assert quick_check("こんにちは") == False
        assert quick_check("Hello 世界") == False

    def test_empty_string(self):
        """Test with empty string."""
        assert quick_check("") == True

    def test_numbers_and_symbols(self):
        """Test that numbers and symbols pass."""
        assert quick_check("123 + 456 = 579") == True
        assert quick_check("print('Hello') # Comment") == True


class TestDetectLang:
    """Test suite for the detect_lang function."""

    def test_english_detection(self):
        """Test detection of English text."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            result = detect_lang("This is a test in English language")
            assert result == "en"

    def test_short_text(self):
        """Test with very short text."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.95])
            mock_get_model.return_value = mock_model
            
            result = detect_lang("Hi")
            assert isinstance(result, str)
            assert len(result) == 2  # ISO language code

    def test_empty_text(self):
        """Test with empty text."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.5])
            mock_get_model.return_value = mock_model
            
            result = detect_lang("")
            assert isinstance(result, str)

    def test_code_like_text(self):
        """Test with code-like text."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.8])
            mock_get_model.return_value = mock_model
            
            result = detect_lang("def func(): return x + y")
            assert isinstance(result, str)

    def test_text_with_newlines(self):
        """Test that text with newlines is handled correctly.
        
        This test would fail with the original code because FastText
        raises ValueError: predict processes one line at a time (remove '\\n')
        """
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            # Test with various newline scenarios
            multiline_text = """This is a multi-line text
            that spans several lines
            and should be handled correctly"""
            
            result = detect_lang(multiline_text)
            assert result == "en"
            
            # Verify that the model.predict was called with single-line text
            call_args = mock_model.predict.call_args[0][0]  # First positional argument
            assert '\n' not in call_args, "Text passed to FastText should not contain newlines"
            assert '\r' not in call_args, "Text passed to FastText should not contain carriage returns"
            
    def test_text_with_only_newlines(self):
        """Test handling of text that becomes empty after newline removal."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            
            # Test with text that's only newlines/whitespace
            result = detect_lang("\n\n\r\n  \t  \n")
            
            # Should return 'unknown' for empty text and not call the model
            assert result == "unknown"
            mock_model.predict.assert_not_called()

    def test_mixed_language_text(self):
        """Test with mixed language text."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.7])
            mock_get_model.return_value = mock_model
            
            result = detect_lang("Hello world and bonjour monde")
            assert isinstance(result, str)


class TestModelLoading:
    """Test suite for the model loading functionality."""

    def test_model_loads_once(self):
        """Test that the model is loaded only once."""
        # Reset the global model
        import triton_eval.language_checks as lang_module
        lang_module._FASTTEXT_MODEL = None
        
        # Mock fasttext.load_model to track calls
        with patch('triton_eval.language_checks.fasttext.load_model') as mock_load:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_load.return_value = mock_model
            
            # Call _get_model multiple times
            model1 = _get_model()
            model2 = _get_model()
            model3 = _get_model()
            
            # Should be the same instance
            assert model1 is model2
            assert model2 is model3
            
            # fasttext.load_model should only be called once
            assert mock_load.call_count == 1

    def test_model_file_not_found_error(self):
        """Test error handling when model file is not found."""
        # Reset the global model
        import triton_eval.language_checks as lang_module
        lang_module._FASTTEXT_MODEL = None
        
        with patch('triton_eval.language_checks.os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            with pytest.raises(FileNotFoundError) as exc_info:
                _get_model()
            
            assert "Could not find lid.176.bin" in str(exc_info.value)
            assert "curl -L -o lid.176.bin" in str(exc_info.value)


class TestLanguageReward:
    """Test suite for the language_reward function."""

    def test_all_english_gets_bonus(self):
        """Test that all-English content gets a bonus."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            problem = "Solve this equation"
            thoughts = "I need to think about this problem"
            answer = "The solution is x equals 5"
            
            reward = language_reward(problem, thoughts, answer)
            assert reward == 0.1  # Default bonus

    def test_custom_bonus_penalty(self):
        """Test custom bonus and penalty values."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            problem = "Solve this"
            thoughts = "Think about it"
            answer = "Solution is 5"
            
            reward = language_reward(problem, thoughts, answer, bonus=0.2, penalty=-0.5)
            assert reward == 0.2

    def test_empty_parts_treated_as_english(self):
        """Test that empty parts are treated as English."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            reward = language_reward("English text", "", "More English")
            assert reward == 0.1

    def test_mixed_languages_proportional_penalty(self):
        """Test proportional penalty for mixed languages."""
        with patch('triton_eval.language_checks.detect_lang') as mock_detect:
            # Mock to return different languages
            mock_detect.side_effect = ["en", "fr", "en"]  # 1 out of 3 non-English
            
            reward = language_reward("English", "French", "English")
            expected = -0.2 * (1/3)  # penalty * (non_english_count / total_parts)
            assert abs(reward - expected) < 0.001

    def test_all_non_english_full_penalty(self):
        """Test full penalty when all parts are non-English."""
        with patch('triton_eval.language_checks.detect_lang') as mock_detect:
            mock_detect.return_value = "fr"  # All French
            
            reward = language_reward("French", "French", "French")
            assert abs(reward - (-0.2)) < 0.001  # Full penalty (handle floating point precision)

    def test_custom_target_language(self):
        """Test with custom target language."""
        with patch('triton_eval.language_checks.detect_lang') as mock_detect:
            mock_detect.return_value = "fr"  # All French
            
            reward = language_reward("French", "French", "French", target="fr")
            assert reward == 0.1  # Should get bonus for French

    def test_noise_stripping_integration(self):
        """Test that noise is properly stripped before language detection."""
        with patch('triton_eval.language_checks.detect_lang') as mock_detect:
            mock_detect.return_value = "en"
            
            problem = "Solve $x^2 + y^2 = z^2$"
            thoughts = "```python\nprint('hello')\n``` Think about it"
            answer = "The answer is 5"
            
            reward = language_reward(problem, thoughts, answer)
            
            # Check that detect_lang was called with stripped text
            calls = mock_detect.call_args_list
            assert len(calls) == 3
            
            # First call should have LaTeX stripped
            assert "$" not in calls[0][0][0]
            # Second call should have code stripped  
            assert "print" not in calls[1][0][0]


class TestLanguageRewardIntegration:
    """Integration tests for language reward with realistic scenarios."""

    def test_typical_triton_response(self):
        """Test with a typical Triton kernel response format."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            response_parts = [
                "Write a CUDA kernel for matrix multiplication",  # problem
                "I need to consider block sizes and memory access patterns",  # thoughts
                """
                @triton.jit
                def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak):
                    # Implementation here
                    pass
                """  # answer
            ]
            
            # This should get bonus since it's all English
            reward = language_reward(*response_parts)
            assert reward == 0.1

    def test_mixed_code_and_comments(self):
        """Test with code containing comments in different languages."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            problem = "Implement a function"
            thoughts = "I will write the code step by step"
            answer = """
            def func():
                # This is an English comment
                return x + y
            """
            
            reward = language_reward(problem, thoughts, answer)
            assert reward == 0.1

    def test_mathematical_expressions(self):
        """Test with mathematical expressions."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            problem = "Solve the quadratic equation"
            thoughts = "Using the formula $x = \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$"
            answer = "The solutions are x = 2 and x = -1"
            
            reward = language_reward(problem, thoughts, answer)
            assert reward == 0.1  # LaTeX should be stripped, English detected

    def test_empty_and_none_handling(self):
        """Test handling of None and empty strings."""
        with patch('triton_eval.language_checks._get_model') as mock_get_model:
            mock_model = MagicMock()
            mock_model.predict.return_value = (['__label__en'], [0.99])
            mock_get_model.return_value = mock_model
            
            # Should handle None gracefully
            reward = language_reward("English text", None, "More English")
            assert reward == 0.1
            
            # Should handle empty strings
            reward = language_reward("", "", "English text")
            assert reward == 0.1 