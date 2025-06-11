import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the axolotl_dev directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'axolotl_dev'))

from rewards import language_scorer, language_reward, REWARD_MAGNITUDES


class TestLanguageScorer:
    """Test suite for the language_scorer function from rewards.py."""

    def test_english_content_detection(self):
        """Test detection of English content."""
        output = """
        <think>
        I need to solve this problem step by step.
        First, I'll analyze the requirements.
        </think>
        The solution is here.
        """
        
        with patch('rewards.detect_lang') as mock_detect:
            mock_detect.return_value = "en"
            result = language_scorer(output)
        
        assert result["output_lang"] == "en"
        assert result["all_english"] == True

    def test_triton_code_content(self):
        """Test handling of Triton code content."""
        output = """
        Here is the solution:
        <triton>
        @triton.jit
        def kernel(x_ptr, y_ptr):
            # Implementation
            pass
        </triton>
        """
        
        with patch('rewards.detect_lang') as mock_detect:
            mock_detect.return_value = "en"
            result = language_scorer(output)
        
        assert result["output_lang"] == "en"
        assert result["all_english"] == True

    def test_mixed_content_with_tags(self):
        """Test content with various tags and English text."""
        output = """
        <think>Some thinking</think>
        This is the main answer content.
        <triton>Some code</triton>
        And more answer content here.
        """
        
        with patch('rewards.detect_lang') as mock_detect:
            mock_detect.return_value = "en"
            result = language_scorer(output)
        
        assert result["output_lang"] == "en"
        assert result["all_english"] == True

    def test_non_english_detection(self):
        """Test detection of non-English content."""
        output = """
        Some content that should be detected as non-English
        """
        
        with patch('rewards.quick_check') as mock_quick:
            with patch('rewards.detect_lang') as mock_detect:
                # Make quick_check return False so detect_lang gets called
                mock_quick.return_value = False
                mock_detect.return_value = "fr"  # French
                result = language_scorer(output)
        
        assert result["output_lang"] == "fr"
        assert result["all_english"] == False
        assert result["non_english_count"] == 1
        assert result["total_parts"] == 1

    def test_empty_content_handled(self):
        """Test that empty content is handled correctly."""
        output = ""
        
        result = language_scorer(output)
        
        # Empty content should be considered English
        assert result["output_lang"] == "en"
        assert result["all_english"] == True
        assert result["total_parts"] == 0

    def test_plain_text_content(self):
        """Test with plain text content."""
        output = "Just plain text without any special blocks."
        
        with patch('rewards.detect_lang') as mock_detect:
            mock_detect.return_value = "en"
            result = language_scorer(output)
        
        assert result["output_lang"] == "en"
        assert result["all_english"] == True

    def test_quick_check_performance_optimization(self):
        """Test that quick_check is used for performance."""
        output = "English content with some 中文 characters"
        
        with patch('rewards.quick_check') as mock_quick:
            with patch('rewards.detect_lang') as mock_detect:
                # quick_check returns False for non-Latin content
                mock_quick.return_value = False
                mock_detect.return_value = "zh"  # Chinese
                
                result = language_scorer(output)
        
        # Should have called quick_check once for the entire output
        assert mock_quick.call_count == 1
        # Should have called detect_lang because quick_check returned False
        assert mock_detect.call_count == 1
        assert result["output_lang"] == "zh"

    def test_raw_content_detection(self):
        """Test that language detection works on raw content without preprocessing."""
        output = "Think with $\\LaTeX$ math and ```code blocks```"
        
        with patch('rewards.quick_check') as mock_quick:
            with patch('rewards.detect_lang') as mock_detect:
                # Make quick_check return False so detect_lang gets called
                mock_quick.return_value = False
                mock_detect.return_value = "en"
                
                result = language_scorer(output)
        
        # detect_lang should be called once with the raw output
        assert mock_detect.call_count == 1
        mock_detect.assert_called_with(output)
        assert result["output_lang"] == "en"


class TestLanguageRewardFunction:
    """Test suite for the language_reward function from rewards.py."""

    def test_all_english_gets_bonus(self):
        """Test that responses with all English content get bonus."""
        completions = [[{"content": "English response with <think>English</think> and <triton>English code</triton>"}]]
        
        with patch('rewards.language_scorer') as mock_scorer:
            mock_scorer.return_value = {
                "all_english": True,
                "non_english_count": 0,
                "total_parts": 3
            }
            
            rewards = language_reward(completions)
        
        assert len(rewards) == 1
        assert rewards[0] == REWARD_MAGNITUDES["language_bonus"]

    def test_non_english_full_penalty(self):
        """Test full penalty for non-English content."""
        completions = [[{"content": "Non-English response"}]]
        
        with patch('rewards.language_scorer') as mock_scorer:
            mock_scorer.return_value = {
                "all_english": False,
                "output_lang": "fr"
            }
            
            rewards = language_reward(completions)
        
        assert len(rewards) == 1
        assert rewards[0] == REWARD_MAGNITUDES["language_penalty"]

    def test_all_non_english_full_penalty(self):
        """Test full penalty when content is non-English."""
        completions = [[{"content": "Non-English response"}]]
        
        with patch('rewards.language_scorer') as mock_scorer:
            mock_scorer.return_value = {
                "all_english": False,
                "output_lang": "fr"
            }
            
            rewards = language_reward(completions)
        
        assert len(rewards) == 1
        assert rewards[0] == REWARD_MAGNITUDES["language_penalty"]

    def test_empty_content_gets_bonus(self):
        """Test that empty content gets English bonus."""
        completions = [[{"content": ""}]]
        
        with patch('rewards.language_scorer') as mock_scorer:
            mock_scorer.return_value = {
                "all_english": True,
                "output_lang": "en",
                "total_parts": 0  # No content
            }
            
            rewards = language_reward(completions)
        
        assert len(rewards) == 1
        assert rewards[0] == REWARD_MAGNITUDES["language_bonus"]

    def test_multiple_completions(self):
        """Test processing multiple completions."""
        completions = [
            [{"content": "First English response"}],
            [{"content": "Second non-English response"}],
            [{"content": "Third English response"}]
        ]
        
        with patch('rewards.language_scorer') as mock_scorer:
            # Mock different scores for each completion
            mock_scorer.side_effect = [
                {"all_english": True, "output_lang": "en"},
                {"all_english": False, "output_lang": "fr"},
                {"all_english": True, "output_lang": "en"}
            ]
            
            rewards = language_reward(completions)
        
        assert len(rewards) == 3
        assert rewards[0] == REWARD_MAGNITUDES["language_bonus"]  # All English
        assert rewards[1] == REWARD_MAGNITUDES["language_penalty"]  # Non-English
        assert rewards[2] == REWARD_MAGNITUDES["language_bonus"]  # All English

    def test_wandb_attributes_integration(self):
        """Test that wandb attributes are properly used."""
        completions = [[{"content": "Test response"}]]
        
        with patch('rewards.language_scorer') as mock_scorer:
            with patch('rewards.wandb_attributes') as mock_wandb_attrs:
                mock_scorer.return_value = {
                    "all_english": True,
                    "non_english_count": 0,
                    "total_parts": 1
                }
                mock_context = MagicMock()
                mock_wandb_attrs.return_value = mock_context
                
                rewards = language_reward(completions)
        
        # Should have used wandb_attributes as context manager
        mock_wandb_attrs.assert_called_once()
        mock_context.__enter__.assert_called_once()
        mock_context.__exit__.assert_called_once()


class TestRewardMagnitudes:
    """Test that reward magnitudes are properly configured."""

    def test_language_magnitudes_exist(self):
        """Test that language reward magnitudes are defined."""
        assert "language_bonus" in REWARD_MAGNITUDES
        assert "language_penalty" in REWARD_MAGNITUDES
        
        assert isinstance(REWARD_MAGNITUDES["language_bonus"], (int, float))
        assert isinstance(REWARD_MAGNITUDES["language_penalty"], (int, float))
        
        # Bonus should be positive, penalty should be negative
        assert REWARD_MAGNITUDES["language_bonus"] > 0
        assert REWARD_MAGNITUDES["language_penalty"] < 0

    def test_magnitude_values_reasonable(self):
        """Test that magnitude values are reasonable."""
        bonus = REWARD_MAGNITUDES["language_bonus"]
        penalty = REWARD_MAGNITUDES["language_penalty"]
        
        # Should be reasonable values (not too extreme)
        assert 0 < bonus < 1
        assert -1 < penalty < 0


class TestLanguageRewardIntegration:
    """Integration tests for the complete language reward pipeline."""

    def test_realistic_triton_response_english(self):
        """Test with a realistic English Triton response."""
        completions = [[{"content": """
        <think>
        I need to implement a matrix multiplication kernel using Triton.
        I'll use block-based computation for efficiency.
        </think>
        
        Here's the implementation:
        
        <triton>
        import triton
        import triton.language as tl
        
        @triton.jit
        def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K):
            # Get program IDs
            pid_m = tl.program_id(0)
            pid_n = tl.program_id(1)
            
            # Compute offsets
            offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
            offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
            
            # Load and compute
            a = tl.load(a_ptr + offs_am)
            b = tl.load(b_ptr + offs_bn)
            c = tl.dot(a, b)
            
            # Store result
            tl.store(c_ptr + offs_am, c)
        </triton>
        """}]]
        
        # Should get language bonus for all-English content
        rewards = language_reward(completions)
        assert len(rewards) == 1
        assert rewards[0] == REWARD_MAGNITUDES["language_bonus"]

    def test_realistic_mixed_language_response(self):
        """Test with a response containing mixed languages."""
        completions = [[{"content": """
        <think>
        Je dois implémenter une fonction de multiplication.
        </think>
        
        Here's the English explanation and code:
        
        <triton>
        # English comments in code
        @triton.jit
        def kernel():
            pass
        </triton>
        """}]]
        
        # Mock to detect the overall content as non-English due to French content
        with patch('rewards.language_scorer') as mock_scorer:
            mock_scorer.return_value = {
                "all_english": False,
                "output_lang": "fr"  # Detected as French overall
            }
            
            rewards = language_reward(completions)
            assert len(rewards) == 1
            assert rewards[0] == REWARD_MAGNITUDES["language_penalty"]  # Full penalty 