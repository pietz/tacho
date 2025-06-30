from unittest.mock import MagicMock

import pytest

from tacho.ai import llm, ping_model, bench_model, BENCHMARK_PROMPT, VALIDATION_PROMPT


@pytest.mark.unit
class TestAI:
    @pytest.mark.asyncio
    async def test_llm_basic_call(self, mock_litellm):
        """Test basic LLM call functionality"""
        result = await llm("gpt-4", "Test prompt", 100)
        
        # Verify litellm was called correctly
        mock_litellm.assert_called_once_with(
            "gpt-4",
            [{"role": "user", "content": "Test prompt"}],
            max_tokens=100
        )
        
        # Verify response is returned
        assert result == mock_litellm.return_value
    
    @pytest.mark.asyncio
    async def test_llm_without_max_tokens(self, mock_litellm):
        """Test LLM call without specifying max tokens"""
        await llm("gpt-4", "Test prompt")
        
        mock_litellm.assert_called_once_with(
            "gpt-4",
            [{"role": "user", "content": "Test prompt"}],
            max_tokens=None
        )
    
    @pytest.mark.asyncio
    async def test_ping_model_success(self, mock_litellm):
        """Test successful model ping"""
        # Create a mock console that can be used in the context
        mock_console_instance = MagicMock()
        
        result = await ping_model("gpt-4", mock_console_instance)
        
        # Verify success
        assert result is True
        
        # Verify console output
        mock_console_instance.print.assert_called_once_with("[green]✓[/green] gpt-4")
        
        # Verify LLM was called with validation prompt
        mock_litellm.assert_called_once_with(
            "gpt-4",
            [{"role": "user", "content": VALIDATION_PROMPT}],
            max_tokens=1
        )
    
    @pytest.mark.asyncio
    async def test_ping_model_failure(self, mock_litellm):
        """Test failed model ping"""
        # Configure mock to raise exception
        mock_litellm.side_effect = Exception("API Error")
        mock_console_instance = MagicMock()
        
        result = await ping_model("invalid-model", mock_console_instance)
        
        # Verify failure
        assert result is False
        
        # Verify error output
        mock_console_instance.print.assert_called_once_with(
            "[red]✗[/red] invalid-model - API Error"
        )
    
    @pytest.mark.asyncio
    async def test_bench_model_success(self, mock_litellm, mocker):
        """Test successful benchmark run"""
        # Mock time to control duration measurement
        mock_time = mocker.patch('tacho.ai.time.time')
        mock_time.side_effect = [100.0, 102.5]  # 2.5 second duration
        
        # Configure mock response with usage data (no reasoning tokens)
        mock_response = MagicMock()
        mock_usage = MagicMock()
        mock_usage.completion_tokens = 150
        # Explicitly configure to not have completion_tokens_details
        mock_usage.completion_tokens_details = None
        mock_response.usage = mock_usage
        mock_litellm.return_value = mock_response
        
        duration, tokens = await bench_model("gpt-4", 500)
        
        # Verify results
        assert duration == 2.5
        assert tokens == 150
        
        # Verify LLM was called correctly
        mock_litellm.assert_called_once_with(
            "gpt-4",
            [{"role": "user", "content": BENCHMARK_PROMPT}],
            max_tokens=500
        )
    
    @pytest.mark.asyncio
    async def test_bench_model_no_usage_data(self, mock_litellm, mocker):
        """Test benchmark when response has no usage data"""
        # Mock time
        mock_time = mocker.patch('tacho.ai.time.time')
        mock_time.side_effect = [100.0, 101.0]
        
        # Configure mock response without usage
        mock_response = MagicMock()
        mock_response.usage = None
        mock_litellm.return_value = mock_response
        
        duration, tokens = await bench_model("gpt-4", 500)
        
        # Should return 0 tokens when no usage data
        assert duration == 1.0
        assert tokens == 0
    
    @pytest.mark.asyncio
    async def test_bench_model_exception_handling(self, mock_litellm):
        """Test that exceptions propagate from bench_model"""
        mock_litellm.side_effect = Exception("Network error")
        
        with pytest.raises(Exception, match="Network error"):
            await bench_model("gpt-4", 500)
    
    @pytest.mark.asyncio
    async def test_bench_model_with_reasoning_tokens(self, mock_litellm, mocker):
        """Test benchmark with reasoning models that have completion_tokens_details"""
        # Mock time
        mock_time = mocker.patch('tacho.ai.time.time')
        mock_time.side_effect = [100.0, 103.0]  # 3 second duration
        
        # Configure mock response with reasoning tokens
        mock_response = MagicMock()
        mock_response.usage.completion_tokens = 50  # Regular completion tokens
        
        # Mock completion_tokens_details with reasoning_tokens
        mock_details = MagicMock()
        mock_details.reasoning_tokens = 200  # Reasoning tokens
        mock_response.usage.completion_tokens_details = mock_details
        
        mock_litellm.return_value = mock_response
        
        duration, tokens = await bench_model("o1-mini", 500)
        
        # Verify results - should include both completion and reasoning tokens
        assert duration == 3.0
        assert tokens == 250  # 50 completion + 200 reasoning
        
        # Verify LLM was called correctly
        mock_litellm.assert_called_once_with(
            "o1-mini",
            [{"role": "user", "content": BENCHMARK_PROMPT}],
            max_tokens=500
        )
    
    @pytest.mark.asyncio
    async def test_bench_model_with_empty_completion_details(self, mock_litellm, mocker):
        """Test benchmark when completion_tokens_details exists but has no reasoning_tokens"""
        # Mock time
        mock_time = mocker.patch('tacho.ai.time.time')
        mock_time.side_effect = [100.0, 102.0]
        
        # Configure mock response with completion_tokens_details but no reasoning_tokens
        mock_response = MagicMock()
        mock_response.usage.completion_tokens = 100
        mock_response.usage.completion_tokens_details = MagicMock(spec=[])  # No reasoning_tokens attribute
        
        mock_litellm.return_value = mock_response
        
        duration, tokens = await bench_model("gpt-4", 500)
        
        # Should only count regular completion tokens
        assert duration == 2.0
        assert tokens == 100