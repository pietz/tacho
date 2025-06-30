from unittest.mock import MagicMock

import pytest
from litellm import (
    AuthenticationError,
    BadRequestError,
    NotFoundError,
    RateLimitError,
    APIConnectionError,
    ContextWindowExceededError,
    ContentPolicyViolationError,
)

from tacho.ai import llm, ping_model, bench_model, BENCHMARK_PROMPT, VALIDATION_PROMPT


@pytest.mark.unit
class TestAI:
    @pytest.mark.asyncio
    async def test_llm_basic_call(self, mock_litellm):
        """Test basic LLM call functionality"""
        result = await llm("gpt-4", "Test prompt", 100)

        # Verify litellm was called correctly
        mock_litellm.assert_called_once_with(
            "gpt-4", [{"role": "user", "content": "Test prompt"}], max_tokens=100
        )

        # Verify response is returned
        assert result == mock_litellm.return_value

    @pytest.mark.asyncio
    async def test_llm_without_max_tokens(self, mock_litellm):
        """Test LLM call without specifying max tokens"""
        await llm("gpt-4", "Test prompt")

        mock_litellm.assert_called_once_with(
            "gpt-4", [{"role": "user", "content": "Test prompt"}], max_tokens=None
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
            "gpt-4", [{"role": "user", "content": VALIDATION_PROMPT}], max_tokens=1
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

        # Verify error output - now includes "..." for generic exceptions
        mock_console_instance.print.assert_called_once_with(
            "[red]✗[/red] invalid-model - API Error..."
        )

    @pytest.mark.asyncio
    async def test_bench_model_success(self, mock_litellm, mocker):
        """Test successful benchmark run"""
        # Mock time to control duration measurement
        mock_time = mocker.patch("tacho.ai.time.time")
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
            "gpt-4", [{"role": "user", "content": BENCHMARK_PROMPT}], max_tokens=500
        )

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
        mock_time = mocker.patch("tacho.ai.time.time")
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
            "o1-mini", [{"role": "user", "content": BENCHMARK_PROMPT}], max_tokens=500
        )

    @pytest.mark.asyncio
    async def test_bench_model_with_empty_completion_details(
        self, mock_litellm, mocker
    ):
        """Test benchmark when completion_tokens_details exists but has no reasoning_tokens"""
        # Mock time
        mock_time = mocker.patch("tacho.ai.time.time")
        mock_time.side_effect = [100.0, 102.0]

        # Configure mock response with completion_tokens_details but no reasoning_tokens
        mock_response = MagicMock()
        mock_response.usage.completion_tokens = 100
        mock_response.usage.completion_tokens_details = MagicMock(
            spec=[]
        )  # No reasoning_tokens attribute

        mock_litellm.return_value = mock_response

        duration, tokens = await bench_model("gpt-4", 500)

        # Should only count regular completion tokens
        assert duration == 2.0
        assert tokens == 100

    @pytest.mark.asyncio
    async def test_ping_model_authentication_error(self, mock_litellm):
        """Test ping_model handling of authentication errors"""
        mock_litellm.side_effect = AuthenticationError(
            message="Invalid API key provided. You can find your API key at https://platform.openai.com/api-keys.",
            llm_provider="openai",
            model="gpt-4o-mini"
        )
        mock_console_instance = MagicMock()

        result = await ping_model("gpt-4o-mini", mock_console_instance)

        assert result is False
        mock_console_instance.print.assert_called_once()
        call_args = mock_console_instance.print.call_args[0][0]
        assert "[red]✗[/red] gpt-4o-mini" in call_args
        assert "Authentication Failed" in call_args
        assert "OPENAI_API_KEY" in call_args

    @pytest.mark.asyncio
    async def test_ping_model_not_found_error(self, mock_litellm):
        """Test ping_model handling of model not found errors"""
        mock_litellm.side_effect = NotFoundError(
            message="The model 'gpt-8' does not exist",
            llm_provider="openai",
            model="gpt-8"
        )
        mock_console_instance = MagicMock()

        result = await ping_model("gpt-8", mock_console_instance)

        assert result is False
        mock_console_instance.print.assert_called_once()
        call_args = mock_console_instance.print.call_args[0][0]
        assert "[red]✗[/red] gpt-8" in call_args
        assert "Model Not Found" in call_args

    @pytest.mark.asyncio
    async def test_ping_model_rate_limit_error(self, mock_litellm):
        """Test ping_model handling of rate limit errors"""
        mock_litellm.side_effect = RateLimitError(
            message="Rate limit exceeded. Please retry after 60 seconds",
            llm_provider="openai",
            model="gpt-4o-mini"
        )
        mock_console_instance = MagicMock()

        result = await ping_model("gpt-4o-mini", mock_console_instance)

        assert result is False
        mock_console_instance.print.assert_called_once()
        call_args = mock_console_instance.print.call_args[0][0]
        assert "[red]✗[/red] gpt-4o-mini" in call_args
        assert "Rate Limit Exceeded" in call_args

    @pytest.mark.asyncio
    async def test_ping_model_api_connection_error(self, mock_litellm):
        """Test ping_model handling of connection errors"""
        mock_litellm.side_effect = APIConnectionError(
            message="Failed to connect to Ollama server at localhost:11434",
            llm_provider="ollama",
            model="ollama/deepseek-r1"
        )
        mock_console_instance = MagicMock()

        result = await ping_model("ollama/deepseek-r1", mock_console_instance)

        assert result is False
        mock_console_instance.print.assert_called_once()
        call_args = mock_console_instance.print.call_args[0][0]
        assert "[red]✗[/red] ollama/deepseek-r1" in call_args
        assert "Ollama server not running" in call_args
        assert "ollama serve" in call_args

    @pytest.mark.asyncio
    async def test_bench_model_authentication_error(self, mock_litellm):
        """Test that authentication errors propagate from bench_model"""
        mock_litellm.side_effect = AuthenticationError(
            message="Invalid API key",
            llm_provider="openai",
            model="gpt-4o-mini"
        )

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await bench_model("gpt-4o-mini", 500)

    @pytest.mark.asyncio
    async def test_bench_model_context_window_error(self, mock_litellm):
        """Test that context window errors propagate from bench_model"""
        mock_litellm.side_effect = ContextWindowExceededError(
            message="This model's maximum context length is 4096 tokens",
            llm_provider="openai",
            model="gpt-3.5-turbo"
        )

        with pytest.raises(ContextWindowExceededError, match="context length"):
            await bench_model("gpt-3.5-turbo", 5000)
