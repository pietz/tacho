import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from tacho.core import calculate_metrics, ping_model, bench_model, ping_models, bench_models


class TestCalculateMetrics:
    def test_calculate_metrics_with_valid_data(self):
        times = [1.0, 2.0, 3.0]
        tokens = [100, 200, 300]
        
        result = calculate_metrics(times, tokens)
        
        assert result["mean_tps"] == 100.0  # Average of 100/1, 200/2, 300/3
        assert result["median_tps"] == 100.0
        assert result["min_tps"] == 100.0
        assert result["max_tps"] == 100.0
        assert result["avg_time"] == 2.0
        assert result["avg_tokens"] == 200.0

    def test_calculate_metrics_with_empty_lists(self):
        result = calculate_metrics([], [])
        assert result == {}

    def test_calculate_metrics_with_zero_time(self):
        times = [0.0, 1.0, 2.0]
        tokens = [100, 200, 300]
        
        result = calculate_metrics(times, tokens)
        
        # Should skip the zero time entry
        assert result["mean_tps"] == 175.0  # Average of 200/1 and 300/2
        assert result["avg_time"] == 1.0  # Average of 0, 1, 2
        assert result["avg_tokens"] == 200.0

    def test_calculate_metrics_with_varying_performance(self):
        times = [0.5, 1.0, 2.0, 4.0]
        tokens = [50, 100, 100, 100]
        
        result = calculate_metrics(times, tokens)
        
        # Corrected comment: 100 + 100 + 50 + 25 = 275, 275/4 = 68.75
        assert result["mean_tps"] == pytest.approx(68.75, rel=1e-2)
        assert result["min_tps"] == 25.0
        assert result["max_tps"] == 100.0
        assert result["avg_time"] == pytest.approx(1.875, rel=1e-2)
        assert result["avg_tokens"] == pytest.approx(87.5, rel=1e-2)


class TestPingModel:
    @pytest.mark.asyncio
    async def test_ping_model_success(self, mock_llm_completion, mock_console):
        model = "gpt-4o-mini"
        
        result = await ping_model(model)
        
        assert result is True
        mock_llm_completion.assert_called_once_with(
            model, 
            [{"role": "user", "content": "Do you have time to help? (yes/no)"}], 
            max_tokens=1
        )
        mock_console.print.assert_any_call(f"[green]✓[/green] {model}")

    @pytest.mark.asyncio
    async def test_ping_model_failure(self, monkeypatch, mock_console):
        model = "invalid-model"
        error_msg = "Model not found"
        
        async def mock_llm_error(*args, **kwargs):
            raise Exception(error_msg)
        
        monkeypatch.setattr("tacho.core.llm", mock_llm_error)
        
        result = await ping_model(model)
        
        assert result is False
        mock_console.print.assert_any_call(f"[red]✗[/red] {model} - {error_msg}")


class TestBenchModel:
    @pytest.mark.asyncio
    async def test_bench_model_success(self, mock_llm_completion, mock_response):
        model = "gpt-4o-mini"
        max_tokens = 500
        mock_response.usage.completion_tokens = 250
        
        duration, tokens = await bench_model(model, max_tokens)
        
        assert duration > 0
        assert tokens == 250
        mock_llm_completion.assert_called_once_with(
            model,
            [{"role": "user", "content": "Generate a ~2000 word summary of the history of the USA."}],
            max_tokens=max_tokens
        )

    @pytest.mark.asyncio
    async def test_bench_model_no_usage(self, monkeypatch):
        model = "gpt-4o-mini"
        max_tokens = 500
        
        response = Mock()
        response.usage = None
        
        async def mock_llm(*args, **kwargs):
            import time
            await asyncio.sleep(0.01)  # Simulate some processing time
            return response
        
        monkeypatch.setattr("tacho.core.llm", mock_llm)
        
        duration, tokens = await bench_model(model, max_tokens)
        
        assert duration > 0
        assert tokens == 0


class TestPingModels:
    @pytest.mark.asyncio
    async def test_ping_models_all_success(self, mock_llm_completion, mock_console, mock_progress):
        models = ["gpt-4o-mini", "claude-3-opus"]
        
        results = await ping_models(models)
        
        assert results == [True, True]
        assert mock_llm_completion.call_count == 2

    @pytest.mark.asyncio
    async def test_ping_models_mixed_results(self, monkeypatch, mock_console, mock_progress):
        models = ["gpt-4o-mini", "invalid-model", "claude-3-opus"]
        
        call_count = 0
        async def mock_llm(model, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if model == "invalid-model":
                raise Exception("Model not found")
            return Mock(usage=Mock(completion_tokens=1))
        
        monkeypatch.setattr("tacho.core.llm", mock_llm)
        
        results = await ping_models(models)
        
        assert results == [True, False, True]
        assert call_count == 3


class TestBenchModels:
    @pytest.mark.asyncio
    async def test_bench_models_success(self, monkeypatch, mock_progress):
        models = ["gpt-4o-mini", "claude-3-opus"]
        runs = 2
        max_tokens = 100
        
        # Mock bench_model to return predictable results
        async def mock_bench(model, tokens):
            if model == "gpt-4o-mini":
                return (1.0, 100)
            else:
                return (2.0, 200)
        
        monkeypatch.setattr("tacho.core.bench_model", mock_bench)
        
        results = await bench_models(models, runs, max_tokens)
        
        assert "gpt-4o-mini" in results
        assert "claude-3-opus" in results
        
        # Check gpt-4o-mini metrics
        gpt_metrics = results["gpt-4o-mini"]
        assert gpt_metrics["mean_tps"] == 100.0
        assert gpt_metrics["avg_time"] == 1.0
        assert gpt_metrics["avg_tokens"] == 100.0
        
        # Check claude-3-opus metrics
        claude_metrics = results["claude-3-opus"]
        assert claude_metrics["mean_tps"] == 100.0
        assert claude_metrics["avg_time"] == 2.0
        assert claude_metrics["avg_tokens"] == 200.0

    @pytest.mark.asyncio
    async def test_bench_models_with_exceptions(self, monkeypatch, mock_progress):
        models = ["gpt-4o-mini", "failing-model"]
        runs = 1
        max_tokens = 100
        
        async def mock_bench(model, tokens):
            if model == "failing-model":
                raise Exception("Benchmark failed")
            return (1.0, 100)
        
        monkeypatch.setattr("tacho.core.bench_model", mock_bench)
        
        results = await bench_models(models, runs, max_tokens)
        
        # Should only have results for successful model
        assert "gpt-4o-mini" in results
        assert "failing-model" not in results or results["failing-model"] == {}