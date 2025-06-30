from statistics import mean
from unittest.mock import AsyncMock, MagicMock

import pytest

from tacho.display import calculate_metrics, display_results, run_pings, run_benchmarks


@pytest.mark.unit
class TestDisplay:
    def test_calculate_metrics_normal_case(self):
        """Test metrics calculation with normal benchmark results"""
        stats = [(2.0, 100), (2.5, 125), (3.0, 150)]

        metrics = calculate_metrics(stats)

        # Expected: [mean_tps, min_tps, max_tps, mean_time]
        assert len(metrics) == 4

        # Calculate expected values
        tps_values = [100 / 2.0, 125 / 2.5, 150 / 3.0]  # [50, 50, 50]
        assert metrics[0] == mean(tps_values)  # mean tps = 50
        assert metrics[1] == min(tps_values)  # min tps = 50
        assert metrics[2] == max(tps_values)  # max tps = 50
        assert metrics[3] == mean([2.0, 2.5, 3.0])  # mean time = 2.5

    def test_calculate_metrics_varying_performance(self):
        """Test metrics with varying performance results"""
        stats = [(1.0, 100), (2.0, 100), (4.0, 100)]

        metrics = calculate_metrics(stats)

        # TPS: [100, 50, 25]
        assert metrics[0] == pytest.approx(58.33, rel=0.01)  # mean
        assert metrics[1] == 25  # min
        assert metrics[2] == 100  # max
        assert metrics[3] == pytest.approx(2.33, rel=0.01)  # mean time

    def test_calculate_metrics_zero_time_handling(self):
        """Test that zero times are filtered out"""
        stats = [(0, 100), (2.0, 100), (3.0, 150)]

        metrics = calculate_metrics(stats)

        # Should only calculate from non-zero times
        tps_values = [100 / 2.0, 150 / 3.0]  # [50, 50]
        assert metrics[0] == 50  # mean tps
        assert metrics[1] == 50  # min tps
        assert metrics[2] == 50  # max tps
        assert metrics[3] == pytest.approx(1.67, rel=0.01)  # mean time (including zero)

    def test_calculate_metrics_empty_results(self):
        """Test metrics calculation with empty results"""
        stats = []

        # Should raise StatisticsError with empty stats
        with pytest.raises(Exception):  # mean() raises on empty list
            calculate_metrics(stats)

    @pytest.mark.asyncio
    async def test_run_pings(self, mocker, mock_progress):
        """Test run_pings orchestration"""
        # Mock ping_model
        mock_ping = mocker.patch("tacho.display.ping_model", new_callable=AsyncMock)
        mock_ping.side_effect = [True, False, True]  # Mixed results

        models = ["gpt-4", "invalid-model", "claude-3"]
        results = await run_pings(models)

        assert results == [True, False, True]
        assert mock_ping.call_count == 3

        # Verify progress was created and task added
        mock_progress.add_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_benchmarks(self, mocker, mock_progress):
        """Test run_benchmarks orchestration"""
        # Mock bench_model
        mock_bench = mocker.patch("tacho.display.bench_model", new_callable=AsyncMock)
        mock_bench.side_effect = [
            (2.0, 100),
            (2.1, 102),
            (2.2, 104),  # Model 1, 3 runs
            (1.5, 95),
            (1.6, 97),
            (1.7, 99),  # Model 2, 3 runs
        ]

        models = ["gpt-4", "claude-3"]
        runs = 3
        tokens = 500

        results = await run_benchmarks(models, runs, tokens)

        assert len(results) == 6  # 2 models × 3 runs
        assert results == [
            (2.0, 100),
            (2.1, 102),
            (2.2, 104),
            (1.5, 95),
            (1.6, 97),
            (1.7, 99),
        ]

        # Verify bench_model was called correctly
        assert mock_bench.call_count == 6
        for call in mock_bench.call_args_list:
            assert call[0][1] == tokens  # max_tokens argument

    def test_display_results(self, mocker, sample_benchmark_results):
        """Test results display formatting"""
        # Mock console and Table
        mock_console = MagicMock()
        mocker.patch("tacho.display.console", mock_console)

        mock_table = MagicMock()
        mock_table_class = mocker.patch("tacho.display.Table", return_value=mock_table)

        models = ["gpt-4", "claude-3"]
        runs = 3

        display_results(models, runs, sample_benchmark_results)

        # Verify Table was created
        mock_table_class.assert_called_once()

        # Verify columns were added
        assert mock_table.add_column.call_count == 5  # Model, Avg, Min, Max, Time

        # Verify rows were added (one per model)
        assert mock_table.add_row.call_count == 2

        # Verify console.print was called with the table
        mock_console.print.assert_called_once_with(mock_table)
