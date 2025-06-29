import pytest
import asyncio
import typer
from typer.testing import CliRunner
from unittest.mock import Mock, patch, AsyncMock
from tacho.cli import app, version_callback


runner = CliRunner()


class TestVersionCallback:
    def test_version_callback_shows_version(self):
        with patch("importlib.metadata.version") as mock_version:
            mock_version.return_value = "1.0.0"
            
            with pytest.raises(typer.Exit):
                version_callback(True)
    
    def test_version_callback_no_action_when_false(self):
        # Should not raise an exception
        version_callback(False)


class TestBenchCommand:
    @patch("tacho.cli.console")
    @patch("tacho.cli.asyncio.run")
    def test_bench_command_success(self, mock_asyncio_run, mock_console):
        # Mock successful model validation and benchmarking
        mock_asyncio_run.side_effect = [
            [True, True],  # ping_models returns all models valid
            {
                "gpt-4o-mini": {
                    "mean_tps": 100.0,
                    "median_tps": 95.0,
                    "min_tps": 80.0,
                    "max_tps": 120.0,
                    "avg_time": 1.5,
                    "avg_tokens": 150
                },
                "claude-3-opus": {
                    "mean_tps": 90.0,
                    "median_tps": 88.0,
                    "min_tps": 70.0,
                    "max_tps": 110.0,
                    "avg_time": 2.0,
                    "avg_tokens": 180
                }
            }  # bench_models returns results
        ]
        
        result = runner.invoke(app, ["bench", "gpt-4o-mini", "claude-3-opus"])
        
        assert result.exit_code == 0
        # Check that asyncio.run was called twice (ping and bench)
        assert mock_asyncio_run.call_count == 2

    @patch("tacho.cli.asyncio.run")
    def test_bench_command_no_valid_models(self, mock_asyncio_run):
        # Mock all models as invalid
        mock_asyncio_run.return_value = [False, False]
        
        result = runner.invoke(app, ["bench", "invalid-1", "invalid-2"])
        
        assert result.exit_code == 1

    @patch("tacho.cli.console")
    @patch("tacho.cli.asyncio.run")
    def test_bench_command_partial_valid_models(self, mock_asyncio_run, mock_console):
        # Mock mixed validation results
        mock_asyncio_run.side_effect = [
            [True, False, True],  # Only first and third models are valid
            {
                "gpt-4o-mini": {
                    "mean_tps": 100.0,
                    "median_tps": 95.0,
                    "min_tps": 80.0,
                    "max_tps": 120.0,
                    "avg_time": 1.5,
                    "avg_tokens": 150
                },
                "claude-3-opus": {
                    "mean_tps": 90.0,
                    "median_tps": 88.0,
                    "min_tps": 70.0,
                    "max_tps": 110.0,
                    "avg_time": 2.0,
                    "avg_tokens": 180
                }
            }
        ]
        
        result = runner.invoke(app, ["bench", "gpt-4o-mini", "invalid-model", "claude-3-opus"])
        
        assert result.exit_code == 0

    @patch("tacho.cli.asyncio.run")
    def test_bench_command_no_results(self, mock_asyncio_run):
        # Mock valid models but no benchmark results
        mock_asyncio_run.side_effect = [
            [True, True],
            {}  # Empty results
        ]
        
        result = runner.invoke(app, ["bench", "model1", "model2"])
        
        assert result.exit_code == 1

    @patch("tacho.cli.bench_models")
    @patch("tacho.cli.ping_models")
    def test_bench_command_with_options(self, mock_ping, mock_bench):
        # Set up async mocks
        async def ping_result():
            return [True]
        
        async def bench_result():
            return {"model": {"mean_tps": 100, "median_tps": 100, "min_tps": 100, "max_tps": 100, "avg_time": 1.0, "avg_tokens": 100}}
        
        mock_ping.return_value = ping_result()
        mock_bench.return_value = bench_result()
        
        with patch("tacho.cli.asyncio.run") as mock_run:
            # Make asyncio.run actually await the coroutines
            mock_run.side_effect = lambda coro: asyncio.get_event_loop().run_until_complete(coro)
            
            result = runner.invoke(app, ["bench", "model", "--runs", "10", "--lim", "1000"])
            
            assert result.exit_code == 0
            # Check that bench_models was called with correct arguments
            mock_bench.assert_called_once_with(["model"], 10, 1000)


class TestPingCommand:
    @patch("tacho.cli.console")
    @patch("tacho.cli.asyncio.run")
    def test_ping_command_all_success(self, mock_asyncio_run, mock_console):
        mock_asyncio_run.return_value = [True, True, True]
        
        result = runner.invoke(app, ["ping", "model1", "model2", "model3"])
        
        assert result.exit_code == 0
        mock_console.print.assert_any_call()
        mock_console.print.assert_any_call("[bold green]All 3 models are accessible![/bold green]")

    @patch("tacho.cli.console")
    @patch("tacho.cli.asyncio.run")
    def test_ping_command_partial_success(self, mock_asyncio_run, mock_console):
        mock_asyncio_run.return_value = [True, False, True]
        
        result = runner.invoke(app, ["ping", "model1", "invalid", "model3"])
        
        assert result.exit_code == 0
        mock_console.print.assert_any_call("[bold yellow]2/3 models are accessible[/bold yellow]")

    @patch("tacho.cli.console")
    @patch("tacho.cli.asyncio.run")
    def test_ping_command_no_success(self, mock_asyncio_run, mock_console):
        mock_asyncio_run.return_value = [False, False]
        
        result = runner.invoke(app, ["ping", "invalid1", "invalid2"])
        
        assert result.exit_code == 1
        mock_console.print.assert_any_call("[bold red]No models are accessible[/bold red]")


class TestMainWithoutSubcommand:
    def test_main_without_subcommand_calls_bench(self):
        with patch("tacho.cli.bench") as mock_bench:
            # Mock to prevent actual execution
            mock_bench.return_value = None
            
            result = runner.invoke(app, ["model1", "model2"])
            
            mock_bench.assert_called_once_with(["model1", "model2"], 5, 500)

    def test_main_with_options_passes_to_bench(self):
        with patch("tacho.cli.bench") as mock_bench:
            # Mock to prevent actual execution
            mock_bench.return_value = None
            
            # When using shorthand syntax, options must come before models
            result = runner.invoke(app, ["--runs", "10", "--lim", "1000", "model1"])
            
            mock_bench.assert_called_once_with(["model1"], 10, 1000)


class TestMainFunction:
    @patch("tacho.cli.app")
    @patch("tacho.cli.os._exit")
    def test_main_handles_keyboard_interrupt(self, mock_exit, mock_app):
        mock_app.side_effect = KeyboardInterrupt()
        
        from tacho.cli import main
        main()
        
        mock_exit.assert_called_once_with(1)

    @patch("tacho.cli.app")
    @patch("tacho.cli.os._exit")
    def test_main_handles_system_exit(self, mock_exit, mock_app):
        mock_app.side_effect = SystemExit(42)
        
        from tacho.cli import main
        main()
        
        mock_exit.assert_called_once_with(42)

    @patch("tacho.cli.app")
    @patch("tacho.cli.os._exit")
    def test_main_handles_generic_exception(self, mock_exit, mock_app):
        mock_app.side_effect = Exception("Unexpected error")
        
        from tacho.cli import main
        main()
        
        mock_exit.assert_called_once_with(1)

    @patch("tacho.cli.app")
    @patch("tacho.cli.os._exit")
    def test_main_success(self, mock_exit, mock_app):
        mock_app.return_value = None
        
        from tacho.cli import main
        main()
        
        mock_exit.assert_called_once_with(0)