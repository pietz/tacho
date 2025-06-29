from unittest.mock import AsyncMock, MagicMock

import pytest
import typer
from typer.testing import CliRunner

from tacho.cli import app, bench, ping, version_callback


@pytest.mark.unit
class TestCLI:
    @pytest.fixture
    def runner(self):
        """Create CLI test runner"""
        return CliRunner()
    
    def test_version_callback(self, mocker):
        """Test version display callback"""
        mock_console = MagicMock()
        mocker.patch('tacho.cli.console', mock_console)
        mocker.patch('importlib.metadata.version', return_value='1.0.0')
        
        with pytest.raises(typer.Exit):
            version_callback(True)
        
        mock_console.print.assert_called_once_with("tacho 1.0.0")
    
    def test_version_callback_not_called(self):
        """Test version callback when version flag is False"""
        # Should not raise Exit when False
        version_callback(False)  # Should complete normally
    
    def test_bench_command_success(self, runner, mocker):
        """Test successful benchmark command"""
        # Test the bench function directly to avoid callback issues
        mock_asyncio_run = mocker.patch('tacho.cli.asyncio.run')
        
        # First call to run_pings
        mock_asyncio_run.side_effect = [
            [True, True],  # run_pings result
            [(2.0, 100), (2.1, 102), (2.2, 104), (2.3, 106), (2.4, 108),  # run_benchmarks results
             (1.5, 95), (1.6, 97), (1.7, 99), (1.8, 101), (1.9, 103)]
        ]
        
        mock_display = mocker.patch('tacho.cli.display_results')
        
        # Call bench directly
        bench(["gpt-4", "claude-3"], runs=5, lim=500)
        
        # Verify display was called with correct arguments
        mock_display.assert_called_once()
        call_args = mock_display.call_args[0]
        assert call_args[0] == ["gpt-4", "claude-3"]  # models
        assert call_args[1] == 5  # runs
    
    def test_bench_command_no_valid_models(self, runner, mocker):
        """Test benchmark command when no models are valid"""
        mock_run_pings = mocker.patch('tacho.cli.run_pings', new_callable=AsyncMock)
        mock_run_pings.return_value = [False, False]  # No valid models
        
        result = runner.invoke(app, ["bench", "invalid1", "invalid2"])
        
        assert result.exit_code == 1
        mock_run_pings.assert_called_once()
    
    def test_bench_command_with_options(self, runner, mocker):
        """Test benchmark command with custom options"""
        mock_asyncio_run = mocker.patch('tacho.cli.asyncio.run')
        
        # Configure asyncio.run mock responses
        mock_asyncio_run.side_effect = [
            [True],  # run_pings result
            [(2.0 + i*0.1, 100 + i) for i in range(10)]  # run_benchmarks results
        ]
        
        mock_display = mocker.patch('tacho.cli.display_results')
        
        # Call bench directly with options
        bench(["gpt-4"], runs=10, lim=1000)
        
        # Verify display was called
        mock_display.assert_called_once()
        call_args = mock_display.call_args[0]
        assert call_args[0] == ["gpt-4"]
        assert call_args[1] == 10
    
    def test_ping_command_success(self, runner, mocker):
        """Test successful ping command"""
        mock_asyncio_run = mocker.patch('tacho.cli.asyncio.run')
        mock_asyncio_run.return_value = [True, False, True]
        
        # Call ping directly
        ping(["gpt-4", "invalid", "claude-3"])
        
        # Should not raise Exit since at least one model succeeded
        mock_asyncio_run.assert_called_once()
    
    def test_ping_command_all_fail(self, runner, mocker):
        """Test ping command when all models fail"""
        mock_run_pings = mocker.patch('tacho.cli.run_pings', new_callable=AsyncMock)
        mock_run_pings.return_value = [False, False]
        
        result = runner.invoke(app, ["ping", "invalid1", "invalid2"])
        
        assert result.exit_code == 1
    
    def test_default_command_invocation(self, runner, mocker):
        """Test that models can be passed without 'bench' subcommand"""
        mock_run_pings = mocker.patch('tacho.cli.run_pings', new_callable=AsyncMock)
        mock_run_pings.return_value = [True, True]
        
        mock_run_benchmarks = mocker.patch('tacho.cli.run_benchmarks', new_callable=AsyncMock)
        mock_run_benchmarks.return_value = [(2.0, 100), (2.1, 102)]
        
        mocker.patch('tacho.cli.display_results')
        
        # Direct invocation without 'bench' subcommand
        result = runner.invoke(app, ["gpt-4", "claude-3"])
        
        assert result.exit_code == 0
        mock_run_pings.assert_called_once()
        mock_run_benchmarks.assert_called_once()
    
    def test_help_command(self, runner):
        """Test help display"""
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "CLI tool for measuring LLM inference speeds" in result.output
    
    def test_bench_function_partial_valid_models(self, mocker):
        """Test bench function filters out invalid models"""
        mock_asyncio_run = mocker.patch('tacho.cli.asyncio.run')
        
        # Configure responses: first ping returns mixed results
        mock_asyncio_run.side_effect = [
            [True, False, True],  # run_pings: gpt-4 valid, invalid fails, claude-3 valid
            [(2.0, 100), (2.1, 102), (2.2, 104),  # Results for gpt-4 (3 runs)
             (1.8, 95), (1.9, 97), (2.0, 99)]     # Results for claude-3 (3 runs)
        ]
        
        mock_display = mocker.patch('tacho.cli.display_results')
        
        # Call bench with mixed valid/invalid models
        bench(["gpt-4", "invalid", "claude-3"], runs=3, lim=250)
        
        # Verify only valid models were benchmarked
        mock_display.assert_called_once()
        call_args = mock_display.call_args[0]
        assert call_args[0] == ["gpt-4", "claude-3"]  # Only valid models
        assert call_args[1] == 3