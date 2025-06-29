import asyncio
from pathlib import Path
from unittest.mock import MagicMock
try:
    from unittest.mock import AsyncMock
except ImportError:
    # Python < 3.8 compatibility
    from mock import AsyncMock

import pytest


@pytest.fixture
def mock_litellm(mocker):
    """Mock litellm.acompletion to avoid actual API calls"""
    mock = mocker.patch('litellm.acompletion', new_callable=AsyncMock)
    
    # Default mock response structure
    mock_response = MagicMock()
    mock_response.usage.completion_tokens = 100
    mock_response.choices = [MagicMock(message=MagicMock(content="Mock response"))]
    
    mock.return_value = mock_response
    return mock


@pytest.fixture
def mock_console(mocker):
    """Mock Rich console to capture output"""
    console_mock = MagicMock()
    mocker.patch('tacho.display.console', console_mock)
    mocker.patch('tacho.cli.console', console_mock)
    # Note: tacho.ai doesn't use console directly anymore
    return console_mock


@pytest.fixture
def temp_tacho_dir(tmp_path, mocker):
    """Create temporary .tacho directory for testing"""
    tacho_dir = tmp_path / ".tacho"
    tacho_dir.mkdir()
    
    # Mock the get_env_path function to use our temp directory
    mocker.patch('tacho.config.get_env_path', return_value=tacho_dir / ".env")
    
    return tacho_dir


@pytest.fixture
def mock_load_dotenv(mocker):
    """Mock dotenv loading"""
    return mocker.patch('tacho.config.load_dotenv')


@pytest.fixture
def mock_progress(mocker):
    """Mock Rich Progress for testing progress indicators"""
    progress_mock = MagicMock()
    progress_mock.__enter__ = MagicMock(return_value=progress_mock)
    progress_mock.__exit__ = MagicMock(return_value=None)
    progress_mock.add_task = MagicMock(return_value="task_id")
    progress_mock.console = MagicMock()
    
    mocker.patch('tacho.display.Progress', return_value=progress_mock)
    return progress_mock


@pytest.fixture
def sample_benchmark_results():
    """Sample benchmark results for testing display functions"""
    return [
        (2.5, 100),  # Model 1, Run 1: 2.5s, 100 tokens
        (2.3, 98),   # Model 1, Run 2: 2.3s, 98 tokens
        (2.7, 102),  # Model 1, Run 3: 2.7s, 102 tokens
        (1.8, 95),   # Model 2, Run 1: 1.8s, 95 tokens
        (1.9, 97),   # Model 2, Run 2: 1.9s, 97 tokens
        (1.7, 93),   # Model 2, Run 3: 1.7s, 93 tokens
    ]


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing"""
    env_vars = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GEMINI_API_KEY": "test-gemini-key",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture(autouse=True)
def mock_cli_load_env(mocker):
    """Mock the load_env call in cli module to prevent file operations during import"""
    mocker.patch('tacho.cli.load_env')