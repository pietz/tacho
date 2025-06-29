import pytest
import asyncio
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def mock_response():
    """Create a mock LiteLLM response"""
    response = Mock()
    response.usage = Mock()
    response.usage.completion_tokens = 100
    return response


@pytest.fixture
def mock_llm_completion(monkeypatch, mock_response):
    """Mock the litellm.acompletion function"""
    async_mock = AsyncMock(return_value=mock_response)
    monkeypatch.setattr("litellm.acompletion", async_mock)
    return async_mock


@pytest.fixture
def mock_console(monkeypatch):
    """Mock the Rich console to prevent output during tests"""
    console_mock = Mock()
    console_mock.print = Mock()
    monkeypatch.setattr("tacho.core.console", console_mock)
    monkeypatch.setattr("tacho.cli.console", console_mock)
    return console_mock


@pytest.fixture
def mock_progress(monkeypatch):
    """Mock Rich Progress to prevent output during tests"""
    class MockProgress:
        def __init__(self, *args, **kwargs):
            pass
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def add_task(self, **kwargs):
            return 0
    
    monkeypatch.setattr("tacho.core.Progress", MockProgress)
    return MockProgress


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()