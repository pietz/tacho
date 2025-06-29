import pytest
import os
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock
from tacho.config import get_env_path, ensure_env_file, configure_logging, load_env


class TestGetEnvPath:
    def test_get_env_path_returns_correct_path(self):
        expected_path = Path.home() / ".tacho" / ".env"
        assert get_env_path() == expected_path


class TestEnsureEnvFile:
    @patch("tacho.config.Path.mkdir")
    @patch("tacho.config.Path.exists")
    @patch("tacho.config.Path.write_text")
    @patch("tacho.config.os.chmod")
    @patch("rich.console.Console")
    def test_ensure_env_file_creates_new_file(self, mock_console_class, mock_chmod, mock_write, mock_exists, mock_mkdir):
        # Setup mocks
        mock_exists.return_value = False
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Run function
        ensure_env_file()
        
        # Verify directory creation
        mock_mkdir.assert_called_once_with(exist_ok=True)
        
        # Verify file creation with template
        mock_write.assert_called_once()
        written_content = mock_write.call_args[0][0]
        assert "# Tacho Configuration File" in written_content
        assert "OPENAI_API_KEY" in written_content
        
        # Verify permissions set on Unix
        if os.name != 'nt':
            mock_chmod.assert_called_once()
        
        # Verify user notification
        assert mock_console.print.call_count == 2

    @patch("tacho.config.Path.exists")
    @patch("tacho.config.Path.mkdir")
    def test_ensure_env_file_skips_existing(self, mock_mkdir, mock_exists):
        # Setup mocks - file already exists
        mock_exists.return_value = True
        
        # Run function
        ensure_env_file()
        
        # Directory should still be created
        mock_mkdir.assert_called_once_with(exist_ok=True)
        
        # But no file operations should happen
        # (checked by not mocking write_text - would error if called)


class TestConfigureLogging:
    @patch("tacho.config.litellm")
    @patch("tacho.config.logging.getLogger")
    def test_configure_logging_sets_correct_levels(self, mock_get_logger, mock_litellm):
        # Setup mocks
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Run function
        configure_logging()
        
        # Verify litellm settings
        assert mock_litellm.suppress_debug_info is True
        assert mock_litellm.set_verbose is False
        
        # Verify logging levels
        assert mock_get_logger.call_count == 2
        mock_get_logger.assert_any_call("LiteLLM")
        mock_get_logger.assert_any_call("litellm")
        
        # Both loggers should be set to CRITICAL
        assert mock_logger.setLevel.call_count == 2
        for call in mock_logger.setLevel.call_args_list:
            assert call[0][0] == 50  # logging.CRITICAL = 50


class TestLoadEnv:
    @patch("tacho.config.load_dotenv")
    @patch("tacho.config.ensure_env_file")
    @patch("tacho.config.configure_logging")
    @patch("tacho.config.get_env_path")
    def test_load_env_calls_all_functions(self, mock_get_path, mock_configure, mock_ensure, mock_load_dotenv):
        # Setup mocks
        mock_env_path = Path("/fake/path/.env")
        mock_get_path.return_value = mock_env_path
        
        # Run function
        load_env()
        
        # Verify all functions called in correct order
        mock_ensure.assert_called_once()
        mock_get_path.assert_called_once()
        mock_load_dotenv.assert_called_once_with(mock_env_path)
        mock_configure.assert_called_once()


class TestWindowsCompatibility:
    @patch("os.name", "nt")  # Windows
    @patch("pathlib.Path.home")
    @patch("pathlib.Path.mkdir")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.write_text")
    @patch("os.chmod")
    @patch("rich.console.Console")
    def test_ensure_env_file_skips_chmod_on_windows(self, mock_console_class, mock_chmod, mock_write, mock_exists, mock_mkdir, mock_home):
        # Setup mocks
        import pathlib
        mock_home.return_value = pathlib.Path("/fake/home")
        mock_exists.return_value = False
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        
        # Run function
        ensure_env_file()
        
        # Verify chmod NOT called on Windows
        mock_chmod.assert_not_called()