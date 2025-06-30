import os
from pathlib import Path

import pytest
import litellm

from tacho.config import get_env_path, ensure_env_file, load_env, configure_logging


@pytest.mark.unit
class TestConfig:
    def test_get_env_path(self):
        """Test that get_env_path returns the correct path"""
        expected_path = Path.home() / ".tacho" / ".env"
        assert get_env_path() == expected_path

    def test_ensure_env_file_creates_directory(self, temp_tacho_dir):
        """Test that ensure_env_file creates the .tacho directory"""
        env_path = temp_tacho_dir / ".env"

        ensure_env_file()

        assert temp_tacho_dir.exists()
        assert env_path.exists()

    def test_ensure_env_file_creates_template(self, temp_tacho_dir):
        """Test that ensure_env_file creates .env with correct template"""
        env_path = temp_tacho_dir / ".env"

        ensure_env_file()

        content = env_path.read_text()
        assert "# Tacho Configuration File" in content
        assert "OPENAI_API_KEY=sk-..." in content
        assert "ANTHROPIC_API_KEY=sk-ant-..." in content
        assert "GEMINI_API_KEY=..." in content

    @pytest.mark.skipif(
        os.name == "nt", reason="Permission test not applicable on Windows"
    )
    def test_ensure_env_file_sets_permissions(self, temp_tacho_dir):
        """Test that ensure_env_file sets restrictive permissions on Unix"""
        env_path = temp_tacho_dir / ".env"

        ensure_env_file()

        # Check file permissions (should be 0o600)
        stat_info = os.stat(env_path)
        assert stat_info.st_mode & 0o777 == 0o600

    def test_ensure_env_file_preserves_existing(self, temp_tacho_dir):
        """Test that ensure_env_file doesn't overwrite existing file"""
        env_path = temp_tacho_dir / ".env"

        # Create existing file with custom content
        custom_content = "CUSTOM_KEY=custom_value"
        env_path.write_text(custom_content)

        ensure_env_file()

        # Verify content wasn't changed
        assert env_path.read_text() == custom_content

    def test_configure_logging(self, mocker):
        """Test that configure_logging suppresses litellm output"""
        mock_logging = mocker.patch("logging.getLogger")
        # Patch the attributes, not the values themselves
        mocker.patch.object(litellm, "suppress_debug_info", False)
        mocker.patch.object(litellm, "set_verbose", True)

        configure_logging()

        # Verify litellm settings were changed
        assert litellm.suppress_debug_info is True
        assert litellm.set_verbose is False

        # Verify logging levels were set
        mock_logging.assert_any_call("LiteLLM")
        mock_logging.assert_any_call("litellm")

        # Verify critical level was set
        mock_logger = mock_logging.return_value
        assert mock_logger.setLevel.call_count >= 2

    def test_load_env(self, temp_tacho_dir, mock_load_dotenv, mocker):
        """Test that load_env loads from correct path and configures logging"""
        mock_ensure = mocker.patch("tacho.config.ensure_env_file")
        mock_configure = mocker.patch("tacho.config.configure_logging")

        load_env()

        # Verify ensure_env_file was called
        mock_ensure.assert_called_once()

        # Verify dotenv was loaded from correct path
        mock_load_dotenv.assert_called_once_with(temp_tacho_dir / ".env")

        # Verify logging was configured
        mock_configure.assert_called_once()
