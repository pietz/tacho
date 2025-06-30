import os
import logging
from pathlib import Path

from dotenv import load_dotenv
import litellm


def get_env_path() -> Path:
    """Get the path to the .env file."""
    env_dir = Path.home() / ".tacho"
    return env_dir / ".env"


def ensure_env_file():
    """Create .env file with helpful comments if it doesn't exist."""
    env_path = get_env_path()
    env_dir = env_path.parent

    # Create directory if needed
    env_dir.mkdir(exist_ok=True)

    # Create .env file with comments if it doesn't exist
    if not env_path.exists():
        template = """# Tacho Configuration File
# Add your API keys here to avoid exporting them each time
# OPENAI_API_KEY=sk-...
#
# ANTHROPIC_API_KEY=sk-ant-...
#
# GEMINI_API_KEY=...
#
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# AWS_REGION_NAME=us-east-1
#
# GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# VERTEXAI_PROJECT=your-gcp-project-id
# VERTEXAI_LOCATION=us-east5  # optional, defaults to us-east5
#
# For more providers, see: https://docs.litellm.ai/docs/providers

"""
        env_path.write_text(template)

        # Set restrictive permissions on Unix-like systems
        if os.name != "nt":  # Not Windows
            os.chmod(env_path, 0o600)

        # Notify user about the created file
        from rich.console import Console

        console = Console()
        console.print(f"\n[yellow]Created config file at {env_path}[/yellow]")
        console.print(
            "Add your API keys to this file to avoid exporting them each time.\n"
        )


def configure_logging():
    """Configure logging to suppress verbose litellm output."""
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
    logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
    logging.getLogger("litellm").setLevel(logging.CRITICAL)


def load_env():
    """Load environment variables from ~/.tacho/.env file."""
    ensure_env_file()
    env_path = get_env_path()
    load_dotenv(env_path)

    # Also configure logging when loading env
    configure_logging()
