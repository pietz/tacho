# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Tacho is a CLI tool for benchmarking LLM inference speeds across multiple models and providers. It measures tokens/second metrics to compare model performance.

## Development Setup

This project uses `uv` for Python dependency management. Key commands:

```bash
# Install dependencies
uv sync

# Run the CLI directly
tacho gpt-4.1-mini gemini-2.0-flash

# Build the package
uv build

# Publish on Pypi is done by the user
```

## Architecture

The project has been refactored into a modular structure:

- **Entry point**: `tacho.cli:main` - wrapper function that uses `os._exit()` to suppress warnings
- **Core modules**:
  - `cli.py`: Main CLI app using Typer, handles command parsing
  - `ai.py`: Core benchmarking logic with async model benchmarking
  - `display.py`: Results presentation and metrics calculation
  - `config.py`: Configuration and constants
- **Main functions**:
  - `validate_models()`: Pre-flight validation of model availability
  - `benchmark_model()`: Core benchmarking logic in ai.py
  - `calculate_metrics()`: Extracts performance metrics from raw benchmark data (in display.py)
  - `run_benchmarks()`: Orchestrates parallel benchmarking of multiple models

## Key Design Decisions

1. **Async/parallel execution**: All benchmarks run concurrently using asyncio for performance.
2. **Progress tracking**: Uses Rich library with simple callback pattern (no complex queues).
3. **Error handling**: Pragmatic approach - validation errors are mapped to user-friendly messages.
4. **CLI design**: Supports both `tacho model1 model2` and `tacho bench model1 model2` syntax via Typer callback.

## Testing

The project includes a comprehensive test suite in the `tests/` directory:

- **Test structure**:
  - `conftest.py`: Shared fixtures and mocks for all tests
  - `test_config.py`: Tests for configuration and environment handling
  - `test_ai.py`: Tests for LLM interaction and benchmarking
  - `test_display.py`: Tests for metrics calculation and display
  - `test_cli.py`: Tests for CLI commands and argument parsing

- **Running tests**:
  ```bash
  # Run all tests
  uv run pytest
  
  # Run with verbose output
  uv run pytest -v
  
  # Run specific test marker
  uv run pytest -m unit
  ```

- **Key testing patterns**:
  - All external API calls are mocked (no API keys needed for tests)
  - Async functions are tested with `pytest-asyncio`
  - CLI testing uses Typer's test runner
  - Fixtures provide isolated test environments

## Common Issues

- **API keys**: Models require environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`)
- **Unclosed session warnings**: The `main()` function uses `os._exit()` to suppress "Unclosed client session" warnings from aiohttp. These warnings are caused by litellm with multiple providers (gemini, ollama, ...) not properly closing HTTP sessions. The warnings are harmless but appear during normal Python cleanup. Using `os._exit()` bypasses the cleanup phase where these warnings would be printed.

## Import Notes

- Keep README.md and CLAUDE.md up-to-date
- Bump version ONLY when the user says so, also reflecting changes in CHANGELOG.md
- Do not build and or publish the package. The user does that.