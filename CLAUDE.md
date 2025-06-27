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
tacho gpt-4.1-mini gemini-2.5-flash

# Build the package
uv build

# Publish on Pypi is done by the user
```

## Architecture

The project is intentionally simple with all logic in a single file (`tacho.py`):

- **Entry point**: `tacho:main` - wrapper function that uses `os._exit()` to suppress warnings
- **Main CLI app**: `app` - Typer CLI application
- **Main functions**:
  - `validate_models()`: Pre-flight validation of model availability
  - `benchmark_model()`: Core benchmarking logic with optional progress tracking
  - `calculate_metrics()`: Extracts performance metrics from raw benchmark data
  - `run_benchmarks()`: Orchestrates parallel benchmarking of multiple models

## Key Design Decisions

1. **Async/parallel execution**: All benchmarks run concurrently using asyncio for performance.
2. **Progress tracking**: Uses Rich library with simple callback pattern (no complex queues).
3. **Error handling**: Pragmatic approach - validation errors are mapped to user-friendly messages.
4. **CLI design**: Supports both `tacho model1 model2` and `tacho bench model1 model2` syntax via Typer callback.

## Testing & Validation

Test suite uses pytest. To run tests:

```bash
# Run all tests
pytest tests/ -v

# Run with API keys for integration tests
OPENAI_API_KEY=xxx GEMINI_API_KEY=xxx pytest tests/
```

## Common Issues

- **API keys**: Models require environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`)
- **Unclosed session warnings**: The `main()` function uses `os._exit()` to suppress "Unclosed client session" warnings from aiohttp. These warnings are caused by litellm with multiple providers (gemini, ollama, ...) not properly closing HTTP sessions. The warnings are harmless but appear during normal Python cleanup. Using `os._exit()` bypasses the cleanup phase where these warnings would be printed.

## Import Notes

- Keep README.md and CLAUDE.md up-to-date
- Bump version ONLY when the user says so, also reflecting changes in CHANGELOG.md
- Do not build and or publish the package. The user does that.