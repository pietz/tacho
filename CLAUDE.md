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

The project is intentionally simple with all logic in a single module (`tacho/cli.py`):

- **Entry point**: `tacho:app` (Typer CLI app)
- **Main functions**:
  - `validate_models()`: Pre-flight validation of model availability
  - `benchmark_model()`: Core benchmarking logic with optional progress tracking
  - `calculate_metrics()`: Extracts performance metrics from raw benchmark data
  - `run_benchmarks()`: Orchestrates parallel benchmarking of multiple models

## Key Design Decisions

1. **Single file architecture**: All code is in `cli.py` for simplicity. Consider splitting once it gets too large
2. **Async/parallel execution**: All benchmarks run concurrently using asyncio for performance.
3. **Progress tracking**: Uses Rich library with simple callback pattern (no complex queues).
4. **Error handling**: Pragmatic approach - validation errors are mapped to user-friendly messages.
5. **CLI design**: Supports both `tacho model1 model2` and `tacho benchmark model1 model2` syntax via Typer callback.

## Testing & Validation

Currently no test suite. To verify changes:

```bash
# Run a benchmark
tacho gpt-4o-mini --runs 2 --lim 100

# Check help
tacho --help
```

## Common Issues

- **API keys**: Models require environment variables (e.g., `OPENAI_API_KEY`, `GEMINI_API_KEY`)


## Import Notes

- Keep README.md and CLAUDE.md up-to-date
- Bump version ONLY when the user says so, also reflecting changes in CHANGELOG.md
- Do not build and or publish the package. The user does that.