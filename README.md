# ⚡ tacho - LLM Speed Test

A fast CLI tool for benchmarking LLM inference speed across multiple models and providers. Get tokens/second metrics to compare model performance.


## Quick Start

Set up your API keys:

```bash
export OPENAI_API_KEY=<your-key-here>
export GEMINI_API_KEY=<your-key-here>
```

Run a benchmark (requires `uv`):

```bash
uvx tacho gpt-4.1 gemini/gemini-2.5-pro vertex_ai/claude-sonnet-4@20250514
✓ gpt-4.1
✓ vertex_ai/claude-sonnet-4@20250514
✓ gemini/gemini-2.5-pro
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━┳━━━━━━━┳━━━━━━━━┓
┃ Model                              ┃ Avg t/s ┃ Min t/s ┃ Max t/s ┃  Time ┃ Tokens ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━╇━━━━━━━╇━━━━━━━━┩
│ gemini/gemini-2.5-pro              │    80.0 │    56.7 │   128.4 │ 13.5s │    998 │
│ vertex_ai/claude-sonnet-4@20250514 │    48.9 │    44.9 │    51.6 │ 10.2s │    500 │
│ gpt-4.1                            │    41.5 │    35.1 │    49.9 │ 12.3s │    500 │
└────────────────────────────────────┴─────────┴─────────┴─────────┴───────┴────────┘
```

> With its default settings, tacho generates 5 runs of 500 tokens each per model producing some inference costs.


## Features

- **Parallel benchmarking** - All models and runs execute concurrently for faster results
- **Token-based metrics** - Measures actual tokens/second, not just response time
- **Multi-provider support** - Works with any provider supported by LiteLLM (OpenAI, Anthropic, Google, Cohere, etc.)
- **Configurable token limits** - Control response length for consistent comparisons
- **Pre-flight validation** - Checks model availability and authentication before benchmarking
- **Graceful error handling** - Clear error messages for authentication, rate limits, and connection issues


## Installation

For regular use, install with `uv`:

```bash
uv tool install tacho
```

Or with pip:

```bash
pip install tacho
```

## Usage

### Basic benchmark

```bash
# Compare models with default settings (5 runs, 500 token limit)
tacho gpt-4.1-nano gemini/gemini-2.0-flash

# Custom settings (options must come before model names)
tacho --runs 3 --tokens 1000 gpt-4.1-nano gemini/gemini-2.0-flash
tacho -r 3 -t 1000 gpt-4.1-nano gemini/gemini-2.0-flash
```

### Command options

- `--runs, -r`: Number of inference runs per model (default: 5)
- `--tokens, -t`: Maximum tokens to generate per response (default: 500)
- `--prompt, -p`: Custom prompt for benchmarking

**Note:** When using the shorthand syntax (without the `bench` subcommand), options must be placed before model names. For example:
- ✅ `tacho -t 2000 gpt-4.1-mini`
- ❌ `tacho gpt-4.1-mini -t 2000`

## Output

Tacho displays a clean comparison table showing:
- **Avg/Min/Max tokens per second** - Primary performance metrics
- **Average time** - Average time per inference run

Models are sorted by performance (highest tokens/second first).

## Supported Providers

Tacho works with any provider supported by LiteLLM.

## Development

To contribute to Tacho, clone the repository and install development dependencies:

```bash
git clone https://github.com/pietz/tacho.git
cd tacho
uv sync
```

### Running Tests

Tacho includes a comprehensive test suite with full mocking of external API calls:

```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest tests/test_cli.py

# Run with coverage
uv run pytest --cov=tacho
```

The test suite includes:
- Unit tests for all core modules
- Mocked LiteLLM API calls (no API keys required)
- CLI command testing
- Async function testing
- Edge case coverage
