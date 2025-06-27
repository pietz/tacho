# tacho - LLM Speed Test

A fast CLI tool for benchmarking LLM inference speed across multiple models and providers. Get tokens/second metrics to compare model performance.


## Quick Start

Set up your API keys:

```bash
export OPENAI_API_KEY=<your-key-here>
export GEMINI_API_KEY=<your-key-here>
```

Run a benchmark (requires `uv`):

```bash
uvx tacho gpt-4.1-nano gemini/gemini-2.0-flash

✓ gemini/gemini-2.0-flash
✓ gpt-4.1-nano
┌─────────────────────────┬───────────┬───────────┬───────────┬──────────┐
│ Model                   │ Avg tok/s │ Min tok/s │ Max tok/s │ Avg Time │
├─────────────────────────┼───────────┼───────────┼───────────┼──────────┤
│ gemini/gemini-2.0-flash │     124.0 │     110.5 │     136.6 │     4.0s │
│ gpt-4.1-nano            │     116.9 │     105.4 │     129.5 │     4.3s │
└─────────────────────────┴───────────┴───────────┴───────────┴──────────┘
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

# Custom settings
tacho gpt-4.1-nano gemini/gemini-2.0-flash --runs 3 --lim 1000
```

### Command options

- `--runs, -r`: Number of inference runs per model (default: 5)
- `--lim, -l`: Maximum tokens to generate per response (default: 500)
- `--prompt, -p`: Custom prompt for benchmarking

## Output

Tacho displays a clean comparison table showing:
- **Avg/Min/Max tokens per second** - Primary performance metrics
- **Average time** - Average time per inference run

Models are sorted by performance (highest tokens/second first).

## Supported Providers

Tacho works with any provider supported by LiteLLM.
