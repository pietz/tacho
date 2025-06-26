# tacho - LLM Speed Test

A fast CLI tool for benchmarking LLM inference speed across multiple models and providers. Get tokens/second metrics to compare model performance.


## Quick Start

Set up your API keys:

```bash
export OPENAI_API_KEY=your-key-here
export ANTHROPIC_API_KEY=your-key-here
```

Run a benchmark (requires `uv`):

```bash
uvx tacho gpt-4.1 claude-sonnet-4
```


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
tacho gpt-4o gpt-4o-mini claude-3-haiku-20240307

# Custom settings
tacho gpt-3.5-turbo claude-3-sonnet-20240229 --runs 10 --lim 500
```

### Test model availability

```bash
# Check if models are accessible before benchmarking
tacho test-models gpt-4 claude-3-opus-20240229 gemini-pro
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

Tacho works with any provider supported by LiteLLM, including:
- OpenAI (GPT-4, GPT-3.5, etc.)
- Anthropic (Claude 3 family)
- Google (Gemini Pro, PaLM)
- Cohere (Command, Command-R)
- Together AI (Llama, Mixtral, etc.)
- Groq
- And many more...

Just ensure you have the appropriate API keys set as environment variables.

## License

MIT