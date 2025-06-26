# Tacho

CLI tool for measuring and comparing LLM inference speeds across different providers.

## Installation

```bash
pip install tacho
```

## Usage

```bash
# Benchmark multiple models
tacho gpt-3.5-turbo claude-3-haiku-20240307

# Custom number of runs
tacho gpt-4 gpt-3.5-turbo --runs 10

# List available providers
tacho list-providers
```

## Requirements

Set up API keys as environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- etc.

## License

MIT