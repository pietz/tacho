import time

import litellm
from litellm import (
    AuthenticationError,
    NotFoundError,
    RateLimitError,
    APIConnectionError,
)

BENCHMARK_PROMPT = """Generate a ~2000 word summary of the history of the USA."""
VALIDATION_PROMPT = "Do you have time to help? (yes/no)"


async def llm(model: str, prompt: str, tokens: int | None = None):
    messages = [{"role": "user", "content": prompt}]
    return await litellm.acompletion(model, messages, max_tokens=tokens)


async def ping_model(model: str, console) -> bool:
    try:
        await llm(model, VALIDATION_PROMPT, 10)
        console.print(f"[green]✓[/green] {model}")
        return True
    except AuthenticationError as e:
        error_msg = "Authentication Failed."
        if hasattr(e, "llm_provider") and e.llm_provider:
            provider = e.llm_provider.upper()
            if provider == "OPENAI":
                error_msg += " (OPENAI_API_KEY)"
            elif provider == "ANTHROPIC":
                error_msg += " (ANTHROPIC_API_KEY)"
            elif provider == "GEMINI":
                error_msg += " (GEMINI_API_KEY)"
            elif provider == "BEDROCK":
                error_msg += (
                    " (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION_NAME)"
                )
        console.print(f"[red]✗[/red] {model} - {error_msg}")
        return False
    except NotFoundError:
        console.print(f"[red]✗[/red] {model} - Model Not Found")
        return False
    except RateLimitError:
        console.print(f"[red]✗[/red] {model} - Rate Limit Exceeded")
        return False
    except APIConnectionError as e:
        error_msg = str(e)
        if "ollama" in model.lower() and "localhost:11434" in error_msg.lower():
            console.print(
                f"[red]✗[/red] {model} - Ollama server not running. Start with 'ollama serve'"
            )
        else:
            console.print(
                f"[red]✗[/red] {model} - Connection failed. Check network/service availability"
            )
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] {model} - {str(e)[:200]}")
        return False


async def bench_model(model: str, max_tokens: int) -> tuple[float, int]:
    """Measure inference time for a single run and return time and tokens"""
    start_time = time.time()
    res = await llm(model, BENCHMARK_PROMPT, max_tokens)
    duration = time.time() - start_time

    tokens = res.usage.completion_tokens
    if (
        hasattr(res.usage, "completion_tokens_details")
        and res.usage.completion_tokens_details
    ):
        if hasattr(res.usage.completion_tokens_details, "reasoning_tokens"):
            tokens += res.usage.completion_tokens_details.reasoning_tokens

    return duration, tokens
