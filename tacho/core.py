import asyncio
import time
import logging
from statistics import mean, median
from collections import defaultdict

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
import litellm
from litellm import (
    AuthenticationError,
    NotFoundError,
    APIConnectionError,
    RateLimitError,
    BadRequestError,
)

# Suppress litellm debug output and logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

console = Console()

BENCHMARK_PROMPT = """Generate a ~2000 word summary of the history of the USA."""
VALIDATION_PROMPT = "Do you have time to help? (yes/no)"


def calculate_metrics(times: list[float], tokens: list[int]) -> dict:
    """Calculate performance metrics from benchmark results"""
    if not times or not tokens:
        return {}

    tokens_per_second = [t / time for t, time in zip(tokens, times) if time > 0]

    return {
        "mean_tps": mean(tokens_per_second) if tokens_per_second else 0,
        "median_tps": median(tokens_per_second) if tokens_per_second else 0,
        "min_tps": min(tokens_per_second) if tokens_per_second else 0,
        "max_tps": max(tokens_per_second) if tokens_per_second else 0,
        "avg_time": mean(times),
        "avg_tokens": mean(tokens),
    }


async def ping_model(model: str) -> bool:
    try:
        messages = [{"role": "user", "content": VALIDATION_PROMPT}]
        await litellm.acompletion(model, messages, max_tokens=1)
        console.print(f"[green]✓[/green] {model}")
        return True
    except AuthenticationError:
        console.print(f"[red]✗[/red] {model} - Authentication failed (check API key)")
        return False
    except NotFoundError:
        console.print(f"[red]✗[/red] {model} - Model not found")
        return False
    except (APIConnectionError, RateLimitError, BadRequestError) as e:
        # Log the specific error type for debugging
        error_type = type(e).__name__
        console.print(f"[red]✗[/red] {model} - {error_type}")
        return False


async def bench_model(model: str, prompt: str, max_tokens: int) -> tuple[float, int]:
    """Measure inference time for a single run and return time and tokens"""
    start_time = time.time()
    messages = [{"role": "user", "content": prompt}]
    response = await litellm.acompletion(model, messages, max_tokens=max_tokens)
    duration = time.time() - start_time
    tokens = response.usage.completion_tokens if response.usage else 0
    return duration, tokens


async def ping_models(models: list[str]) -> dict[str, tuple[bool, str]]:
    """Validate all models before benchmarking"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(
            description="[bold cyan]Checking Model Access...[bold cyan]", total=None
        )
        results = await asyncio.gather(*[ping_model(model) for model in models])
    return results


async def bench_models(
    models: list[str], prompt: str, runs: int, max_tokens: int
) -> dict:
    """Run benchmarks for all models in parallel"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(
            description="[bold cyan]Running Benchmark...[/bold cyan]", total=None
        )

        # Create tasks with model associations
        tasks = [
            (model, bench_model(model, prompt, max_tokens))
            for model in models
            for _ in range(runs)
        ]
        
        # Execute all benchmarks in parallel
        results = await asyncio.gather(
            *[task for _, task in tasks], 
            return_exceptions=True
        )

    # Group results by model
    model_results = defaultdict(list)
    for (model, _), result in zip(tasks, results):
        if isinstance(result, tuple) and len(result) == 2:
            model_results[model].append(result)

    # Calculate metrics for each model
    return {
        model: calculate_metrics(
            [time for time, _ in data],
            [tokens for _, tokens in data]
        ) if data else {}
        for model, data in model_results.items()
    }
