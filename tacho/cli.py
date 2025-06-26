import asyncio
import time
from typing import Optional
from statistics import mean, median
from importlib.metadata import version

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
import litellm

try:
    __version__ = version("tacho")
except Exception:
    __version__ = "dev"

app = typer.Typer(help="CLI tool for measuring and comparing LLM inference speeds")
console = Console()

BENCHMARK_PROMPT = """Generate a ~1000 word summary of the history of the USA."""
VALIDATION_PROMPT = "Do you have time to help? (yes/no)"


async def validate_models(models: list[str]) -> dict[str, tuple[bool, str]]:
    """Validate all models before benchmarking"""

    async def validate_model(model: str) -> tuple[str, tuple[bool, str]]:
        """Validate a single model"""
        try:
            response = await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": VALIDATION_PROMPT}],
                max_tokens=3,
            )
            return model, (True, "OK")
        except Exception as e:
            error_msg = str(e)
            if "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return model, (False, "Authentication failed")
            elif "model" in error_msg.lower() and "not found" in error_msg.lower():
                return model, (False, "Model not found")
            else:
                return model, (False, f"Error: {error_msg[:50]}...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="[bold cyan]Validating Models...", total=None)

        # Run all validations in parallel
        tasks = [validate_model(model) for model in models]
        results = await asyncio.gather(*tasks)

    # Show checkmark after validation completes
    console.print("[green]✓[/green] [bold cyan]Validating Models[/bold cyan]")

    return dict(results)


async def measure_inference(
    model: str, prompt: str, max_tokens: int
) -> tuple[float, int, str]:
    """Measure inference time for a single run and return time, tokens, and response"""
    start_time = time.time()

    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time

    completion_tokens = response.usage.completion_tokens if response.usage else 0
    content = response.choices[0].message.content

    return elapsed_time, completion_tokens, content


async def benchmark_model_simple(
    model: str, prompt: str, runs: int, max_tokens: int
) -> dict:
    """Benchmark a single model without progress bar"""
    # Create all inference tasks at once
    tasks = [measure_inference(model, prompt, max_tokens) for _ in range(runs)]

    # Run all inferences in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    times = []
    tokens_list = []
    response_lengths = []

    for result in results:
        if isinstance(result, tuple):
            elapsed_time, tokens, response = result
            times.append(elapsed_time)
            tokens_list.append(tokens)
            response_lengths.append(len(response))

    if times and tokens_list:
        # Calculate tokens per second for each run
        tokens_per_second = [
            t / time for t, time in zip(tokens_list, times) if time > 0
        ]

        return {
            "times": times,
            "tokens": tokens_list,
            "tokens_per_second": tokens_per_second,
            "mean_tps": mean(tokens_per_second) if tokens_per_second else 0,
            "median_tps": median(tokens_per_second) if tokens_per_second else 0,
            "min_tps": min(tokens_per_second) if tokens_per_second else 0,
            "max_tps": max(tokens_per_second) if tokens_per_second else 0,
            "avg_time": mean(times),
            "avg_tokens": mean(tokens_list),
            "avg_response_length": mean(response_lengths),
        }
    return None


async def benchmark_model_with_progress(
    model: str, prompt: str, runs: int, max_tokens: int, progress_queue: asyncio.Queue
) -> dict:
    """Benchmark a single model with progress updates"""

    # Create all inference tasks at once
    async def measure_and_report(model: str, prompt: str, max_tokens: int):
        result = await measure_inference(model, prompt, max_tokens)
        await progress_queue.put(1)  # Signal completion
        return result

    tasks = [measure_and_report(model, prompt, max_tokens) for _ in range(runs)]

    # Run all inferences in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results
    times = []
    tokens_list = []
    response_lengths = []

    for result in results:
        if isinstance(result, tuple):
            elapsed_time, tokens, response = result
            times.append(elapsed_time)
            tokens_list.append(tokens)
            response_lengths.append(len(response))

    if times and tokens_list:
        # Calculate tokens per second for each run
        tokens_per_second = [
            t / time for t, time in zip(tokens_list, times) if time > 0
        ]

        return {
            "times": times,
            "tokens": tokens_list,
            "tokens_per_second": tokens_per_second,
            "mean_tps": mean(tokens_per_second) if tokens_per_second else 0,
            "median_tps": median(tokens_per_second) if tokens_per_second else 0,
            "min_tps": min(tokens_per_second) if tokens_per_second else 0,
            "max_tps": max(tokens_per_second) if tokens_per_second else 0,
            "avg_time": mean(times),
            "avg_tokens": mean(tokens_list),
            "avg_response_length": mean(response_lengths),
        }
    return None


async def run_benchmarks(
    models: list[str], prompt: str, runs: int, max_tokens: int
) -> dict:
    """Run benchmarks for all models in parallel"""
    results = {}
    total_tasks = len(models) * runs

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            description="[bold cyan]Running Benchmark", total=total_tasks
        )

        # Create a queue for progress updates
        progress_queue = asyncio.Queue()

        async def progress_updater():
            """Update progress bar as tasks complete"""
            completed = 0
            while completed < total_tasks:
                await progress_queue.get()
                completed += 1
                progress.update(task, advance=1)

        # Start progress updater
        updater_task = asyncio.create_task(progress_updater())

        # Create coroutines for all models
        async def benchmark_and_report(model: str) -> tuple[str, dict]:
            """Benchmark a model and report when done"""
            result = await benchmark_model_with_progress(
                model, prompt, runs, max_tokens, progress_queue
            )
            return model, result

        # Run all benchmarks in parallel
        tasks = [benchmark_and_report(model) for model in models]
        benchmark_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Wait for progress updater to finish
        await updater_task

    # Show checkmark after benchmark completes
    console.print("[green]✓[/green] [bold cyan]Running Benchmark[/bold cyan]")

    # Combine results
    for item in benchmark_results:
        if isinstance(item, tuple):
            model, result = item
            if result:
                results[model] = result

    return results


@app.command()
def benchmark(
    models: list[str] = typer.Argument(
        ...,
        help="List of models to benchmark (e.g., gpt-4o gemini-2.5-flash)",
    ),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of runs per model"),
    prompt: Optional[str] = typer.Option(
        None, "--prompt", "-p", help="Custom prompt to use for benchmarking"
    ),
    lim: int = typer.Option(
        500, "--lim", "-l", help="Maximum tokens to generate per response"
    ),
):
    """Benchmark inference speed of different LLM models"""
    validation_results = asyncio.run(validate_models(models))
    valid_models = []
    invalid_models = []
    for model, (is_valid, message) in validation_results.items():
        if is_valid:
            valid_models.append(model)
        else:
            invalid_models.append((model, message))

    # Show validation results
    if invalid_models:
        console.print("\n[red]Failed validations:[/red]")
        for model, message in invalid_models:
            console.print(f"  [red]✗[/red] {model}: {message}")

    if not valid_models:
        console.print(
            "\n[red]No valid models to benchmark. Please check your API keys and model names.[/red]"
        )
        raise typer.Exit(1)

    if invalid_models:
        console.print(
            f"\n[yellow]Proceeding with {len(valid_models)} valid model(s)[/yellow]"
        )

    prompt_to_use = prompt or BENCHMARK_PROMPT

    # Run benchmarks
    results = asyncio.run(run_benchmarks(valid_models, prompt_to_use, runs, lim))

    # Display results
    if results:
        display_results(results)
    else:
        console.print("[red]No successful benchmarks completed[/red]")


def display_results(results: dict):
    """Display benchmark results in a formatted table"""

    console.print()  # Blank line before table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Mean tok/s", justify="right", style="bold green")
    table.add_column("Median tok/s", justify="right")
    table.add_column("Min tok/s", justify="right")
    table.add_column("Max tok/s", justify="right")
    table.add_column("Avg Time", justify="right")
    table.add_column("Avg Tokens", justify="right")

    # Sort by mean tokens per second (descending)
    sorted_models = sorted(
        results.keys(), key=lambda x: results[x]["mean_tps"], reverse=True
    )

    for model in sorted_models:
        data = results[model]
        table.add_row(
            model,
            f"{data['mean_tps']:.1f}",
            f"{data['median_tps']:.1f}",
            f"{data['min_tps']:.1f}",
            f"{data['max_tps']:.1f}",
            f"{data['avg_time']:.2f}s",
            f"{data['avg_tokens']:.0f}",
        )

    console.print(table)


@app.command(name="test-models")
def test_models(
    models: list[str] = typer.Argument(
        ..., help="List of models to test (e.g., gpt-3.5-turbo claude-3-haiku-20240307)"
    ),
):
    """Test if models are accessible and properly configured"""

    # Run async validation
    validation_results = asyncio.run(validate_models(models))

    # Display results in a table
    table = Table(
        title="Model Test Results", show_header=True, header_style="bold cyan"
    )
    table.add_column("Model", style="yellow")
    table.add_column("Status", style="green")
    table.add_column("Details")

    for model, (is_valid, message) in validation_results.items():
        status = "[green]✓ Available[/green]" if is_valid else "[red]✗ Failed[/red]"
        table.add_row(model, status, message)

    console.print(table)
    console.print(
        "\n[dim]Note: Set API keys as environment variables (e.g., OPENAI_API_KEY)[/dim]"
    )


def cli():
    import sys

    # Check if user provided models directly (not a subcommand)
    if (
        len(sys.argv) > 1
        and not sys.argv[1].startswith("-")
        and sys.argv[1] not in ["test-models", "benchmark", "--help"]
    ):
        # User provided models directly, insert 'benchmark' command
        sys.argv.insert(1, "benchmark")

    app()


if __name__ == "__main__":
    cli()
