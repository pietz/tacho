import asyncio
import time
import logging
from typing import Optional
from statistics import mean, median
from importlib.metadata import version

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import litellm

# Suppress litellm debug output and logging
litellm.suppress_debug_info = True
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.CRITICAL)
logging.getLogger("litellm").setLevel(logging.CRITICAL)

# Suppress aiohttp unclosed session warnings
import warnings
warnings.filterwarnings("ignore", message="Unclosed client session")
warnings.filterwarnings("ignore", message="Unclosed connector")

try:
    __version__ = version("tacho")
except Exception:
    __version__ = "dev"

app = typer.Typer(help="CLI tool for measuring and comparing LLM inference speeds")
console = Console()

BENCHMARK_PROMPT = """Generate a ~2000 word summary of the history of the USA."""
VALIDATION_PROMPT = "Do you have time to help? (yes/no)"


def calculate_metrics(times: list[float], tokens_list: list[int]) -> dict:
    """Calculate performance metrics from benchmark results"""
    if not times or not tokens_list:
        return {}
    
    tokens_per_second = [t / time for t, time in zip(tokens_list, times) if time > 0]
    
    return {
        "mean_tps": mean(tokens_per_second) if tokens_per_second else 0,
        "median_tps": median(tokens_per_second) if tokens_per_second else 0,
        "min_tps": min(tokens_per_second) if tokens_per_second else 0,
        "max_tps": max(tokens_per_second) if tokens_per_second else 0,
        "avg_time": mean(times),
        "avg_tokens": mean(tokens_list),
    }


async def validate_models(models: list[str]) -> dict[str, tuple[bool, str]]:
    """Validate all models before benchmarking"""
    
    async def validate_model(model: str) -> tuple[str, tuple[bool, str]]:
        """Validate a single model"""
        try:
            await litellm.acompletion(
                model=model,
                messages=[{"role": "user", "content": VALIDATION_PROMPT}],
                max_tokens=3,
            )
            return model, (True, "OK")
        except Exception as e:
            error_msg = str(e).lower()
            error_types = {
                "authentication": "Authentication failed",
                "api key": "Authentication failed",
                "not found": "Model not found",
            }
            
            for key, message in error_types.items():
                if key in error_msg:
                    return model, (False, message)
            
            return model, (False, f"Error: {str(e)[:50]}...")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description="[bold cyan]Validating Models...", total=None)
        
        # Run all validations in parallel
        results = await asyncio.gather(*[validate_model(model) for model in models])

    # Show checkmark after validation completes
    console.print("[green]✓[/green] [bold cyan]Validating Models[/bold cyan]")
    
    return dict(results)


async def measure_inference(
    model: str, prompt: str, max_tokens: int
) -> tuple[float, int]:
    """Measure inference time for a single run and return time and tokens"""
    start_time = time.time()
    
    response = await litellm.acompletion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    
    elapsed_time = time.time() - start_time
    completion_tokens = response.usage.completion_tokens if response.usage else 0
    
    return elapsed_time, completion_tokens


async def benchmark_model(
    model: str, prompt: str, runs: int, max_tokens: int, update_progress=None
) -> dict:
    """Benchmark a single model"""
    
    async def measure_and_update(model: str, prompt: str, max_tokens: int):
        result = await measure_inference(model, prompt, max_tokens)
        if update_progress:
            update_progress()
        return result
    
    # Create measurement function based on whether we need progress updates
    measure_fn = measure_and_update if update_progress else measure_inference
    
    # Run all inferences in parallel
    tasks = [measure_fn(model, prompt, max_tokens) for _ in range(runs)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    times = []
    tokens_list = []
    
    for result in results:
        if isinstance(result, tuple):
            elapsed_time, tokens = result
            times.append(elapsed_time)
            tokens_list.append(tokens)
    
    # Calculate and return metrics
    metrics = calculate_metrics(times, tokens_list)
    if metrics:
        metrics.update({
            "times": times,
            "tokens": tokens_list,
        })
    
    return metrics


async def run_benchmarks(
    models: list[str], prompt: str, runs: int, max_tokens: int
) -> dict:
    """Run benchmarks for all models in parallel"""
    results = {}
    total_tasks = len(models) * runs
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            description=f"[bold cyan]Benchmarking {len(models)} models with {max_tokens} tokens on {runs} runs", 
            total=total_tasks
        )
        
        # Create update function for progress
        def update_progress():
            progress.advance(task)
        
        # Run all benchmarks in parallel
        benchmark_tasks = []
        for model in models:
            coro = benchmark_model(model, prompt, runs, max_tokens, update_progress)
            benchmark_tasks.append((model, coro))
        
        # Execute all benchmarks
        benchmark_results = await asyncio.gather(
            *[coro for _, coro in benchmark_tasks],
            return_exceptions=True
        )
    
    # Show checkmark after benchmark completes
    console.print(f"[green]✓[/green] [bold cyan]Benchmarking {len(models)} models with {max_tokens} tokens on {runs} runs[/bold cyan]")
    
    # Combine results
    for (model, _), result in zip(benchmark_tasks, benchmark_results):
        if isinstance(result, dict) and result:
            results[model] = result
    
    return results


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    models: Optional[list[str]] = typer.Argument(None),
    runs: int = typer.Option(5, "--runs", "-r"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    lim: int = typer.Option(500, "--lim", "-l"),
):
    """Default command when models are provided directly"""
    if ctx.invoked_subcommand is None and models:
        # User provided models directly, run benchmark
        benchmark(models, runs, prompt, lim)


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

if __name__ == "__main__":
    app()