import asyncio
import os
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .core import BENCHMARK_PROMPT, ping_models, bench_models

app = typer.Typer(help="CLI tool for measuring LLM inference speeds")
console = Console()


@app.callback(invoke_without_command=True)
def cli_main(
    ctx: typer.Context,
    models: Optional[list[str]] = typer.Argument(None),
    runs: int = typer.Option(5, "--runs", "-r"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p"),
    lim: int = typer.Option(500, "--lim", "-l"),
):
    """Default command when models are provided directly"""
    if ctx.invoked_subcommand is None and models:
        bench(models, runs, prompt, lim)


@app.command()
def bench(
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
    res = asyncio.run(ping_models(models))
    valid_models = [models[i] for i in range(len(models)) if res[i]]

    if not valid_models:
        raise typer.Exit(1)

    prompt_to_use = prompt or BENCHMARK_PROMPT
    results = asyncio.run(bench_models(valid_models, prompt_to_use, runs, lim))

    if not results:
        raise typer.Exit(1)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Avg tok/s", justify="right", style="bold green")
    # table.add_column("Median tok/s", justify="right")
    table.add_column("Min tok/s", justify="right")
    table.add_column("Max tok/s", justify="right")
    table.add_column("Avg Time", justify="right")
    # table.add_column("Avg Tokens", justify="right")
    
    # Sort by mean tokens per second (descending)
    sorted_models = sorted(
        results.keys(), key=lambda x: results[x]["mean_tps"], reverse=True
    )
    
    for model in sorted_models:
        data = results[model]
        table.add_row(
            model,
            f"{data['mean_tps']:.1f}",
            #f"{data['median_tps']:.1f}",
            f"{data['min_tps']:.1f}",
            f"{data['max_tps']:.1f}",
            f"{data['avg_time']:.1f}s",
            #f"{data['avg_tokens']:.0f}",
        )
    
    console.print(table)


def main():
    """Main entry point that suppresses warnings on exit."""
    os.environ["PYTHONWARNINGS"] = "ignore"
    try:
        result = app(standalone_mode=False)
    except SystemExit as e:
        result = e.code
    except (KeyboardInterrupt, EOFError):
        # Handle common user interruptions gracefully
        result = 1
    except Exception:
        # Catch any other unexpected exceptions to ensure clean exit
        # This is intentionally broad as it's the last resort handler
        result = 1
    os._exit(result or 0)


if __name__ == "__main__":
    main()
