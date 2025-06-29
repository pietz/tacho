import os
import asyncio

import typer
from rich.console import Console
from rich.table import Table

from .config import load_env
from .core import ping_models, bench_models

load_env()

app = typer.Typer(
    help="CLI tool for measuring LLM inference speeds",
)
console = Console()


def version_callback(value: bool):
    if value:
        from importlib.metadata import version

        console.print(f"tacho {version('tacho')}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def cli_main(
    ctx: typer.Context,
    models: list[str] | None = typer.Argument(None),
    runs: int = typer.Option(5, "--runs", "-r"),
    lim: int = typer.Option(500, "--lim", "-l"),
    version: bool | None = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
):
    """Default command when models are provided directly"""
    if ctx.invoked_subcommand is None and models:
        bench(models, runs, lim)


@app.command()
def bench(
    models: list[str] = typer.Argument(
        ...,
        help="List of models to benchmark using LiteLLM names",
    ),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of runs per model"),
    lim: int = typer.Option(
        500, "--lim", "-l", help="Maximum tokens to generate per response"
    ),
):
    """Benchmark inference speed of different LLM models"""
    res = asyncio.run(ping_models(models))
    valid_models = [models[i] for i in range(len(models)) if res[i]]

    if not valid_models:
        raise typer.Exit(1)

    results = asyncio.run(bench_models(valid_models, runs, lim))

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
            # f"{data['median_tps']:.1f}",
            f"{data['min_tps']:.1f}",
            f"{data['max_tps']:.1f}",
            f"{data['avg_time']:.1f}s",
            # f"{data['avg_tokens']:.0f}",
        )

    console.print(table)


@app.command()
def ping(
    models: list[str] = typer.Argument(
        ...,
        help="List of models to check availability (e.g., gpt-4o gemini-2.5-flash)",
    ),
):
    """Check which LLM models are accessible without running benchmarks"""
    res = asyncio.run(ping_models(models))

    # Count successful models
    successful = sum(res)

    # Print summary
    console.print()
    if successful == len(models):
        console.print(
            f"[bold green]All {len(models)} models are accessible![/bold green]"
        )
    elif successful > 0:
        console.print(
            f"[bold yellow]{successful}/{len(models)} models are accessible[/bold yellow]"
        )
    else:
        console.print("[bold red]No models are accessible[/bold red]")

    # Exit with appropriate code
    if successful == 0:
        raise typer.Exit(1)


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
