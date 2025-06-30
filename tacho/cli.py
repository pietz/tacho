import os
import asyncio

import typer
from rich.console import Console

from .config import load_env
from .display import run_pings, run_benchmarks, display_results

load_env()

app = typer.Typer(help="CLI tool for measuring LLM inference speeds")
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
    tokens: int = typer.Option(500, "--tokens", "-t"),
    version: bool | None = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
):
    """Default command when models are provided directly"""
    if ctx.invoked_subcommand is None and models:
        bench(models, runs, tokens)


@app.command()
def bench(
    models: list[str] = typer.Argument(
        ...,
        help="List of models to benchmark using LiteLLM names",
    ),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of runs per model"),
    tokens: int = typer.Option(
        500, "--tokens", "-t", help="Maximum tokens to generate per response"
    ),
):
    """Benchmark inference speed of different LLM models"""
    res = asyncio.run(run_pings(models))
    valid_models = [models[i] for i in range(len(models)) if res[i]]
    if not valid_models:
        raise typer.Exit(1)
    res = asyncio.run(run_benchmarks(valid_models, runs, tokens))
    display_results(valid_models, runs, res)


@app.command()
def ping(
    models: list[str] = typer.Argument(
        ...,
        help="List of models to check availability (e.g., gpt-4o gemini-2.5-flash)",
    ),
):
    """Check which LLM models are accessible without running benchmarks"""
    asyncio.run(run_pings(models))


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
