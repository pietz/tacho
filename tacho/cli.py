import os
import asyncio

import typer
from rich.console import Console

from .config import load_env
from .display import run_pings, run_benchmarks, display_results

load_env()

app = typer.Typer()
console = Console()


def version_callback(value: bool):
    if value:
        from importlib.metadata import version

        console.print(f"tacho {version('tacho')}")
        raise typer.Exit()


async def run_cli(models: list[str], runs: int, tokens: int):
    """Run pings and benchmarks in a single event loop"""
    res = await run_pings(models)
    valid_models = [models[i] for i in range(len(models)) if res[i]]
    if not valid_models:
        return None, None
    bench_res = await run_benchmarks(valid_models, runs, tokens)
    return valid_models, bench_res


@app.command()
def cli(
    models: list[str] = typer.Argument(
        ...,
        help="List of models to benchmark using LiteLLM names",
    ),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of runs per model", min=1),
    tokens: int = typer.Option(
        500, "--tokens", "-t", help="Maximum tokens to generate per response", min=1
    ),
    version: bool | None = typer.Option(
        None, "--version", callback=version_callback, is_eager=True
    ),
):
    """CLI tool for measuring LLM inference speeds"""
    valid_models, res = asyncio.run(run_cli(models, runs, tokens))
    if valid_models is None:
        raise typer.Exit(1)
    display_results(valid_models, runs, res)


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
