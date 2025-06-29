import asyncio
from statistics import mean, median

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from collections import defaultdict

from .ai import ping_model, bench_model

console = Console()


async def run_pings(models: list[str]):
    """Run ping checks with progress indicator"""
    spinner = SpinnerColumn()
    text = TextColumn("[progress.description]{task.description}")
    with Progress(spinner, text, transient=True) as prog:
        prog.add_task("[bold cyan]Checking Model Access...[bold cyan]", total=None)
        return await asyncio.gather(*[ping_model(m) for m in models])


async def run_benchmarks(models: list[str], runs: int, tokens: int):
    """Run benchmarks with progress indicator"""
    spinner = SpinnerColumn()
    text = TextColumn("[progress.description]{task.description}")
    with Progress(spinner, text, transient=True) as prog:
        prog.add_task("[bold cyan]Running Benchmark...[/bold cyan]", total=None)
        tasks = []
        for m in models:
            for _ in range(runs):
                tasks.append(bench_model(m, tokens))
        return await asyncio.gather(*tasks)


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


def display_results(models: list[str], runs: int, results: list):
    """Process and display benchmark results in a formatted table"""
    # Process raw results into metrics by model
    model_results = defaultdict(list)

    # Map results back to models based on known ordering
    idx = 0
    for model in models:
        for _ in range(runs):
            if (
                idx < len(results)
                and isinstance(results[idx], tuple)
                and len(results[idx]) == 2
            ):
                model_results[model].append(results[idx])
            idx += 1

    # Calculate metrics for each model
    processed_results = {
        model: calculate_metrics(
            [time for time, _ in data], [tokens for _, tokens in data]
        )
        if data
        else {}
        for model, data in model_results.items()
    }

    if not processed_results:
        return

    # Display the table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Avg tok/s", justify="right", style="bold green")
    table.add_column("Min tok/s", justify="right")
    table.add_column("Max tok/s", justify="right")
    table.add_column("Avg Time", justify="right")

    # Sort by mean tokens per second (descending)
    sorted_models = sorted(
        processed_results.keys(),
        key=lambda x: processed_results[x]["mean_tps"],
        reverse=True,
    )

    for model in sorted_models:
        data = processed_results[model]
        table.add_row(
            model,
            f"{data['mean_tps']:.1f}",
            f"{data['min_tps']:.1f}",
            f"{data['max_tps']:.1f}",
            f"{data['avg_time']:.1f}s",
        )

    console.print(table)
