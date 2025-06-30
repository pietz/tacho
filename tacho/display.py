import asyncio
from statistics import mean

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .ai import ping_model, bench_model

console = Console()


async def run_pings(models: list[str]):
    """Run ping checks with progress indicator"""
    spinner = SpinnerColumn()
    text = TextColumn("[progress.description]{task.description}")
    with Progress(spinner, text, transient=True, console=console) as prog:
        prog.add_task("[bold cyan]Checking Model Access...[bold cyan]", total=None)
        return await asyncio.gather(
            *[ping_model(m, console=prog.console) for m in models]
        )


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


def calculate_metrics(stats: list) -> dict:
    """Calculate performance metrics from benchmark results"""
    tps = [t / s for s, t in stats if s > 0]
    return [
        mean(tps),
        min(tps),
        max(tps),
        mean([x[0] for x in stats]),
        mean([x[1] for x in stats]),
    ]


def display_results(models: list[str], runs: int, results: list):
    """Process and display benchmark results in a formatted table"""
    metrics = []
    for i, model in enumerate(models):
        model_results = results[i * runs : (i + 1) * runs]
        metrics.append([model] + calculate_metrics(model_results))
    metrics = sorted(metrics, key=lambda x: x[1], reverse=True)

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Avg t/s", justify="right", style="bold green")
    table.add_column("Min t/s", justify="right")
    table.add_column("Max t/s", justify="right")
    table.add_column("Time", justify="right")
    table.add_column("Tokens", justify="right")

    for m in metrics:
        table.add_row(
            m[0],
            f"{m[1]:.1f}",
            f"{m[2]:.1f}",
            f"{m[3]:.1f}",
            f"{m[4]:.1f}s",
            f"{m[5]:.0f}",
        )

    console.print(table)
