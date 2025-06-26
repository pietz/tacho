import time
from typing import Optional
from statistics import mean, median

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import litellm

__version__ = "0.1.2"

app = typer.Typer(help="CLI tool for measuring and comparing LLM inference speeds")
console = Console()

DEFAULT_PROMPT = """Generate a ~1000 word summary of the history of the USA."""


def measure_inference_time(model: str, prompt: str) -> tuple[float, str]:
    """Measure inference time for a single run"""
    start_time = time.time()
    
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    return elapsed_time, response.choices[0].message.content


@app.command()
def benchmark(
    models: list[str] = typer.Argument(
        ..., 
        help="List of models to benchmark (e.g., gpt-3.5-turbo claude-3-haiku-20240307)"
    ),
    runs: int = typer.Option(
        5, 
        "--runs", "-r",
        help="Number of runs per model"
    ),
    prompt: Optional[str] = typer.Option(
        None,
        "--prompt", "-p",
        help="Custom prompt to use for benchmarking"
    ),
):
    """Benchmark inference speed of different LLM models"""
    
    prompt_to_use = prompt or DEFAULT_PROMPT
    
    console.print(f"[bold cyan]Benchmarking {len(models)} models with {runs} runs each[/bold cyan]")
    console.print(f"[dim]Prompt length: {len(prompt_to_use)} characters[/dim]\n")
    
    results = {}
    
    for model in models:
        console.print(f"[yellow]Testing {model}...[/yellow]")
        times = []
        response_lengths = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Running {runs} inferences...", total=runs)
            
            for i in range(runs):
                try:
                    elapsed_time, response = measure_inference_time(model, prompt_to_use)
                    times.append(elapsed_time)
                    response_lengths.append(len(response))
                    progress.update(task, advance=1)
                except Exception as e:
                    console.print(f"[red]Error with {model}: {str(e)}[/red]")
                    break
        
        if times:
            results[model] = {
                "times": times,
                "mean": mean(times),
                "median": median(times),
                "min": min(times),
                "max": max(times),
                "avg_response_length": mean(response_lengths)
            }
        
        console.print()
    
    # Display results
    if results:
        display_results(results)
    else:
        console.print("[red]No successful benchmarks completed[/red]")


def display_results(results: dict):
    """Display benchmark results in a formatted table"""
    
    table = Table(title="Benchmark Results", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Mean (s)", justify="right")
    table.add_column("Median (s)", justify="right")
    table.add_column("Min (s)", justify="right")
    table.add_column("Max (s)", justify="right")
    table.add_column("Avg Response Length", justify="right")
    
    # Sort by mean time
    sorted_models = sorted(results.keys(), key=lambda x: results[x]["mean"])
    
    for model in sorted_models:
        data = results[model]
        table.add_row(
            model,
            f"{data['mean']:.3f}",
            f"{data['median']:.3f}",
            f"{data['min']:.3f}",
            f"{data['max']:.3f}",
            f"{int(data['avg_response_length'])}"
        )
    
    console.print(table)
    
    # Show winner
    fastest = sorted_models[0]
    console.print(f"\n[bold green]🏆 Fastest model: {fastest} "
                  f"(mean: {results[fastest]['mean']:.3f}s)[/bold green]")


@app.command(name="list-providers")
def list_providers():
    """List available LLM providers and example model names"""
    
    providers = {
        "OpenAI": ["gpt-4", "gpt-3.5-turbo"],
        "Anthropic": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "Google": ["gemini-pro"],
        "Cohere": ["command", "command-light"],
        "Together AI": ["mistralai/Mixtral-8x7B-Instruct-v0.1"],
    }
    
    table = Table(title="Available Providers", show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="yellow")
    table.add_column("Example Models", style="green")
    
    for provider, models in providers.items():
        table.add_row(provider, ", ".join(models))
    
    console.print(table)
    console.print("\n[dim]Note: Set API keys as environment variables (e.g., OPENAI_API_KEY)[/dim]")


def cli():
    import sys
    
    # Check if user provided models directly (not a subcommand)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('-') and sys.argv[1] not in ['list-providers', 'benchmark', '--help']:
        # User provided models directly, insert 'benchmark' command
        sys.argv.insert(1, 'benchmark')
    
    app()


if __name__ == "__main__":
    cli()