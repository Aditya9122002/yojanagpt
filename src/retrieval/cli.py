"""
cli.py — Command line interface for testing the RAG pipeline.

Usage:
  python -m src.retrieval.cli "PM Kisan ke liye kaun eligible hai?"
  python -m src.retrieval.cli "What documents do I need for NOS-SWD scholarship?"
  python -m src.retrieval.cli "PM கிசான் திட்டம் என்ன?"
"""

from __future__ import annotations

import logging
import os

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

load_dotenv()

from .pipeline import YojanaRAGPipeline
from src.translation.detector import get_language_name

app = typer.Typer(
    name="yojanagpt-retrieval",
    help="YojanaGPT — test the RAG pipeline",
    add_completion=False,
)

console = Console()


def _setup_logging(log_level: str = "WARNING") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.WARNING),
        format="%(levelname)s | %(name)s | %(message)s",
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask YojanaGPT"),
    top_k: int = typer.Option(5, "--top-k", help="Number of chunks to retrieve"),
    chroma_dir: str = typer.Option("data/chromadb", "--chroma-dir", help="ChromaDB directory"),
    show_chunks: bool = typer.Option(False, "--show-chunks", help="Show retrieved chunks"),
    no_translate: bool = typer.Option(False, "--no-translate", help="Disable translation"),
    log_level: str = typer.Option("WARNING", "--log-level", help="Logging level"),
) -> None:
    """Ask YojanaGPT a question about government schemes."""
    _setup_logging(log_level)

    console.rule("[bold blue]YojanaGPT[/bold blue]")
    console.print(f"\n[bold]Question:[/bold] {question}\n")

    with console.status("Loading pipeline..."):
        try:
            pipeline = YojanaRAGPipeline(
                chroma_dir=chroma_dir,
                top_k=top_k,
                enable_translation=not no_translate,
            )
        except Exception as e:
            console.print(f"[bold red]Failed to initialise pipeline:[/bold red] {e}")
            raise typer.Exit(code=1)

    with console.status("Searching schemes and generating answer..."):
        try:
            response = pipeline.ask(question, top_k=top_k)
        except Exception as e:
            console.print(f"[bold red]Pipeline error:[/bold red] {e}")
            raise typer.Exit(code=1)

    # Show detected language if not English
    if response.detected_language != "en":
        lang_name = get_language_name(response.detected_language)
        console.print(
            f"[dim]Detected language: {lang_name} | "
            f"English query: {response.english_question[:60]}[/dim]\n"
        )

    console.rule("[bold green]Answer[/bold green]")
    console.print(Markdown(response.answer))

    if response.sources:
        console.rule("[bold]Sources[/bold]")
        for src in response.sources:
            console.print(f"  • [bold]{src['scheme_name']}[/bold]")
            console.print(f"    [dim]{src['source_url']}[/dim]")

    if show_chunks and response.chunks_used:
        console.rule("[bold]Retrieved Chunks[/bold]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("Scheme", style="cyan", max_width=30)
        table.add_column("Type", style="green", max_width=15)
        table.add_column("Distance", style="yellow", max_width=10)
        table.add_column("Text preview", max_width=60)
        for chunk in response.chunks_used:
            table.add_row(
                chunk.scheme_name[:30],
                chunk.chunk_type,
                f"{chunk.distance:.4f}",
                chunk.text[:100] + "...",
            )
        console.print(table)

    console.rule()


if __name__ == "__main__":
    app()