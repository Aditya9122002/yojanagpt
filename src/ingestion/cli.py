"""
cli.py — Command line interface for the YojanaGPT ingestion pipeline.

Commands:
  ingest   Run the ingestion pipeline (test or full mode)
  info     Show ChromaDB collection statistics
  reset    Delete the ChromaDB collection and start fresh

Usage:
  python -m src.ingestion.cli ingest --test
  python -m src.ingestion.cli ingest --full
  python -m src.ingestion.cli info
  python -m src.ingestion.cli reset
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

load_dotenv()

from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.vectorstore import VectorStore

# ── Setup ─────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="yojanagpt-ingestion",
    help="YojanaGPT — ingestion pipeline for scheme embeddings",
    add_completion=False,
)

console = Console()


def _setup_logging(log_level: str = "INFO") -> None:
    """Configure logging with rich formatting."""
    Path("logs").mkdir(exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    from rich.logging import RichHandler
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_path=False),
            logging.FileHandler("logs/ingestion.log", encoding="utf-8"),
        ],
        force=True,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    test: bool = typer.Option(
        False,
        "--test",
        help="Ingest only 20 schemes for testing",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Ingest all schemes from the JSONL file",
    ),
    input_file: str = typer.Option(
        "data/raw/schemes_detail.jsonl",
        "--input",
        help="Path to the scraped JSONL file",
    ),
    chroma_dir: str = typer.Option(
        "data/chromadb",
        "--chroma-dir",
        help="Directory to store ChromaDB data",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        help="Schemes to process per batch",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    ),
) -> None:
    """
    Run the ingestion pipeline — chunk, embed, and store schemes in ChromaDB.

    Run test mode first to verify everything works,
    then run full mode to ingest all schemes.

    Examples:

      python -m src.ingestion.cli ingest --test

      python -m src.ingestion.cli ingest --full
    """
    if not test and not full:
        console.print(
            "[bold red]Error:[/bold red] Specify either --test or --full\n"
            "  Example: python -m src.ingestion.cli ingest --test"
        )
        raise typer.Exit(code=1)

    if test and full:
        console.print(
            "[bold red]Error:[/bold red] Cannot use --test and --full together"
        )
        raise typer.Exit(code=1)

    _setup_logging(log_level)

    limit = 20 if test else None
    mode_label = f"TEST (limit=20)" if test else "FULL"

    console.rule("[bold blue]YojanaGPT Ingestion Pipeline[/bold blue]")
    console.print(f"  Mode      : [bold]{mode_label}[/bold]")
    console.print(f"  Input     : [bold]{input_file}[/bold]")
    console.print(f"  ChromaDB  : [bold]{chroma_dir}[/bold]")
    console.print(f"  Batch size: [bold]{batch_size}[/bold]")
    console.rule()

    # Check input file exists before starting
    if not Path(input_file).exists():
        console.print(
            f"\n[bold red]Error:[/bold red] Input file not found: {input_file}\n"
            f"Run the scraper first:\n"
            f"  python -m src.scraper.cli scrape --test"
        )
        raise typer.Exit(code=1)

    try:
        pipeline = IngestionPipeline(
            chroma_dir=chroma_dir,
            batch_size=batch_size,
        )
        summary = pipeline.run(input_file=input_file, limit=limit)

        # Print results table
        console.rule("[bold green]Ingestion Complete[/bold green]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value", style="green")

        table.add_row("Schemes processed", str(summary["schemes_processed"]))
        table.add_row("Schemes failed",    str(summary["schemes_failed"]))
        table.add_row("Chunks stored",     str(summary["chunks_stored"]))
        table.add_row("Total in ChromaDB", str(summary["total_chunks_in_store"]))
        table.add_row("Duration",          f"{summary['duration_seconds']}s")

        console.print(table)
        console.rule()

    except KeyboardInterrupt:
        console.print("\n[yellow]Ingestion interrupted.[/yellow]")
        console.print("ChromaDB data is safe — partial progress is preserved.")
        raise typer.Exit(code=0)

    except Exception as e:
        console.print(f"\n[bold red]Ingestion failed:[/bold red] {str(e)}")
        console.print("Check [bold]logs/ingestion.log[/bold] for details.")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """
    Show statistics about the ChromaDB collection.

    Displays how many chunks are stored and where ChromaDB data lives.
    """
    _setup_logging()

    try:
        store = VectorStore()
        info = store.get_collection_info()

        console.rule("[bold blue]ChromaDB Collection Info[/bold blue]")
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Key", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Collection", info["collection_name"])
        table.add_row("Total chunks", str(info["total_chunks"]))
        table.add_row("Storage path", info["persist_dir"])

        console.print(table)
        console.rule()

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def reset() -> None:
    """
    Delete the ChromaDB collection and all stored embeddings.

    Use this when you want to re-ingest from scratch.
    This cannot be undone.
    """
    _setup_logging()

    console.print(
        "\n[bold yellow]Warning:[/bold yellow] This will delete all stored "
        "embeddings in ChromaDB.\nThis cannot be undone.\n"
    )

    confirmed = typer.confirm("Are you sure?")
    if not confirmed:
        console.print("Aborted.")
        raise typer.Exit(code=0)

    try:
        store = VectorStore()
        store.delete_collection()
        console.print("[green]Collection deleted successfully.[/green]")
        console.print("Run [bold]ingest[/bold] to re-populate.")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()