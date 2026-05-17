"""
cli.py — Command line interface for the YojanaGPT scraper.

Commands:
  scrape   Run the scraper in test or full mode

Usage:
  # Scrape 20 schemes for testing
  python -m src.scraper.cli scrape --test

  # Scrape all 4676 schemes
  python -m src.scraper.cli scrape --full

  # Scrape with custom limit
  python -m src.scraper.cli scrape --test --limit 50
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

# Load .env file before anything else
load_dotenv()

from .scraper import YojanaGPTScraper

# ── Setup ─────────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="yojanagpt-scraper",
    help="YojanaGPT — scraper for myscheme.gov.in government schemes",
    add_completion=False,
)

console = Console()


def _setup_logging(log_level: str = "INFO") -> None:
    """
    Configure logging with rich formatting for terminal output.
    Also writes to a log file in the logs/ directory.

    Args:
        log_level: Logging level string — DEBUG, INFO, WARNING, ERROR
    """
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)

    level = getattr(logging, log_level.upper(), logging.INFO)

    # Rich handler for pretty terminal output
    rich_handler = RichHandler(
        console=console,
        rich_tracebacks=True,
        show_time=True,
        show_path=False,
    )
    rich_handler.setLevel(level)

    # File handler for persistent log file
    file_handler = logging.FileHandler("logs/scraper.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[rich_handler, file_handler],
        force=True,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

@app.command()
def scrape(
    test: bool = typer.Option(
        False,
        "--test",
        help="Run in test mode — scrape only a small number of schemes",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run in full mode — scrape all 4676 schemes (takes ~90 mins)",
    ),
    limit: int = typer.Option(
        20,
        "--limit",
        help="Number of schemes to scrape in test mode (default: 20)",
    ),
    output_dir: str = typer.Option(
        "data/raw",
        "--output",
        help="Directory to save scraped data",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG, INFO, WARNING, ERROR",
    ),
) -> None:
    """
    Scrape government scheme data from myscheme.gov.in.

    Run in test mode first to verify everything works,
    then run full mode to collect all 4676 schemes.

    Examples:

      python -m src.scraper.cli scrape --test

      python -m src.scraper.cli scrape --test --limit 50

      python -m src.scraper.cli scrape --full
    """
    # Must specify either --test or --full
    if not test and not full:
        console.print(
            "[bold red]Error:[/bold red] Specify either --test or --full\n"
            "  Example: python -m src.scraper.cli scrape --test"
        )
        raise typer.Exit(code=1)

    if test and full:
        console.print(
            "[bold red]Error:[/bold red] Cannot use --test and --full together"
        )
        raise typer.Exit(code=1)

    # Setup logging
    _setup_logging(log_level)
    logger = logging.getLogger(__name__)

    # Print startup banner
    mode_label = f"TEST (limit={limit})" if test else "FULL (all 4676 schemes)"
    console.rule("[bold blue]YojanaGPT Scraper[/bold blue]")
    console.print(f"  Mode     : [bold]{mode_label}[/bold]")
    console.print(f"  Output   : [bold]{output_dir}[/bold]")
    console.print(f"  Log level: [bold]{log_level}[/bold]")
    console.rule()

    # Confirm before running full mode
    if full:
        console.print(
            "\n[yellow]Full mode will scrape all 4676 schemes.[/yellow]"
            "\nThis takes approximately 90 minutes."
            "\nProgress is saved every 100 schemes so you can resume if interrupted.\n"
        )
        confirmed = typer.confirm("Continue?")
        if not confirmed:
            console.print("Aborted.")
            raise typer.Exit(code=0)

    # Run the scraper
    try:
        scraper = YojanaGPTScraper(
            test_mode=test,
            test_limit=limit,
            raw_dir=Path(output_dir),
        )
        scraper.run()

        console.rule("[bold green]Scrape Complete[/bold green]")
        console.print(f"  Data saved to: [bold]{output_dir}[/bold]")
        console.print(f"  Log file     : [bold]logs/scraper.log[/bold]")
        console.rule()

    except KeyboardInterrupt:
        console.print("\n[yellow]Scrape interrupted by user.[/yellow]")
        console.print("Progress has been saved. Run the same command to resume.")
        raise typer.Exit(code=0)

    except Exception as e:
        logger.exception("Scraper failed with unexpected error")
        console.print(f"\n[bold red]Scraper failed:[/bold red] {str(e)}")
        console.print("Check [bold]logs/scraper.log[/bold] for full details.")
        raise typer.Exit(code=1)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()