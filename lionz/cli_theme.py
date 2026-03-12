# lionz/cli_theme.py
"""DigiTx-inspired terminal theme for LION CLI.

Coral & greige palette:
  - cfonts block banner with coral-to-greige gradient
  - Numbered section headers ("01 · SECTION NAME")
  - Status lines with Unicode indicators
  - Coral progress bars and spinners
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from rich import box
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.table import Table
from rich.text import Text

# ── Brand ─────────────────────────────────────────────────────────

BRAND = "L I O N"
TAGLINE = "The New Standard in PET Lesion Segmentation"
ORG = "ENHANCE.PET · QIMP · Medical University of Vienna"

# ── Palette ───────────────────────────────────────────────────────

CORAL = "#E87461"
GREIGE = "#B5A89A"
MUTED = "dim"


# ── Banner ────────────────────────────────────────────────────────


def print_banner(version: str, console: Console) -> None:
    """Print the LION banner using cfonts with coral-to-greige gradient."""
    try:
        from cfonts import render

        output = render(
            "LION",
            font="block",
            gradient=[CORAL, GREIGE],
            transition=True,
            space=False,
        )
        indented = "\n".join(" " + line for line in output.split("\n"))
        console.file.write("\n" + indented)
    except ImportError:
        console.print(f"\n  [bold {CORAL}]{BRAND}[/bold {CORAL}]\n")

    console.print()
    console.print(f"  [{GREIGE}]{TAGLINE}[/{GREIGE}]")
    console.print(f"  [{MUTED}]v{version} · {ORG}[/{MUTED}]")
    console.print()


def print_version(version: str, console: Console) -> None:
    """Print a compact branded version line."""
    t = Text()
    t.append(BRAND, style=f"bold {CORAL}")
    t.append(f"  v{version}", style=MUTED)
    console.print(t)


# ── Section headers ──────────────────────────────────────────────


def section(
    title: str,
    console: Console,
    number: str | None = None,
) -> None:
    """Print a numbered section header."""
    console.print()
    t = Text()
    if number:
        t.append(f"  {number}", style=f"bold {CORAL}")
        t.append(" · ", style=MUTED)
    else:
        t.append("  ", style="")
    t.append(title.upper(), style="bold")
    console.print(t)
    rule = "─" * len(TAGLINE)
    console.print(f"  {rule}", style=GREIGE)


# ── Tables ───────────────────────────────────────────────────────


def make_table(title: str | None = None, **kwargs: object) -> Table:
    """Create a table with rounded greige border."""
    return Table(
        title=title,
        box=box.ROUNDED,
        border_style=GREIGE,
        title_style=f"bold {CORAL}",
        header_style="bold",
        padding=(0, 1),
        **kwargs,
    )


def make_kv_table() -> Table:
    """Create a headerless two-column key–value table."""
    t = Table(
        box=box.ROUNDED,
        border_style=GREIGE,
        show_header=False,
        show_edge=False,
        padding=(0, 1, 0, 2),
    )
    t.add_column("Key", style=f"bold {CORAL}", no_wrap=True)
    t.add_column("Value")
    return t


# ── Status lines ─────────────────────────────────────────────────


def info(msg: str) -> str:
    """Info-level status line (coral arrow, dim text)."""
    return f"  [{CORAL}]›[/{CORAL}] [{MUTED}]{msg}[/{MUTED}]"


def ok(msg: str) -> str:
    """Success status line (green check)."""
    return f"  [bold green]✓[/bold green] {msg}"


def warn(msg: str) -> str:
    """Warning status line (yellow bang)."""
    return f"  [bold yellow]![/bold yellow] [yellow]{msg}[/yellow]"


def err(msg: str) -> str:
    """Error status line (red cross)."""
    return f"  [bold red]✗[/bold red] {msg}"


def kv(key: str, value: str) -> str:
    """Key-value line."""
    return f"  [{CORAL}]{key}[/{CORAL}] [{MUTED}]{value}[/{MUTED}]"


# ── Progress helpers ────────────────────────────────────────────


@contextmanager
def spinner(label: str, console: Console) -> Generator[None, None, None]:
    """Coral dots spinner for indeterminate operations."""
    p = Progress(
        TextColumn(" "),
        SpinnerColumn("dots", style=Style(color=CORAL)),
        TextColumn(f"[{MUTED}]{label}[/{MUTED}]"),
        console=console,
        transient=True,
    )
    with p:
        p.add_task(label, total=None)
        yield


@contextmanager
def progress_bar(total: int, label: str, console: Console) -> Generator[object, None, None]:
    """Coral progress bar for countable operations."""
    p = Progress(
        TextColumn(f"  [{CORAL}]▸[/{CORAL}]"),
        BarColumn(complete_style=Style(color=CORAL), finished_style=Style(color=CORAL)),
        MofNCompleteColumn(),
        TextColumn(f"[{MUTED}]{label}[/{MUTED}]"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
    with p:
        task = p.add_task(label, total=total)
        yield lambda: p.advance(task)
