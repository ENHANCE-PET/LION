#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LION System Utilities
---------------------

Console output, logging, and device detection.
Uses cli_theme for visual styling.
"""

import importlib.metadata
import logging
import os
import sys
from datetime import datetime

import torch
from rich.console import Console

from lionz import constants
from lionz import cli_theme as theme


class OutputManager:
    """Manages console output, logging, and progress indicators."""

    def __init__(self, verbose_console: bool, verbose_log: bool):
        self.verbose_console = verbose_console
        self.verbose_log = verbose_log
        self.console = Console(highlight=False, quiet=not self.verbose_console)

        self._section_counter = 0
        self._spinner_ctx = None

        self.logger: logging.Logger | None = None
        self.nnunet_log_filename = os.devnull

    def configure_logging(self, log_file_directory: str | None):
        """Configure file logging."""
        if not self.verbose_log or self.logger:
            return

        if log_file_directory is None:
            log_file_directory = os.getcwd()

        timestamp = datetime.now().strftime("%H-%M-%d-%m-%Y")
        self.nnunet_log_filename = os.path.join(
            log_file_directory, f"lionz-v{constants.VERSION}_nnUNet_{timestamp}.log"
        )

        self.logger = logging.getLogger(f"lionz-v{constants.VERSION}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False

        if not any(isinstance(handler, logging.FileHandler) for handler in self.logger.handlers):
            log_format = "%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s"
            formatter = logging.Formatter(log_format)
            log_filename = os.path.join(log_file_directory, f"lionz-v{constants.VERSION}_{timestamp}.log")
            file_handler = logging.FileHandler(log_filename, mode="w")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def log_update(self, text: str):
        """Log a message to file."""
        if self.verbose_log and self.logger:
            self.logger.info(text)

    def section(self, title: str, icon: str = ""):
        """Print a numbered section header."""
        self._section_counter += 1
        theme.section(title, self.console, number=f"{self._section_counter:02d}")

    def info(self, msg: str):
        """Print info message."""
        self.console.print(theme.info(msg))

    def ok(self, msg: str):
        """Print success message."""
        self.console.print(theme.ok(msg))

    def warn(self, msg: str):
        """Print warning message."""
        self.console.print(theme.warn(msg))

    def err(self, msg: str):
        """Print error message."""
        self.console.print(theme.err(msg))

    def kv(self, key: str, value: str):
        """Print key-value pair."""
        self.console.print(theme.kv(key, value))

    def message(self, message: str, *, style: str = "text", icon: str | None = None, emphasis: bool = False):
        """Print a styled message."""
        if style == "success":
            self.ok(message)
        elif style == "error":
            self.err(message)
        elif style == "warning":
            self.warn(message)
        else:
            self.info(message)

    def console_update(self, content: str, style: str | None = None):
        """Print to console."""
        self.info(content)

    def context_panel(self, title: str, body, icon: str = ""):
        """Print content (kept for compatibility but simplified)."""
        self.console.print(body)

    def spinner_start(self, text: str | None = None):
        """Start a spinner (uses context manager internally)."""
        if not self.verbose_console:
            return
        self._spinner_ctx = theme.spinner(text or "", self.console)
        self._spinner_ctx.__enter__()

    def spinner_update(self, text: str | None = None):
        """Update spinner text (no-op with context manager spinner)."""
        pass

    def spinner_stop(self):
        """Stop the spinner."""
        if self._spinner_ctx:
            self._spinner_ctx.__exit__(None, None, None)
            self._spinner_ctx = None

    def spinner_succeed(self, text: str | None = None):
        """Stop spinner with success message."""
        self.spinner_stop()
        if text:
            self.ok(text)

    def display_logo(self):
        """Display the LION banner."""
        version = importlib.metadata.version("lionz")
        theme.print_banner(version, self.console)

    def display_citation(self):
        """Display citation information."""
        self.console.print(f"  [reverse {theme.GREIGE}] Citation [/reverse {theme.GREIGE}] [{theme.MUTED}]Pires, Gutschmayer, Shiyam Sundar et al. · 10.5281/zenodo.12626789[/{theme.MUTED}]")

    def create_table(self, header: list[str], styles: list[str] | None = None):
        """Create a styled table."""
        table = theme.make_table()
        for h in header:
            table.add_column(h)
        return table

    def create_progress_bar(self, transient: bool = True):
        """Create a progress bar."""
        from rich.progress import Progress, BarColumn, TextColumn, MofNCompleteColumn, TimeElapsedColumn
        from rich.style import Style
        return Progress(
            TextColumn(f"  [{theme.CORAL}]▸[/{theme.CORAL}]"),
            BarColumn(complete_style=Style(color=theme.CORAL), finished_style=Style(color=theme.CORAL)),
            MofNCompleteColumn(),
            TextColumn(f"[{theme.MUTED}]{{task.description}}[/{theme.MUTED}]"),
            TimeElapsedColumn(),
            console=self.console,
            transient=transient,
        )

    def create_file_progress_bar(self):
        """Create a progress bar for file downloads."""
        from rich.progress import Progress, BarColumn, TextColumn, FileSizeColumn, TransferSpeedColumn, TimeRemainingColumn
        from rich.style import Style
        return Progress(
            TextColumn(f"  [{theme.CORAL}]▸[/{theme.CORAL}]"),
            BarColumn(complete_style=Style(color=theme.CORAL), finished_style=Style(color=theme.CORAL)),
            FileSizeColumn(),
            TransferSpeedColumn(),
            TimeRemainingColumn(),
            TextColumn(f"[{theme.MUTED}]{{task.description}}[/{theme.MUTED}]"),
            console=self.console,
            transient=True,
        )

    def themed_progress(self, *additional_columns, transient: bool = True):
        """Create a themed progress bar."""
        return self.create_progress_bar(transient=transient)


def get_virtual_env_root() -> str:
    """Get the root directory of the virtual environment."""
    return os.path.dirname(os.path.dirname(sys.executable))


def check_device(
    output_manager: OutputManager = None,
    announce: bool = True,
) -> tuple[str, int | None]:
    """Check available compute devices."""
    if output_manager is None:
        output_manager = OutputManager(False, False)

    def emit(message: str, is_success: bool):
        if announce:
            if is_success:
                output_manager.ok(message)
            else:
                output_manager.info(message)
        output_manager.log_update(f" Accelerator: {message}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        emit(f"CUDA · {device_count} GPU(s)", True)
        return "cuda", device_count

    if torch.backends.mps.is_available():
        emit("Apple MPS · GPU acceleration", True)
        return "mps", None

    emit("CPU mode", False)
    return "cpu", None


ENVIRONMENT_ROOT_PATH: str = get_virtual_env_root()
MODELS_DIRECTORY_PATH: str = os.path.join(ENVIRONMENT_ROOT_PATH, "models", "nnunet_trained_models")
