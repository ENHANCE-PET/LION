#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.metadata
import logging
import os
import sys
from datetime import datetime
from typing import Iterable

import emoji
import pyfiglet
import torch
from halo import Halo
from rich import box
from rich.align import Align
from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    FileSizeColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from lionz import constants

try:
    from cfonts import render as cfonts_render
    from cfonts import say as cfonts_say
except ImportError:  # pragma: no cover - optional dependency
    cfonts_render = None
    cfonts_say = None


class OutputManager:
    def __init__(self, verbose_console: bool, verbose_log: bool):
        self.verbose_console = verbose_console
        self.verbose_log = verbose_log

        theme = Theme(
            {
                "lion.text": constants.CLI_COLORS["text"],
                "lion.secondary": constants.CLI_COLORS["secondary"],
                "lion.info": constants.CLI_COLORS["info"],
                "lion.success": constants.CLI_COLORS["success"],
                "lion.warning": constants.CLI_COLORS["warning"],
                "lion.error": constants.CLI_COLORS["error"],
            }
        )

        self.console = Console(theme=theme, highlight=False, quiet=not self.verbose_console)
        self.spinner = Halo(spinner="earth", enabled=self.verbose_console)

        self.logger: logging.Logger | None = None
        self.nnunet_log_filename = os.devnull

    def themed_progress(self, *additional_columns: Iterable, transient: bool = True) -> Progress:
        base_columns = [
            TextColumn(
                f"[bold {constants.CLI_COLORS['secondary']}]{{task.description}}",
                justify="left",
            ),
            BarColumn(
                bar_width=None,
                style=constants.CLI_COLORS["muted"],
                complete_style=constants.CLI_COLORS["primary"],
                finished_style=constants.CLI_COLORS["primary"],
                pulse_style=constants.CLI_COLORS["accent"],
            ),
            TextColumn(
                f"[{constants.CLI_COLORS['info']}]{{task.percentage:>3.0f}}%",
                justify="right",
            ),
        ]
        columns = [*base_columns, *additional_columns]
        return Progress(*columns, console=self.console, transient=transient, expand=True)

    def create_file_progress_bar(self) -> Progress:
        return self.themed_progress(FileSizeColumn(), TransferSpeedColumn(), TimeRemainingColumn())

    def create_progress_bar(self, transient: bool = True) -> Progress:
        return self.themed_progress(transient=transient)

    def create_table(self, header: list[str], styles: list[str] | None = None) -> Table:
        table = Table(
            show_header=True,
            header_style=f"bold {constants.CLI_COLORS['secondary']}",
            style=constants.CLI_COLORS["text"],
            show_edge=False,
            box=box.SIMPLE_HEAD,
            pad_edge=False,
        )
        if styles is None:
            styles = [constants.CLI_COLORS["text"]] * len(header)
        for header_label, style in zip(header, styles):
            resolved_style = style or constants.CLI_COLORS["text"]
            table.add_column(header_label, style=resolved_style)
        return table

    def configure_logging(self, log_file_directory: str | None):
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
        if self.verbose_log and self.logger:
            self.logger.info(text)

    def console_update(self, content: str | RenderableType, style: str | None = None):
        if isinstance(content, str):
            resolved_style = constants.CLI_COLORS.get(style, style) if style else constants.CLI_COLORS["text"]
            text = Text(content, style=resolved_style)
            self.console.print(text)
        else:
            self.console.print(content)

    def section(self, title: str, icon: str = ""):
        heading_text = emoji.emojize(f"{icon} {title}" if icon else title).strip()
        header = Text(heading_text, style=f"bold {constants.CLI_COLORS['secondary']}")
        accent_width = max(
            constants.ACCENT_LINE_MIN_WIDTH,
            min(self.console.width - 4, constants.ACCENT_LINE_MAX_WIDTH),
        )
        accent_line = Text(constants.ACCENT_LINE_GLYPH * accent_width, style=constants.CLI_COLORS["muted"])

        self.console.print()
        self.console.print(header)
        self.console.print(accent_line)
        

    def message(self, message: str, *, style: str = "text", icon: str | None = None, emphasis: bool = False):
        resolved_style = constants.CLI_COLORS.get(style, style)
        text = Text()
        if icon:
            text.append(f"{emoji.emojize(icon, language='alias')} ", style=resolved_style)
        applied_style = f"bold {resolved_style}" if emphasis else resolved_style
        text.append(message, style=applied_style)
        self.console.print(text)

    def context_panel(self, title: str, body: str | RenderableType, icon: str = ":memo:"):
        header = Text(
            f"{emoji.emojize(icon, language='alias')} {title}" if icon else title,
            style=f"bold {constants.CLI_COLORS['secondary']}",
        )

        if isinstance(body, str):
            body_renderable: RenderableType = Align.left(Text(body, style=constants.CLI_COLORS["text"]))
        elif isinstance(body, Text):
            body_renderable = Align.left(body)
        else:
            body_renderable = body

        panel = Panel(
            body_renderable,
            title=header,
            title_align="left",
            border_style=constants.CLI_COLORS["border"],
            padding=constants.CONTEXT_PANEL_PADDING,
        )
        self.console.print(panel)
        self.console.print()

    def spinner_update(self, text: str | None = None):
        if self.spinner.enabled:
            self.spinner.text = text

    def spinner_stop(self):
        if self.spinner.enabled:
            self.spinner.stop()

    def spinner_start(self, text: str | None = None):
        if self.spinner.enabled:
            self.spinner.start(text)

    def spinner_succeed(self, text: str | None = None):
        if not self.spinner.enabled:
            if text:
                self.console.print(f"✅ {text}")
            return

        self.spinner.stop()
        if text:
            self.console.print(f" {text}")

    def display_logo(self):
        version = importlib.metadata.version("lionz")

        self.console.print()
        self.console.print()
        banner_payload = self._render_banner(f"LION {version}")
        if banner_payload:
            if isinstance(banner_payload, str):
                self.console.print(banner_payload, markup=False)
            else:
                self.console.print(banner_payload)

        tagline = Text(
            constants.TAGLINE,
            style=f"bold {constants.CLI_COLORS['secondary']}",
            justify="center",
        )
        community = Text(
            constants.COMMUNITY_STATEMENT,
            style=constants.CLI_COLORS["secondary"],
            justify="center",
        )
        self.console.print(Align.center(tagline))
        self.console.print(Align.center(community))
        self.console.print()

    def display_citation(self):
        citation = Text(
            "10.5281/zenodo.12626789\n"
            "Copyright 2023, "
            "Quantitative Imaging and Medical Physics Team, Medical University of Vienna",
            style=constants.CLI_COLORS["text"],
            justify="left",
        )
        self.context_panel("Citation", citation, icon=":books:")

    def _render_banner(self, text: str) -> RenderableType | None:
        colors = constants.BANNER_COLORS
        font = getattr(constants, "BANNER_FONT", "block")

        if cfonts_say:
            try:
                cfonts_say(text, colors=colors, align="center", font=font, space=False)
                return None
            except TypeError:
                pass

        if cfonts_render:
            rendered = cfonts_render(text, colors=colors, align="center", font=font, space=False)
            if isinstance(rendered, dict):
                return rendered.get("string", "")
            if isinstance(rendered, str):
                return rendered

        ascii_art = pyfiglet.figlet_format(text, font="speed").rstrip()
        return Text(ascii_art, style=constants.CLI_COLORS["primary"], justify="center")


def get_virtual_env_root() -> str:
    python_exe = sys.executable
    return os.path.dirname(os.path.dirname(python_exe))


def check_device(
    output_manager: OutputManager = OutputManager(False, False),
    announce: bool = True,
) -> tuple[str, int | None]:
    def emit(message: str, style: str, icon: str):
        if announce:
            output_manager.message(message, style=style, icon=icon)
        output_manager.log_update(f" Accelerator selection: {message}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        emit(
            f"CUDA is available with {device_count} GPU(s). Predictions will be run on GPU.",
            style="success",
            icon=":high_voltage:",
        )
        return "cuda", device_count

    if torch.backends.mps.is_available():
        emit(
            "Apple MPS backend is available. Predictions will be run on Apple Silicon GPU.",
            style="info",
            icon="🍎",
        )
        return "mps", None

    if not torch.backends.mps.is_built():
        emit(
            "MPS not available because the current PyTorch install was not built with MPS enabled.",
            style="warning",
            icon=":warning:",
        )
        return "cpu", None

    emit(
        "CUDA/MPS not available. Predictions will be run on CPU.",
        style="info",
        icon=":gear:",
    )
    return "cpu", None


ENVIRONMENT_ROOT_PATH: str = get_virtual_env_root()
MODELS_DIRECTORY_PATH: str = os.path.join(ENVIRONMENT_ROOT_PATH, "models", "nnunet_trained_models")
