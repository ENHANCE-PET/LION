#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Constants
---------------

This module contains the constants that are used in the LIONZ project.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.10.0
"""

import os
import sys
from importlib import metadata
from pathlib import Path


def _load_version_from_pyproject() -> str | None:
    """Best-effort extraction of the project version from pyproject.toml."""

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject_path.is_file():
        return None

    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        tomllib = None

    if tomllib is not None:  # pragma: no cover - depends on runtime
        try:
            with pyproject_path.open("rb") as pyproject_file:
                data = tomllib.load(pyproject_file)
            return data.get("project", {}).get("version")
        except (OSError, ValueError, AttributeError):
            return None

    try:  # pragma: no cover - optional dependency
        import tomli
    except ModuleNotFoundError:  # pragma: no cover - best-effort fallback
        tomli = None

    if tomli is not None:
        try:
            with pyproject_path.open("rb") as pyproject_file:
                data = tomli.load(pyproject_file)
            return data.get("project", {}).get("version")
        except (OSError, ValueError, AttributeError):
            return None

    try:
        for line in pyproject_path.read_text(encoding="utf-8").splitlines():
            cleaned = line.strip()
            if cleaned.startswith("version"):
                _, value = cleaned.split("=", 1)
                return value.strip().strip('"').strip("'")
    except OSError:
        return None

    return None


def _resolve_package_version() -> str:
    fallback = _load_version_from_pyproject()
    if fallback:
        return fallback
    try:
        return metadata.version("lionz")
    except metadata.PackageNotFoundError:  # pragma: no cover - development tree
        return "0.0.0"


def get_virtual_env_root() -> str:
    """
    Returns the root directory of the virtual environment.

    :return: The root directory of the virtual environment.
    :rtype: str
    """
    python_exe = sys.executable
    virtual_env_root = os.path.dirname(os.path.dirname(python_exe))
    return virtual_env_root


# Get the root directory of the virtual environment
project_root = get_virtual_env_root()
BINARY_PATH = os.path.join(project_root, 'bin')
VERSION = _resolve_package_version()

# Define the paths to the trained models and the LIONZ model
NNUNET_RESULTS_FOLDER = os.path.join(project_root, 'models', 'nnunet_trained_models')
LIONZ_MODEL_FOLDER = os.path.join(NNUNET_RESULTS_FOLDER, 'nnUNet', '3d_fullres')


# Define the allowed modalities
ALLOWED_MODALITIES = ['CT', 'PT']

# Define the name of the temporary folder
TEMP_FOLDER = 'temp'

# CLI colour palette derived from the shared QIMP CLI style.
CLI_COLORS = {
    "primary": "#ff79c6",
    "secondary": "#4163ca",
    "success": "#34c658",
    "warning": "#f0c37b",
    "error": "#eb7777",
    "info": "#2b60dc",
    "accent": "#bd93f9",
    "muted": "#44475a",
    "text": "#44475a",
    "border": "#d795be",
}

BANNER_COLORS = [
    CLI_COLORS["secondary"],
    CLI_COLORS["primary"],
]

BANNER_FONT = "block"
TAGLINE = "The New Standard in PET Lesion Segmentation."
COMMUNITY_STATEMENT = "A part of the ENHANCE.PET initiative. Join us at www.enhance.pet to build the future of PET imaging together."
MISSION_STATEMENT = ""

ACCENT_LINE_GLYPH = "─"
ACCENT_LINE_MIN_WIDTH = 12
ACCENT_LINE_MAX_WIDTH = 48
CONTEXT_PANEL_PADDING = (1, 2)

# Define folder names
SEGMENTATIONS_FOLDER = 'segmentations'
STATS_FOLDER = 'stats'
WORKFLOW_FOLDER = 'workflow'

# PREPROCESSING PARAMETERS

MATRIX_THRESHOLD = 200 * 200 * 600
Z_AXIS_THRESHOLD = 200
MARGIN_PADDING = 20
INTERPOLATION = 'bspline'
CHUNK_THRESHOLD_RESAMPLING = 150
MARGIN_SCALING_FACTOR = 2
# DISPLAY PARAMETERS

MIP_ROTATION_STEP = 40
MIP_VOXEL_SPACING = (4, 4, 4)
FRAME_DURATION = 0.4

# Training dataset number 

TRAINING_DATASET_SIZE_FDG = '5341' 
TRAINING_DATASET_SIZE_PSMA = '2299'

# MODELS
KEY_FOLDER_NAME = "folder_name"
KEY_URL = "url"
KEY_LIMIT_FOV = "limit_fov"
KEY_DESCRIPTION = "description"
KEY_DESCRIPTION_TEXT = "Tissue of Interest"
KEY_DESCRIPTION_MODALITY = "Modality"
KEY_DESCRIPTION_IMAGING = "Imaging"
DEFAULT_SPACING = (1.5, 1.5, 1.5)
FILE_NAME_DATASET_JSON = "dataset.json"
FILE_NAME_PLANS_JSON = "plans.json"
TUMOR_LABEL = 0


USAGE_MESSAGE = """
LIONZ (Lesion segmentatION) focuses on precise tumor segmentation in PET/CT datasets.

Example:
  lionz -d /Documents/Data_to_lionz/ -m fdg
"""
