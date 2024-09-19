#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Constants
---------------

This module contains the constants that are used in the LIONZ project.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import os
import sys


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

# Define the paths to the trained models and the LIONZ model
NNUNET_RESULTS_FOLDER = os.path.join(project_root, 'models', 'nnunet_trained_models')
LIONZ_MODEL_FOLDER = os.path.join(NNUNET_RESULTS_FOLDER, 'nnUNet', '3d_fullres')

# Define the allowed modalities
ALLOWED_MODALITIES = ['CT', 'PT']

# Define the name of the temporary folder
TEMP_FOLDER = 'temp'

# Define color codes for console output
ANSI_ORANGE = '\033[38;5;208m'
ANSI_GREEN = '\033[38;5;40m'
ANSI_VIOLET = '\033[38;5;141m'
ANSI_RED = '\033[38;5;196m'
ANSI_RESET = '\033[0m'

# Define folder names
SEGMENTATIONS_FOLDER = 'segmentations'
STATS_FOLDER = 'stats'
WORKFLOW_FOLDER = 'workflow'

# PREPROCESSING PARAMETERS

MATRIX_THRESHOLD = 200 * 200 * 600
Z_AXIS_THRESHOLD = 200
MARGIN_PADDING = 20
INTERPOLATION = 'bspline'
CHUNK_THRESHOLD = 200
MARGIN_SCALING_FACTOR = 2

# DISPLAY PARAMETERS

MIP_ROTATION_STEP = 40
MIP_VOXEL_SPACING = (4, 4, 4)
FRAME_DURATION = 0.4

# Training dataset number 

TRAINING_DATASET_SIZE_FDG = '1022' 
TRAINING_DATASET_SIZE_PSMA = '812'
