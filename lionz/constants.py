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
VERSION = '0.10'

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
CHUNK_THRESHOLD_RESAMPLING = 150
MARGIN_SCALING_FACTOR = 2
# DISPLAY PARAMETERS

MIP_ROTATION_STEP = 40
MIP_VOXEL_SPACING = (4, 4, 4)
FRAME_DURATION = 0.4

# Training dataset number 

TRAINING_DATASET_SIZE_FDG = '1022' 
TRAINING_DATASET_SIZE_PSMA = '812'

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
    Usage:
      lionz -d <MAIN_DIRECTORY> -m <MODEL_NAME>
    Example:  
      lionz -d /Documents/Data_to_lionz/ -m clin_ct_lesions

    Description:
      LIONZ (Lesion segmentatION) - A state-of-the-art AI solution that
      emphasizes precise lesion segmentation in diverse imaging datasets.
    """