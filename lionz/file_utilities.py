#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ File Utilities
---------------------

This module provides utility functions specifically designed to handle file operations for the LIONZ application.

LIONZ, standing for Lesion segmentatION, offers an advanced solution for lesion segmentation tasks within medical imaging datasets. The file utilities module ensures efficient, reliable, and organized manipulation of files and directories—be it reading, writing, or organizing data, configuration files, model artifacts, and more. Such functions simplify the I/O operations and maintain the consistency and integrity of the application's data structure.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import os
import shutil
from datetime import datetime
import subprocess
import stat
import platform
from lionz.constants import SEGMENTATIONS_FOLDER, STATS_FOLDER, BINARY_PATH
from lionz import system

def get_c3d_path():
    if get_system()[0] == 'windows':
        C3D_PATH = os.path.join(BINARY_PATH, f'beast-binaries-{get_system()[0]}-{get_system()[1]}',
                                'c3d.exe')
    elif get_system()[0] in ['linux', 'mac']:
        C3D_PATH = os.path.join(BINARY_PATH, f'beast-binaries-{get_system()[0]}-{get_system()[1]}',
                                'c3d')
    else:
        raise ValueError('Unsupported OS')
    return C3D_PATH


def set_permissions(file_path: str, system_type: str, output_manager: system.OutputManager) -> None:
    """
    Sets the permissions of a file based on the operating system.

    :param str file_path: The absolute or relative path to the file.
    :param str system_type: The type of the operating system ('windows', 'linux', 'mac').
    :return: None
    :rtype: None
    :raises FileNotFoundError: If the file specified by 'file_path' does not exist.
    :raises ValueError: If the provided operating system type is not supported.
    :raises subprocess.CalledProcessError: If the 'icacls' command fails on Windows.
    :raises PermissionError: If the 'chmod' command fails on Linux or macOS.

    **Example**

    .. code-block:: python

        >>> set_permissions('/path/to/file', 'linux', output_manager)

    """
    if not os.path.exists(file_path):
        output_manager.log_update(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if system_type.lower() == 'windows':
            subprocess.check_call(["icacls", file_path, "/grant", "Everyone:(F)"], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        elif system_type.lower() in ['linux', 'mac']:
            os.chmod(file_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
        else:
            output_manager.log_update(f"Unsupported operating system type provided: {system_type}")
            raise ValueError(f"Unsupported operating system type: {system_type}")
    except subprocess.CalledProcessError as e:
        output_manager.log_update(f"Failed to set permissions for file '{file_path}' on a Windows system. Subprocess error: {e}")
        raise e
    except PermissionError as e:
        output_manager.log_update(f"Insufficient permissions to change file permissions for '{file_path}' on a {system_type} system. Error: {e}")
        raise e
    except Exception as e:
        output_manager.log_update(f"An unexpected error occurred while setting permissions for file '{file_path}'. Error details: {e}")
        raise e


def get_system():
    """
    Get the operating system and architecture.

    :return: A tuple containing the operating system and architecture.
    :rtype: tuple
    :raises: ValueError if the operating system or architecture is not supported.

    This function gets the operating system and architecture by using the `platform.system` and `platform.machine`
    functions. It converts the output of these functions to match the keys used in the rest of the code. If the operating
    system or architecture is not supported, it raises a ValueError.

    :Example:
        >>> get_system()
        ('linux', 'x86_64')
    """
    system = platform.system().lower()
    architecture = platform.machine().lower()

    # Convert system output to match your keys
    if system == "darwin":
        system = "mac"
    elif system == "windows":
        system = "windows"
    elif system == "linux":
        system = "linux"
    else:
        raise ValueError("Unsupported OS type")

    # Convert architecture output to match your keys
    if architecture in ["x86_64", "amd64"]:
        architecture = "x86_64"
    elif "arm" in architecture:
        architecture = "arm64"
    else:
        raise ValueError("Unsupported architecture")

    return system, architecture


def get_files(directory: str, prefix: str, suffix: str | tuple) -> list[str]:
    """
    Returns the list of files in the directory with the specified wildcard.

    :param directory: The directory path.
    :type directory: str

    :param suffix: The wildcard to be used.
    :type suffix: str

    :param prefix: The wildcard to be used.
    :type prefix: str

    :return: The list of files.
    :rtype: list
    """

    if isinstance(suffix, str):
        suffix = (suffix,)

    files = []
    for file in os.listdir(directory):
        if file.startswith(prefix) and file.endswith(suffix):
            files.append(os.path.join(directory, file))
    return files


def create_directory(directory_path: str) -> str:
    """
    Creates a directory at the specified path and returns its path.
    
    :param directory_path: The path to the directory.
    :type directory_path: str
    
    :return: The path of the created directory.
    :rtype: str
    """
    if not os.path.isdir(directory_path):
        os.makedirs(directory_path)
    return directory_path


def lion_folder_structure(parent_directory: str) -> tuple[str, str, str]:
    """
    Creates the moose folder structure.

    :param parent_directory: The path to the parent directory.
    :type parent_directory: str

    :return: A tuple containing the paths to the moose directory, output directory, and stats directory.
    :rtype: tuple
    """
    lion_dir = os.path.join(parent_directory, 'lionz-' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    create_directory(lion_dir)

    segmentation_dir = os.path.join(lion_dir, SEGMENTATIONS_FOLDER)
    stats_dir = os.path.join(lion_dir, STATS_FOLDER)
    create_directory(segmentation_dir)
    create_directory(stats_dir)
    return lion_dir, segmentation_dir, stats_dir


def copy_file(file: str, destination: str) -> None:
    """
    Copies a file to the specified destination.

    :param file: The path to the file to be copied.
    :type file: str

    :param destination: The path to the destination directory.
    :type destination: str
    """
    shutil.copy(file, destination)


def get_nifti_file_stem(file_path: str) -> str:
    file_stem = os.path.basename(file_path).split('.gz')[0].split('.nii')[0]
    return file_stem