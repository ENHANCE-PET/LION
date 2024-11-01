#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Resources
---------------

This module contains functions that are responsible for downloading and managing resources for the LIONZ application.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.10.0
"""

import logging
import os
import zipfile
import requests
from lionz import system


def download(item_name: str, item_path: str, item_dict: dict, output_manager: system.OutputManager) -> str:
    """
    Downloads the item (model or binary) for the current system.

    :param item_name: The name of the item to download.
    :type item_name: str
    :param item_path: The path to store the item.
    :type item_path: str
    :param item_dict: The dictionary containing item info.
    :type item_dict: dict
    :return: The path to the downloaded item.
    :rtype: str
    :raises: None

    This function downloads the item specified by `item_name` from the URL specified in the `item_dict` dictionary. It
    shows the download progress using the `rich` library and extracts the downloaded zip file using the `zipfile`
    library. If the item has already been downloaded, it skips the download and returns the path to the local copy of
    the item.

    :Example:
        >>> download('registration_binaries', '/path/to/item', {'registration_binaries': {'url': 'http://example.com/binaries.zip', 'filename': 'binaries.zip', 'directory': 'binaries'}})
    """
    item_info = item_dict[item_name]
    url = item_info["url"]
    filename = os.path.join(item_path, item_info["filename"])
    directory = os.path.join(item_path, item_info["directory"])

    if not os.path.exists(directory):
        # Download the item
        output_manager.log_update(f" Downloading {directory}")

        # show progress using rich
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("Content-Length", 0))
        chunk_size = 1024 * 10

        progress = output_manager.create_file_progress_bar()
        with progress:
            task = progress.add_task("[white] Downloading system specific c3d binaries", total=total_size)
            for chunk in response.iter_content(chunk_size=chunk_size):
                open(filename, "ab").write(chunk)
                progress.update(task, advance=chunk_size)
        output_manager.log_update(f"    - system specific  c3d binaries downloaded.")

        # Unzip the item
        progress = output_manager.create_file_progress_bar()
        with progress:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                total_size = sum((file.file_size for file in zip_ref.infolist()))
                task = progress.add_task("[white] Extracting system specific c3d binaries",
                                         total=total_size)
                # Get the parent directory of 'directory'
                parent_directory = os.path.dirname(directory)
                for file in zip_ref.infolist():
                    zip_ref.extract(file, parent_directory)
                    extracted_size = file.file_size
                    progress.update(task, advance=extracted_size)

        output_manager.log_update(f" {os.path.basename(directory)} extracted.")

        # Delete the zip file
        os.remove(filename)
        output_manager.console_update(f" c3d binaries - download complete.")
        output_manager.log_update(f" c3d binaries - download complete.")
    else:
        output_manager.console_update(
            f" A local instance of the system specific c3d binary has been detected.")
        output_manager.log_update(f" A local instance of c3dd binary has been detected.")

    return os.path.join(item_path, item_name)