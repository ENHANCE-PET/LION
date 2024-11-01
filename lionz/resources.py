#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ Resources
---------------

This module contains utility functions and resources that are crucial for the operations of the LIONZ application.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets. The resources module is designed to manage and provide auxiliary resources, such as configuration files, model weights, and other important artifacts necessary for the proper functioning of the application.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

BINARIES = {
    "windows-x86_64": {
        "url": "https://enhance-pet.s3.eu-central-1.amazonaws.com/awesome/beast-binaries-windows-x86_64.zip",
        "filename": "beast-binaries-windows-x86_64.zip",
        "directory": "beast-binaries-windows-x86_64",
    },
    "linux-x86_64": {
        "url": "https://enhance-pet.s3.eu-central-1.amazonaws.com/awesome/beast-binaries-linux-x86_64.zip",
        "filename": "beast-binaries-linux-x86_64.zip",
        "directory": "beast-binaries-linux-x86_64",
    },
    "mac-x86_64": {
        "url": "https://enhance-pet.s3.eu-central-1.amazonaws.com/awesome/beast-binaries-mac-x86_64.zip",
        "filename": "beast-binaries-mac-x86_64.zip",
        "directory": "beast-binaries-mac-x86_64",
    },
    "mac-arm64": {
        "url": "https://enhance-pet.s3.eu-central-1.amazonaws.com/awesome/beast-binaries-mac-arm64.zip",
        "filename": "beast-binaries-mac-arm64.zip",
        "directory": "beast-binaries-mac-arm64",
    },
}