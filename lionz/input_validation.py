#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module contains functions that validate the user inputs for the LIONz project.

It checks parameters like the existence of the parent folder and the validity of the model name.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>, 

.. versionadded:: 0.1.0
"""

import logging
import os
from typing import List

import emoji

from lionz import constants
from lionz import system
from lionz import models


def determine_model_expectations(model_routine: dict[tuple, list[models.ModelWorkflow]], output_manager: system.OutputManager) -> list:
    """
    Display expected modality for the model.

    This function displays the expected modality for the given model name. It also checks for a special case where
    'FDG-PET-CT' should be split into 'FDG-PET' and 'CT'.

    :param model_routine: The model routine
    :type model_routine: dict[tuple, list[models.ModelWorkflow]]
    :param output_manager: The output manager
    :type output_manager: system.OutputManager
    :return: A list of modalities.
    :rtype: list
    """
    required_modalities = []
    required_prefixes = []

    header = ["Nr", "Model Name", "Imaging", "Required Modality", "Required Prefix (non-DICOM)",
              "Nr of training datasets"]
    styles = [None, "cyan", None, None, None, None]
    table = output_manager.create_table(header, styles)

    model_nr = 0
    for model_workflows in model_routine.values():
        for model_workflow in model_workflows:
            model_nr += 1
            modalities, prefixes = model_workflow.target_model.get_expectation()
            required_modalities = required_modalities + modalities
            required_prefixes = required_prefixes + prefixes

            model_identifier = model_workflow.target_model.model_identifier
            modality = model_workflow.target_model.modality
            imaging = f"{model_workflow.target_model.imaging_type}ical".capitalize()
            nr_training_data = model_workflow.target_model.nr_training_data
            table.add_row(str(model_nr), model_identifier, imaging, modality, ', '.join(prefixes), nr_training_data)

    output_manager.console_update(table)

    required_modalities = list(set(required_modalities))
    required_prefixes = list(set(required_prefixes))

    output_manager.log_update(f" Required modalities: {required_modalities} | No. of modalities: {len(required_modalities)} "
                              f"| Required prefix for non-DICOM files: {required_prefixes} ")
    output_manager.console_update(f"{constants.ANSI_ORANGE} Warning: Subjects which don't have the required modalities [check file prefix] "
                                  f"will be skipped. {constants.ANSI_RESET}")
    output_manager.log_update(" Skipping subjects without the required modalities (check file prefix).\n"
                              " These subjects will be excluded from analysis and their data will not be used.")

    return required_modalities


def validate_inputs(parent_folder: str, model_name: str) -> bool:
    """
    Validates the user inputs for the main function.
    
    :param parent_folder: The parent folder containing subject folders.
    :type parent_folder: str
    
    :param model_name: The name of the model to use for segmentation.
    :type model_name: str
    
    :return: True if the inputs are valid, False otherwise.
    :rtype: bool
    """
    return validate_parent_folder(parent_folder) and validate_model_name(model_name)


def validate_parent_folder(parent_folder: str) -> bool:
    """Validates if the parent folder exists."""
    if os.path.isdir(parent_folder):
        return True
    else:
        message = f"The parent folder {parent_folder} does not exist."
        logging.error(message)
        print_error(message)
        return False


def validate_model_name(model_name: str) -> bool:
    """Validates if the model name is available."""
    if model_name in models.AVAILABLE_MODELS:
        return True
    else:
        message = f"The model name {model_name} is invalid."
        logging.error(message)
        print_error(message)
        return False


def print_error(message: str):
    """Prints an error message with standard formatting."""
    print(f"{emoji.emojize(':cross_mark:')} {constants.ANSI_RED} {message} {constants.ANSI_RESET}")


def select_lion_compliant_subjects(subject_paths: list[str], modality_tags: list[str], output_manager: system.OutputManager) -> list[str]:
    """
    Selects the subjects that have the files that have names that are compliant with the lionz.

    :param subject_paths: The path to the list of subjects that are present in the parent directory.
    :type subject_paths: List[str]
    :param modality_tags: The list of appropriate modality prefixes that should be attached to the files for
                          them to be lion compliant.
    :type modality_tags: List[str]
    :param output_manager: The output manager that will be used to write the output files.
    :type output_manager: system.OutputManager
    :return: The list of subject paths that are lion compliant.
    :rtype: List[str]
    """
    # go through each subject in the parent directory
    lion_compliant_subjects = []
    for subject_path in subject_paths:
        # go through each subject and see if the files have the appropriate modality prefixes

        files = [file for file in os.listdir(subject_path) if file.endswith('.nii') or file.endswith('.nii.gz')]
        prefixes = [file.startswith(tag) for tag in modality_tags for file in files]
        if sum(prefixes) == len(modality_tags):
            lion_compliant_subjects.append(subject_path)
    output_manager.console_update(f"{constants.ANSI_ORANGE} Number of lion compliant subjects: {len(lion_compliant_subjects)} out of {len(subject_paths)} {constants.ANSI_RESET}")
    output_manager.log_update(f" Number of lion compliant subjects: {len(lion_compliant_subjects)} out of {len(subject_paths)}")

    return lion_compliant_subjects
