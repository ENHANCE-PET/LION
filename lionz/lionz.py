#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIONZ: Lesion Segmentation Tool
-------------------------------

This module, `lionz.py`, serves as the main entry point for the LIONZ toolkit.
It provides capabilities for tumor and lesion segmentation in PET/CT datasets.

Notes
-----
.. note:: 
   For a full understanding of the capabilities and functionalities of this module, 
   refer to the individual function and class docstrings.

Attributes
----------
__author__ : str
    Module author(s).
    
__email__ : str
    Contact email for module inquiries.

__version__ : str
    Current version of the module.

Examples
--------
To use this module, you can either import it into another script or run it directly:

.. code-block:: python

    import lionz
    # Use functions or classes

or:

.. code-block:: bash

    $ python lionz.py

See Also
--------
constants : Module containing constant values used throughout the toolkit.
display : Module responsible for displaying information and graphics.
image_processing : Module with functions and classes for image processing tasks.
input_validation : Module that provides functionalities for validating user inputs.
resources : Contains resource files and data necessary for the toolkit.
download : Handles downloading of data, models, or other necessary resources.

"""

__author__ = "Lalith kumar shiyam sundar, Sebastian Gutschmayer, Manuel pires"
__email__ = "lalith.shiyamsundar@meduniwien.ac.at, sebastian.gutschmayer@meduniwien.ac.at, manuel.pires@meduniwien.ac.at"
__version__ = "0.1"

# Imports for the module


import argparse
import glob
import logging
import os
import time
from datetime import datetime

import colorama
import emoji
from halo import Halo

from lionz import constants
from lionz import display
from lionz import download
from lionz import file_utilities
from lionz import image_conversion
from lionz import input_validation
from lionz import image_processing
from lionz.predict import predict_tumor, post_process
from lionz.resources import AVAILABLE_MODELS, check_cuda, TRACER_WORKFLOWS

from lionz.nnUNet_custom_trainer.utility import add_custom_trainers_to_local_nnunetv2


# Main function for the module
def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                        filename=datetime.now().strftime('lionz-v.0.1.0.%H-%M-%d-%m-%Y.log'), filemode='w')
    colorama.init()

    # Argument parser
    parser = argparse.ArgumentParser(
        description=display.get_usage_message(),
        formatter_class=argparse.RawTextHelpFormatter,  # To retain the custom formatting
        add_help=False  # We'll add our own help option later
    )

    # Main directory containing subject folders
    parser.add_argument(
        "-d", "--main_directory",
        type=str,
        required=True,
        metavar="<MAIN_DIRECTORY>",
        help="Specify the main directory containing subject folders."
    )

    # Name of the model to use for segmentation
    model_help_text = "Choose the model for segmentation from the following:\n" + "\n".join(AVAILABLE_MODELS)
    parser.add_argument(
        "-m", "--model_name",
        type=str,
        choices=AVAILABLE_MODELS,
        required=True,
        metavar="<MODEL_NAME>",
        help=model_help_text
    )

    # Whether the obtained segmentations should be thresholded
    parser.add_argument(
        "-t", "--thresholding",
        required=False,
        default=False,
        action='store_true',
        help="Use to threshold the segmentations"
    )

    # Custom help option
    parser.add_argument(
        "-h", "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="Show this help message and exit."
    )

    # Parse the arguments
    args = parser.parse_args()

    # Get the main directory and model name
    parent_folder = os.path.abspath(args.main_directory)
    model_name = args.model_name

    # Check for thresholding
    thresholding = args.thresholding

    # Display messages
    display.logo()
    display.citation()

    logging.info('----------------------------------------------------------------------------------------------------')
    logging.info('                                     STARTING LIONZ-v.0.1.0                                         ')
    logging.info('----------------------------------------------------------------------------------------------------')

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    logging.info(' ')
    logging.info('- Main directory: ' + parent_folder)
    logging.info('- Model name: ' + model_name)
    logging.info(' ')
    print(' ')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    print(' ')
    print(f'{constants.ANSI_ORANGE} Training dataset size > FDG: {constants.TRAINING_DATASET_SIZE_FDG} | PSMA: {constants.TRAINING_DATASET_SIZE_PSMA} {constants.ANSI_RESET}')
    modalities = display.expectations(model_name)
    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    logging.info('- Custom trainer: ' + custom_trainer_status)
    accelerator = check_cuda()
    inputs_valid = input_validation.validate_inputs(parent_folder, model_name)
    if not inputs_valid:
        exit(1)
    else:
        logging.info(f"Input validation successful.")

    # ------------------------------
    # DOWNLOAD THE MODEL
    # ------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    print('')
    model_path = constants.NNUNET_RESULTS_FOLDER
    file_utilities.create_directory(model_path)
    download.model(model_name, model_path)

    # ------------------------------
    # INPUT STANDARDIZATION
    # ------------------------------
    print('')
    print(
        f'{constants.ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} STANDARDIZING INPUT DATA TO NIFTI:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' STANDARDIZING INPUT DATA TO NIFTI:')
    logging.info(' ')
    image_conversion.standardize_to_nifti(parent_folder)
    print(f"{constants.ANSI_GREEN} Standardization complete.{constants.ANSI_RESET}")
    logging.info(" Standardization complete.")

    # ------------------------------
    # CHECK FOR LIONZ COMPLIANCE
    # ------------------------------

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if
                os.path.isdir(os.path.join(parent_folder, d))]
    lion_compliant_subjects = input_validation.select_lion_compliant_subjects(subjects, modalities)

    num_subjects = len(lion_compliant_subjects)
    if num_subjects < 1:
        print(f'{constants.ANSI_RED} {emoji.emojize(":cross_mark:")} No lion compliant subject found to continue!{constants.ANSI_RESET} {emoji.emojize(":light_bulb:")} See: https://github.com/LalithShiyam/LION#directory-conventions-for-lion-%EF%B8%8F')
        return

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR LION COMPLIANT SUBJECTS
    # -------------------------------------------------

    print('')
    print(f'{constants.ANSI_VIOLET} {emoji.emojize(":crystal_ball:")} PREDICT:{constants.ANSI_RESET}')
    print('')
    logging.info(' ')
    logging.info(' PERFORMING PREDICTION:')
    logging.info(' ')

    spinner = Halo(text=' Initiating', spinner='dots')
    spinner.start()
    start_total_time = time.time()

    for i, subject in enumerate(lion_compliant_subjects):
        # SETTING UP DIRECTORY STRUCTURE
        spinner.text = f'[{i + 1}/{num_subjects}] Setting up directory structure for {os.path.basename(subject)}...'
        logging.info(' ')
        logging.info(f'{constants.ANSI_VIOLET} SETTING UP LION-Z DIRECTORY:'
                     f'{constants.ANSI_RESET}')
        logging.info(' ')
        lion_dir, input_dirs, output_dir, stats_dir, workflow_dir = file_utilities.lion_folder_structure(subject,
                                                                                                         model_name,
                                                                                                         modalities)
        logging.info(f" LION directory for subject {os.path.basename(subject)} at: {lion_dir}")

        # ORGANISE DATA ACCORDING TO MODALITY
        spinner.text = f'[{i + 1}/{num_subjects}] Organising data according to modality for {os.path.basename(subject)}...'
        file_utilities.organise_files_by_modality([subject], modalities, lion_dir)

        # ORGANIZE IMAGES ACCORDING TO WORKFLOW STAGES
        spinner.text = f'Processing subject {i + 1}/{num_subjects}: Organizing images for {os.path.basename(subject)} ' \
                       f'by workflow.'
        file_utilities.create_model_based_workflows(lion_dir, model_name)

        # PREDICT IMAGES
        start_time = time.time()
        spinner.text = f'[{i + 1}/{num_subjects}] Predicting images for {os.path.basename(subject)}...'
        logging.info(' ')
        logging.info(f'{constants.ANSI_VIOLET} PREDICTING IMAGES:'
                     f'{constants.ANSI_RESET}')
        logging.info(' ')
        segmentation_file = predict_tumor(workflow_dir, model_name, output_dir, accelerator, thresholding)
        # Post-processing the segmentation file
        reference_modality = TRACER_WORKFLOWS[model_name]['reference_modality']
        # get the reference_modality directory from the lionz directory
        reference_modality_dir = os.path.join(lion_dir, reference_modality)
        # get the reference_modality image from the reference_modality directory extension .nii or .nii.gz
        nifti_files = glob.glob(os.path.join(reference_modality_dir, '*.nii*'))
        reference_modality_file = nifti_files[0]
        # resample the segmentation file to the reference_modality image
        post_process(reference_modality_file, segmentation_file, segmentation_file)
        # rename the segmentation file with the subject name as prefix
        os.rename(segmentation_file, os.path.join(output_dir, os.path.basename(subject) + '_tumor_seg.nii.gz'))
        new_segmentation_file = os.path.join(output_dir, os.path.basename(subject) + '_tumor_seg.nii.gz')
        end_time = time.time()
        elapsed_time = end_time - start_time
        spinner.text = f' {constants.ANSI_GREEN}[{i + 1}/{num_subjects}] Prediction done for {os.path.basename(subject)} using {model_name}!' \
                       f' | Elapsed time: {round(elapsed_time / 60, 1)} min{constants.ANSI_RESET}'
        time.sleep(3)
        spinner.text = f'[{i + 1}/{num_subjects}] Calculating fused MIP of PET image and tumor mask for ' \
                       f'{os.path.basename(subject)}...'
        image_processing.create_rotational_mip_gif(reference_modality_file,
                                                   new_segmentation_file,
                                                   os.path.join(output_dir,
                                                                os.path.basename(subject) +
                                                                '_rotational_mip.gif'),
                                                   rotation_step=constants.MIP_ROTATION_STEP,
                                                   output_spacing=constants.MIP_VOXEL_SPACING)
        spinner.text = f'{constants.ANSI_GREEN} [{i + 1}/{num_subjects}] Fused MIP of PET image and tumor mask ' \
                       f'calculated' \
                       f' for {os.path.basename(subject)}! '
        time.sleep(3)

        tumor_volume, average_intensity = image_processing.compute_tumor_metrics(new_segmentation_file,
                                                                                 reference_modality_file)
        # if tumor_volume is zero then the segmentation should have a suffix _no_tumor_seg.nii.gz
        if tumor_volume == 0:
            os.rename(new_segmentation_file, os.path.join(output_dir, os.path.basename(subject) + '_no_tumor_seg.nii.gz'))
        image_processing.save_metrics_to_csv(tumor_volume, average_intensity, os.path.join(stats_dir,
                                                                                           os.path.basename(subject) +
                                                                                           '_metrics.csv'))
    end_total_time = time.time()
    total_elapsed_time = (end_total_time - start_total_time) / 60
    time_per_dataset = total_elapsed_time / len(lion_compliant_subjects)

    spinner.succeed(f'{constants.ANSI_GREEN} All predictions done! | Total elapsed time for '
                    f'{len(lion_compliant_subjects)} datasets: {round(total_elapsed_time, 1)} min'
                    f' | Time per dataset: {round(time_per_dataset, 2)} min {constants.ANSI_RESET}')


def lion(model_name: str, input_dir: str, seg_output_dir: str, accelerator: str, thresholding: bool = False) -> None:
    """
    Execute the LION tumour segmentation process.

    This function carries out the following steps:
    1. Sets the path for model results.
    2. Creates the required directory for the model.
    3. Downloads the model based on the provided `model_name`.
    4. Validates and prepares the input directory to be compatible with nnUNet.
    5. Executes the prediction process.

    :param model_name: The name of the model to be used for predictions. This model will be downloaded and used 
                       for the tumour segmentation process.
    :type model_name: str

    :param input_dir: Path to the directory containing the images (in nifti, either .nii or .nii.gz) to be processed.
    :type input_dir: str

    :param output_dir: Path to the directory where the segmented output will be saved.
    :type output_dir: str

    :param accelerator: Specifies the type of accelerator to be used. Common values include "cpu" and "cuda" for 
                        GPU acceleration.
    :type accelerator: str

    :return: None
    :rtype: None

    :Example:

    >>> lion('fdg', '/path/to/input/images', '/path/to/save/output', 'cuda')

    """
    model_path = constants.NNUNET_RESULTS_FOLDER
    modalities = display.expectations(model_name)
    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    logging.info('- Custom trainer: ' + custom_trainer_status)
    file_utilities.create_directory(model_path)
    download.model(model_name, model_path)

    lion_dir, input_dirs, output_dir, stats_dir, workflow_dir = file_utilities.lion_folder_structure(input_dir,
                                                                                                        model_name,
                                                                                                        modalities)
    file_utilities.organise_files_by_modality([input_dir], modalities, lion_dir)
    file_utilities.create_model_based_workflows(lion_dir, model_name)
    segmentation_file = predict_tumor(workflow_dir, model_name, output_dir, accelerator, thresholding)
    # Post-processing the segmentation file
    reference_modality = TRACER_WORKFLOWS[model_name]['reference_modality']
    # get the reference_modality directory from the lionz directory
    reference_modality_dir = os.path.join(lion_dir, reference_modality)
    # get the reference_modality image from the reference_modality directory extension .nii or .nii.gz
    nifti_files = glob.glob(os.path.join(reference_modality_dir, '*.nii*'))
    reference_modality_file = nifti_files[0]
    # resample the segmentation file to the reference_modality image
    post_process(reference_modality_file, segmentation_file, segmentation_file)
    # rename the segmentation file with the subject name as prefix
    os.rename(segmentation_file, os.path.join(seg_output_dir, os.path.basename(input_dir) + '_tumor_seg.nii.gz'))
    new_segmentation_file = os.path.join(seg_output_dir, os.path.basename(input_dir) + '_tumor_seg.nii.gz')

    # Save some statistics
    tumor_volume, average_intensity = image_processing.compute_tumor_metrics(new_segmentation_file,
                                                                            reference_modality_file)
    if tumor_volume == 0:
        os.rename(new_segmentation_file, os.path.join(seg_output_dir, os.path.basename(input_dir) +
                                                      '_no_tumor_seg.nii.gz'))
    image_processing.save_metrics_to_csv(tumor_volume, average_intensity, os.path.join(stats_dir,
                                                                                        os.path.basename(input_dir) +
                                                                                        '_metrics.csv'))
    

if __name__ == '__main__':
    main()
