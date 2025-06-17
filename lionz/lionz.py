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
import os

import numpy

os.environ["nnUNet_raw"] = ""
os.environ["nnUNet_preprocessed"] = ""
os.environ["nnUNet_results"] = ""

import argparse
import logging
import time
from datetime import datetime

import colorama
import emoji
import SimpleITK
import multiprocessing as mp
import concurrent.futures

from lionz import constants
from lionz import file_utilities
from lionz import image_conversion
from lionz import input_validation
from lionz import image_processing
from lionz import system
from lionz import models
from lionz import predict
from lionz.models import AVAILABLE_MODELS


from lionz.nnUNet_custom_trainer.utility import add_custom_trainers_to_local_nnunetv2


# Main function for the module
def main():
    logging.basicConfig(format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO,
                        filename=datetime.now().strftime('lionz-v.0.10.0.%H-%M-%d-%m-%Y.log'), filemode='w')
    colorama.init()

    # Argument parser
    parser = argparse.ArgumentParser(
        description=constants.USAGE_MESSAGE,
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
        "-t", "--threshold",
        required=False,
        type=float,
        default=False,
        help="Use to define a threshold value and apply to the tumor segmentations"
    )

    parser.add_argument(
        "-v-off", "--verbose_off",
        action="store_false",
        help="Deactivate verbose console."
    )

    parser.add_argument(
        "-log-off", "--logging_off",
        action="store_false",
        help="Deactivate logging."
    )

    parser.add_argument(
        "-gen-mip", "--generate_mip",
        required=False,
        default=False,
        action="store_true",
        help="Use to skip mip creation"
    )

    parser.add_argument(
        '-pride', '--lions_pride',
        nargs='?',
        const=2,
        type=int,
        help='Specify the concurrent jobs (default: 2)'
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

    verbose_console = args.verbose_off
    verbose_log = args.logging_off

    output_manager = system.OutputManager(verbose_console, verbose_log)
    output_manager.display_logo()
    output_manager.display_authors()
    output_manager.display_citation()

    # Get the main directory and model name
    parent_folder = os.path.abspath(args.main_directory)
    model_name = args.model_name

    # Check for thresholding
    threshold = args.threshold

    # Check for mip generation
    generate_mip = args.generate_mip

    lion_instances = args.lions_pride

    output_manager.configure_logging(parent_folder)
    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update('                                     STARTING LIONZ-v.0.10.0                                         ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')

    # ------------------------------
    # DOWNLOAD THE MODEL
    # ------------------------------

    output_manager.console_update('')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":globe_with_meridians:")} MODEL DOWNLOAD:{constants.ANSI_RESET}')
    output_manager.console_update('')
    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(model_name, output_manager)

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------

    output_manager.log_update(' ')
    output_manager.log_update('- Main directory: ' + parent_folder)
    output_manager.log_update('- Model name: ' + model_name)
    output_manager.log_update(' ')
    output_manager.console_update(' ')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":memo:")} NOTE:{constants.ANSI_RESET}')
    output_manager.console_update(' ')

    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    modalities = input_validation.determine_model_expectations(model_routine, output_manager)
    output_manager.log_update('- Custom trainer: ' + custom_trainer_status)
    accelerator, device_count = system.check_device()
    inputs_valid = input_validation.validate_inputs(parent_folder, model_name)
    if not inputs_valid:
        exit(1)
    else:
        output_manager.log_update(f"Input validation successful.")

    if lion_instances is not None:
        output_manager.console_update(f" Number of lion instances run in parallel: {lion_instances}")


    # ------------------------------
    # INPUT STANDARDIZATION
    # ------------------------------
    output_manager.console_update('')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":magnifying_glass_tilted_left:")} STANDARDIZING INPUT DATA TO NIFTI:{constants.ANSI_RESET}')
    output_manager.console_update('')
    output_manager.log_update(' ')
    output_manager.log_update(' STANDARDIZING INPUT DATA TO NIFTI:')
    output_manager.log_update(' ')
    image_conversion.standardize_to_nifti(parent_folder, output_manager)
    output_manager.console_update(f"{constants.ANSI_GREEN} Standardization complete.{constants.ANSI_RESET}")
    output_manager.log_update(" Standardization complete.")

    # ------------------------------
    # CHECK FOR LIONZ COMPLIANCE
    # ------------------------------

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if
                os.path.isdir(os.path.join(parent_folder, d))]
    lion_compliant_subjects = input_validation.select_lion_compliant_subjects(subjects, modalities, output_manager)

    num_subjects = len(lion_compliant_subjects)
    if num_subjects < 1:
        print(f'{constants.ANSI_RED} {emoji.emojize(":cross_mark:")} No lion compliant subject found to continue!{constants.ANSI_RESET} {emoji.emojize(":light_bulb:")} See: https://github.com/LalithShiyam/LION#directory-conventions-for-lion-%EF%B8%8F')
        return

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR LION COMPLIANT SUBJECTS
    # -------------------------------------------------

    output_manager.console_update('')
    output_manager.console_update(f'{constants.ANSI_VIOLET} {emoji.emojize(":crystal_ball:")} PREDICT:{constants.ANSI_RESET}')
    output_manager.console_update('')
    output_manager.log_update(' ')
    output_manager.log_update(' PERFORMING PREDICTION:')
    output_manager.log_update(' ')

    output_manager.spinner_start()
    start_total_time = time.time()

    if lion_instances is not None:
        output_manager.log_update(f"- Branching out with {lion_instances} concurrent jobs.")

        mp_context = mp.get_context('spawn')
        processed_subjects = 0
        output_manager.spinner_update(f'[{processed_subjects}/{num_subjects}] subjects processed.')

        if device_count is not None and device_count > 1:
            accelerator_assignments = [f"{accelerator}:{i % device_count}" for i in range(len(subjects))]
        else:
            accelerator_assignments = [accelerator] * len(subjects)

        with concurrent.futures.ProcessPoolExecutor(max_workers=lion_instances, mp_context=mp_context) as executor:
            futures = []
            for i, (subject, accelerator) in enumerate(zip(lion_compliant_subjects, accelerator_assignments)):
                futures.append(executor.submit(lion_subject, subject, i, num_subjects, model_routine, accelerator, None, threshold, generate_mip))

            for _ in concurrent.futures.as_completed(futures):
                processed_subjects += 1
                output_manager.spinner_update(f'[{processed_subjects}/{num_subjects}] subjects processed.')

    else:
        for i, subject in enumerate(lion_compliant_subjects):
            lion_subject(subject, i, num_subjects, model_routine, accelerator, output_manager, threshold, generate_mip)

    end_total_time = time.time()
    total_elapsed_time = (end_total_time - start_total_time) / 60
    time_per_dataset = total_elapsed_time / len(lion_compliant_subjects)
    time_per_model = time_per_dataset / len(model_name)

    output_manager.spinner_succeed(f'{constants.ANSI_GREEN} All predictions done! | Total elapsed time for '
                                   f'{len(lion_compliant_subjects)} datasets: {round(total_elapsed_time, 1)} min'
                                   f' | Time per dataset: {round(time_per_dataset, 2)} min')
    output_manager.log_update(f' ')
    output_manager.log_update(f' ALL SUBJECTS PROCESSED')
    output_manager.log_update(f'  - Number of Subjects: {len(lion_compliant_subjects)}')
    output_manager.log_update(f'  - Number of Models:   {len(model_name)}')
    output_manager.log_update(f'  - Time (total):       {round(total_elapsed_time, 1)}min')
    output_manager.log_update(f'  - Time (per subject): {round(time_per_dataset, 2)}min')
    output_manager.log_update(f'  - Time (per model):   {round(time_per_model, 2)}min')

    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update('                                     FINISHED LION-Z V.0.10.0                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')


def lion(input_data: str | tuple[numpy.ndarray, tuple[float, float, float]],
         model_name: str, output_dir: str = None, accelerator: str = None, threshold: int = False) -> str | numpy.ndarray | SimpleITK.Image:
    """
    Execute the LION tumour segmentation process.

    :param input_data: The input data to process, which can be one of the following:
                       - str: A file path to a NIfTI file.
                       - tuple[numpy.ndarray, tuple[float, float, float]]: A tuple containing a numpy array and spacing.
                       - SimpleITK.Image: An image object to process.

    :param model_name: The name(s) of the model(s) to be used for segmentation.
    :type model_name: str or list[str]

    :param output_dir: Path to the directory where the output will be saved if the input is a file path.
    :type output_dir: Optional[str]

    :param accelerator: Specifies the accelerator type, e.g., "cpu" or "cuda".
    :type accelerator: Optional[str]

    :return: The output type aligns with the input type:
             - str (file path): If `input_data` is a file path.
             - SimpleITK.Image: If `input_data` is a SimpleITK.Image.
             - numpy.ndarray: If `input_data` is a numpy array.
    :rtype: str or SimpleITK.Image or numpy.ndarray

    :Example:

    >>> lion('/path/to/input/images', 'model_name', '/path/to/save/output', 'cuda', threshold)
    >>> lion((numpy_array, (3, 3, 3)), 'model_name', '/path/to/save/output', 'cuda', threshold)
    >>> lion(simple_itk_image, 'model_name', '/path/to/save/output', 'cuda', threshold)

    """
    # Load the image and set a default filename based on input type
    if isinstance(input_data, str):
        image = SimpleITK.ReadImage(input_data)
        file_name = file_utilities.get_nifti_file_stem(input_data)
    elif isinstance(input_data, SimpleITK.Image):
        image = input_data
        file_name = 'image_from_simpleitk'
    elif isinstance(input_data, tuple) and isinstance(input_data[0], numpy.ndarray) and isinstance(input_data[1], tuple):
        numpy_array, spacing = input_data
        image = SimpleITK.GetImageFromArray(numpy_array)
        image.SetSpacing(spacing)
        file_name = 'image_from_array'
    else:
        raise ValueError(
            "Invalid input format. `input_data` must be either a file path (str), "
            "a SimpleITK.Image, or a tuple (numpy array, spacing)."
        )
    output_manager = system.OutputManager(False, False)

    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(model_name, output_manager)

    for desired_spacing, model_workflows in model_routine.items():
        resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline',
                                                                                              desired_spacing)
        for model_workflow in model_workflows:
            segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model_workflow[0], accelerator,
                                                                        os.devnull, threshold)

            segmentation = SimpleITK.GetImageFromArray(segmentation_array)
            segmentation.SetSpacing(desired_spacing)
            segmentation.SetOrigin(image.GetOrigin())
            segmentation.SetDirection(image.GetDirection())
            resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

            # Return based on input type
            if isinstance(input_data, str):  # Return file path if input was a file path
                if output_dir is None:
                    output_dir = os.path.dirname(input_data)
                segmentation_image_path = os.path.join(
                    output_dir, f"{model_workflow.target_model.multilabel_prefix}segmentation_{file_name}.nii.gz"
                )
                SimpleITK.WriteImage(resampled_segmentation, segmentation_image_path)
                return segmentation_image_path
            elif isinstance(input_data, SimpleITK.Image):  # Return SimpleITK.Image if input was SimpleITK.Image
                return resampled_segmentation
            elif isinstance(input_data, tuple):  # Return numpy array if input was numpy array
                return SimpleITK.GetArrayFromImage(resampled_segmentation)


def lion_subject(subject: str, subject_index: int, number_of_subjects: int, model_routine: dict, accelerator: str,
                  output_manager: system.OutputManager | None, threshold: int = None, generate_mip: bool = False):
    # SETTING UP DIRECTORY STRUCTURE
    subject_name = os.path.basename(subject)

    if output_manager is None:
        output_manager = system.OutputManager(False, False)

    output_manager.log_update(' ')
    output_manager.log_update(f' SUBJECT: {subject_name}')

    model_names = []
    for workflows in model_routine.values():
        for workflow in workflows:
            model_names.append(workflow.target_model.model_identifier)

    subject_peak_performance = None

    output_manager.spinner_update(
        f'[{subject_index + 1}/{number_of_subjects}] Setting up directory structure for {subject_name}...')
    output_manager.log_update(' ')
    output_manager.log_update(f' SETTING UP LION-Z DIRECTORY:')
    output_manager.log_update(' ')
    lion_dir, segmentations_dir, stats_dir = file_utilities.lion_folder_structure(subject)
    output_manager.log_update(f" LION directory for subject {subject_name} at: {lion_dir}")

    # RUN PREDICTION
    start_time = time.time()
    output_manager.log_update(' ')
    output_manager.log_update(' RUNNING PREDICTION:')
    output_manager.log_update(' ')

    file_path = file_utilities.get_files(subject, 'PT_', ('.nii', '.nii.gz'))[0]
    image = SimpleITK.ReadImage(file_path)
    file_name = file_utilities.get_nifti_file_stem(file_path)

    for desired_spacing, model_workflows in model_routine.items():
        resampling_time_start = time.time()
        resampled_array = image_processing.ImageResampler.resample_image_SimpleITK_DASK_array(image, 'bspline', desired_spacing)
        output_manager.log_update(
            f' - Resampling at {"x".join(map(str, desired_spacing))} took: {round((time.time() - resampling_time_start), 2)}s')

        for model_workflow in model_workflows:
            # ----------------------------------
            # RUN MODEL WORKFLOW
            # ----------------------------------
            model_time_start = time.time()
            output_manager.spinner_update(
                f'[{subject_index + 1}/{number_of_subjects}] Running prediction for {subject_name} using {model_workflow[0]}...')
            output_manager.log_update(f'   - Model {model_workflow.target_model}')
            segmentation_array = predict.predict_from_array_by_iterator(resampled_array, model_workflow[0],
                                                                        accelerator,
                                                                        output_manager.nnunet_log_filename)

            segmentation = SimpleITK.GetImageFromArray(segmentation_array)
            segmentation.SetSpacing(desired_spacing)
            segmentation.SetOrigin(image.GetOrigin())
            segmentation.SetDirection(image.GetDirection())
            resampled_segmentation = image_processing.ImageResampler.resample_segmentation(image, segmentation)

            if threshold:
                resampled_segmentation = image_processing.threshold_segmentation_sitk(image, resampled_segmentation, threshold)

            segmentation_image_path = os.path.join(segmentations_dir,
                                                   f"{file_name}_tumor_seg.nii.gz")
            output_manager.log_update(f'     - Writing segmentation for {model_workflow.target_model}')
            SimpleITK.WriteImage(resampled_segmentation, segmentation_image_path)
            output_manager.log_update(
                f"     - Prediction complete for {model_workflow.target_model} within {round((time.time() - model_time_start) / 60, 1)} min.")

            # ----------------------------------
            # CREATING MIP
            # ----------------------------------
            if generate_mip:
                output_manager.spinner_update(f'[{subject_index + 1}/{number_of_subjects}] Calculating fused MIP of PET image and tumor mask for {os.path.basename(subject)}...')

                image_processing.create_rotational_mip_gif(image,
                                                           resampled_segmentation,
                                                           os.path.join(segmentations_dir,
                                                                        os.path.basename(subject) +
                                                                        '_rotational_mip.gif'),
                                                           output_manager,
                                                           rotation_step=constants.MIP_ROTATION_STEP,
                                                           output_spacing=constants.MIP_VOXEL_SPACING)

                output_manager.spinner_update(f'{constants.ANSI_GREEN} [{subject_index + 1}/{number_of_subjects}] Fused MIP of PET image and tumor mask '
                               f'calculated'
                               f' for {os.path.basename(subject)}! ')
                time.sleep(3)

            # ----------------------------------
            # EXTRACT TUMOR METRICS
            # ----------------------------------

            tumor_volume, average_intensity = image_processing.compute_tumor_metrics(file_path,
                                                                                     segmentation_image_path, output_manager)
            # if tumor_volume is zero then the segmentation should have a suffix _no_tumor_seg.nii.gz
            if tumor_volume == 0:
                os.rename(segmentation_image_path,
                          os.path.join(segmentations_dir, os.path.basename(subject) + '_no_tumor_seg.nii.gz'))
            image_processing.save_metrics_to_csv(tumor_volume, average_intensity, os.path.join(stats_dir,
                                                                                               os.path.basename(
                                                                                                   subject) +
                                                                                               '_metrics.csv'))


    end_time = time.time()
    elapsed_time = end_time - start_time
    output_manager.spinner_update(
        f' {constants.ANSI_GREEN}[{subject_index + 1}/{number_of_subjects}] Prediction done for {subject_name} using {len(model_names)} model: '
        f' | Elapsed time: {round(elapsed_time / 60, 1)} min{constants.ANSI_RESET}')
    time.sleep(1)
    output_manager.log_update(
        f' Prediction done for {subject_name} using {len(model_names)} model: {", ".join(model_names)}!'
        f' | Elapsed time: {round(elapsed_time / 60, 1)} min')

    return subject_peak_performance


if __name__ == '__main__':
    main()
