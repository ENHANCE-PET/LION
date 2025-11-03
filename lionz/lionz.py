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

import logging
import time
from datetime import datetime

import colorama
import emoji
import rich_click as click
import rich_click
import SimpleITK
import multiprocessing as mp
import concurrent.futures
from rich.text import Text

rich_click.USE_MARKDOWN = False
rich_click.SHOW_ARGUMENTS = True
rich_click.SHOW_OPTION_DEFAULTS = True
rich_click.MAX_WIDTH = 100
rich_click.SHOW_METAVARS_COLUMN = False

from lionz import constants
from lionz import file_utilities
from lionz import image_conversion
from lionz import input_validation
from lionz import image_processing
from lionz import system
from lionz import models
from lionz import predict
from lionz.models import (
    AVAILABLE_MODELS,
    MODEL_METADATA,
    KEY_IMAGING_TYPE,
    KEY_MODALITY,
    KEY_REQUIRED_MODALITIES,
    KEY_REQUIRED_PREFIXES,
    KEY_NR_TRAINING,
)


from lionz.nnUNet_custom_trainer.utility import add_custom_trainers_to_local_nnunetv2


def execute_cli(
    main_directory: str,
    model_name: str,
    threshold: float | None,
    verbose_console: bool,
    verbose_log: bool,
    generate_mip: bool,
    lions_pride: int | None,
) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        filename=datetime.now().strftime(f"lionz-v.{constants.VERSION}.%H-%M-%d-%m-%Y.log"),
        filemode="w",
    )
    colorama.init()

    output_manager = system.OutputManager(verbose_console, verbose_log)
    output_manager.display_logo()
    print('')
    output_manager.display_citation()

    parent_folder = os.path.abspath(main_directory)
    selected_model = model_name
    accelerator, device_count = system.check_device(output_manager, announce=False)

    effective_threshold = threshold
    generate_mip_output = generate_mip
    lion_instances = lions_pride

    output_manager.configure_logging(parent_folder)
    output_manager.log_update("----------------------------------------------------------------------------------------------------")
    output_manager.log_update(
        f"                                     STARTING LIONZ-v.{constants.VERSION}                                         "
    )
    output_manager.log_update("----------------------------------------------------------------------------------------------------")

    # ----------------------------------
    # INPUT VALIDATION AND PREPARATION
    # ----------------------------------
    output_manager.log_update(' ')
    output_manager.log_update('- Main directory: ' + parent_folder)
    output_manager.log_update('- Model name: ' + selected_model)
    output_manager.log_update(' ')

    model_identifiers = selected_model if isinstance(selected_model, list) else [selected_model]
    note_text = Text(justify="left")
    for index, model_identifier in enumerate(model_identifiers):
        metadata = MODEL_METADATA.get(model_identifier, {})
        imaging_type = metadata.get(KEY_IMAGING_TYPE, "clin").title()
        modality_label = metadata.get(KEY_MODALITY, "PT")
        required_modalities = metadata.get(KEY_REQUIRED_MODALITIES) or [modality_label]
        required_prefixes = metadata.get(KEY_REQUIRED_PREFIXES) or [modality_label.replace('-', '_') + "_"]
        training_count = metadata.get(KEY_NR_TRAINING, "Not available")

        if index:
            note_text.append("\n", style=constants.CLI_COLORS["muted"])

        note_text.append("Model Name: ", style=constants.CLI_COLORS["accent"])
        note_text.append(f"{model_identifier}\n", style=constants.CLI_COLORS["text"])
        note_text.append("Imaging Type: ", style=constants.CLI_COLORS["accent"])
        note_text.append(f"{imaging_type}\n", style=constants.CLI_COLORS["text"])
        note_text.append("Required Modality: ", style=constants.CLI_COLORS["accent"])
        note_text.append(f"{', '.join(required_modalities)}\n", style=constants.CLI_COLORS["text"])
        note_text.append("Required Prefix (non-DICOM images): ", style=constants.CLI_COLORS["accent"])
        note_text.append(f"{', '.join(required_prefixes)}\n", style=constants.CLI_COLORS["text"])
        note_text.append("Training Datasets: ", style=constants.CLI_COLORS["accent"])
        note_text.append(f"{training_count}", style=constants.CLI_COLORS["text"])

    note_text.append("\n\n", style=constants.CLI_COLORS["muted"])
    note_text.append(
        emoji.emojize("⚠️  Subjects that don't have the required modalities (check file prefix) will be skipped."),
        style=constants.CLI_COLORS["warning"],
    )
    note_text.append("\n", style=constants.CLI_COLORS["muted"])

    if accelerator == "mps":
        device_line = "🍎 Apple MPS backend is available. Predictions will be run on Apple Silicon GPU."
    elif accelerator == "cuda":
        if device_count:
            device_line = emoji.emojize(
                f":high_voltage: CUDA is available with {device_count} GPU(s). Predictions will be run on GPU."
            )
        else:
            device_line = emoji.emojize(
                ":high_voltage: CUDA is available. Predictions will be run on GPU."
            )
    else:
        device_line = emoji.emojize(
            ":gear: CUDA/MPS not available. Predictions will be run on CPU."
        )

    note_text.append(device_line, style=constants.CLI_COLORS["warning"])

    output_manager.context_panel("Note", note_text, icon=":memo:")

    # ------------------------------
    # DOWNLOAD THE MODEL
    # ------------------------------

    output_manager.section("Model Download", ":globe_with_meridians:")
    model_path = system.MODELS_DIRECTORY_PATH
    file_utilities.create_directory(model_path)
    model_routine = models.construct_model_routine(selected_model, output_manager)
    model_count = sum(len(workflows) for workflows in model_routine.values())
    model_count_for_stats = model_count or 1

    custom_trainer_status = add_custom_trainers_to_local_nnunetv2()
    modalities = input_validation.determine_model_expectations(model_routine, output_manager)
    output_manager.log_update('- Custom trainer: ' + custom_trainer_status)
    inputs_valid = input_validation.validate_inputs(parent_folder, selected_model)
    if not inputs_valid:
        raise click.ClickException("Input validation failed.")
    else:
        output_manager.log_update(f"Input validation successful.")

    if lion_instances is not None:
        output_manager.message(
            f"Number of LION instances run in parallel: {lion_instances}",
            style="accent",
            icon=" :lion_face:",
        )


    # ------------------------------
    # INPUT STANDARDIZATION
    # ------------------------------
    output_manager.console.print()
    output_manager.section("Standardizing input data to NIfTI", ":magnifying_glass_tilted_left:")
    output_manager.log_update(' ')
    output_manager.log_update(' STANDARDIZING INPUT DATA TO NIFTI:')
    output_manager.log_update(' ')
    image_conversion.standardize_to_nifti(parent_folder, output_manager)
    output_manager.message(" Standardization complete.", style="success")
    output_manager.log_update(" Standardization complete.")

    # ------------------------------
    # CHECK FOR LIONZ COMPLIANCE
    # ------------------------------

    subjects = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if
                os.path.isdir(os.path.join(parent_folder, d))]
    lion_compliant_subjects = input_validation.select_lion_compliant_subjects(subjects, modalities, output_manager)

    prediction_subjects: list[str] = []
    skipped_subjects: list[str] = []
    for subject_path in lion_compliant_subjects:
        has_expected = False
        for prefix in ('PT_', 'PT-'):
            if file_utilities.get_files(subject_path, prefix, ('.nii', '.nii.gz')):
                has_expected = True
                break
        if has_expected:
            prediction_subjects.append(subject_path)
        else:
            skipped_subjects.append(subject_path)

    for skipped_subject in skipped_subjects:
        subject_name = os.path.basename(skipped_subject)
        message = (
            f"No files matching the expected prefix 'PT_' or 'PT-' were found for subject {subject_name}. "
            "Skipping this subject."
        )
        output_manager.message(message, style="warning", icon=":warning:")
        output_manager.log_update(f"   X {message}")

    if prediction_subjects:
        queued_text = Text(
            " Subjects queued for prediction: ",
            style=f"italic {constants.CLI_COLORS['info']}"
        )
        queued_text.append(
            str(len(prediction_subjects)),
            style=f"bold {constants.CLI_COLORS['accent']}"
        )
        output_manager.console.print(queued_text)

    num_subjects = len(prediction_subjects)
    if num_subjects < 1:
        output_manager.message(
            "No LION compliant subject found to continue!",
            style="error",
            icon=" :cross_mark:",
            emphasis=True,
        )
        output_manager.message(
            "See: https://github.com/LalithShiyam/LION#directory-conventions-for-lion-%EF%B8%8F",
            style="info",
            icon=" :light_bulb:",
        )
        return

    # -------------------------------------------------
    # RUN PREDICTION ONLY FOR LION COMPLIANT SUBJECTS
    # -------------------------------------------------
    output_manager.console.print()
    output_manager.section("Prediction", ":crystal_ball:")
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

        compliant_count = len(prediction_subjects)
        if device_count is not None and device_count > 1:
            accelerator_assignments = [f"{accelerator}:{i % device_count}" for i in range(compliant_count)]
        else:
            accelerator_assignments = [accelerator] * compliant_count

        with concurrent.futures.ProcessPoolExecutor(max_workers=lion_instances, mp_context=mp_context) as executor:
            futures = []
            for i, (subject, accelerator) in enumerate(zip(prediction_subjects, accelerator_assignments)):
                futures.append(
                    executor.submit(
                        lion_subject,
                        subject,
                        i,
                        num_subjects,
                        model_routine,
                        accelerator,
                        None,
                        effective_threshold,
                        generate_mip_output,
                    )
                )

            for _ in concurrent.futures.as_completed(futures):
                processed_subjects += 1
                output_manager.spinner_update(f'[{processed_subjects}/{num_subjects}] subjects processed.')

    else:
        for i, subject in enumerate(prediction_subjects):
            lion_subject(
                subject,
                i,
                num_subjects,
                model_routine,
                accelerator,
                output_manager,
                effective_threshold,
                generate_mip_output,
            )

    end_total_time = time.time()
    total_elapsed_time = (end_total_time - start_total_time) / 60
    processed_datasets = max(len(prediction_subjects), 1)
    time_per_dataset = total_elapsed_time / processed_datasets
    time_per_model = time_per_dataset / model_count_for_stats

    completion_message = (
        f"All predictions done. Total time: {round(total_elapsed_time, 1)} min "
        f"(per dataset: {round(time_per_dataset, 2)} min)."
    )
    output_manager.spinner_succeed(completion_message)
    output_manager.log_update(f' ')
    output_manager.log_update(f' ALL SUBJECTS PROCESSED')
    output_manager.log_update(f'  - Number of Subjects: {len(prediction_subjects)}')
    output_manager.log_update(f'  - Number of Models:   {model_count}')
    output_manager.log_update(f'  - Time (total):       {round(total_elapsed_time, 1)}min')
    output_manager.log_update(f'  - Time (per subject): {round(time_per_dataset, 2)}min')
    output_manager.log_update(f'  - Time (per model):   {round(time_per_model, 2)}min')

    output_manager.log_update('----------------------------------------------------------------------------------------------------')
    output_manager.log_update(f'                                     FINISHED LION-Z V.{constants.VERSION}                                       ')
    output_manager.log_update('----------------------------------------------------------------------------------------------------')


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=constants.USAGE_MESSAGE.strip(),
)
@click.option(
    "-d",
    "--main-directory",
    "main_directory",
    type=click.Path(path_type=str),
    required=True,
    metavar="<MAIN_DIRECTORY>",
    help="Specify the main directory containing subject folders.",
)
@click.option(
    "-m",
    "--model-name",
    "model_name",
    type=click.Choice(sorted(AVAILABLE_MODELS), case_sensitive=False),
    required=True,
    metavar="<MODEL_NAME>",
    help="Select the model to run.",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=None,
    show_default=False,
    help="Threshold value applied to the tumor segmentations.",
)
@click.option(
    "-v-off",
    "--verbose-off",
    "verbose_off",
    is_flag=True,
    default=False,
    help="Deactivate verbose console output.",
)
@click.option(
    "-log-off",
    "--logging-off",
    "logging_off",
    is_flag=True,
    default=False,
    help="Deactivate logging.",
)
@click.option(
    "-g",
    "--generate-mip",
    "generate_mip",
    is_flag=True,
    default=False,
    help="Generate rotational MIP previews alongside segmentations.",
)
@click.option(
    "-p",
    "--lions-pride",
    "lions_pride",
    type=int,
    default=None,
    metavar="<JOBS>",
    help="Number of concurrent jobs (set to 2 or more to enable parallel execution).",
)
def main(
    main_directory: str,
    model_name: str,
    threshold: float | None,
    verbose_off: bool,
    logging_off: bool,
    generate_mip: bool,
    lions_pride: int | None,
) -> None:
    """
    LIONZ (Lesion segmentatION) — precise tumor segmentation for PET/CT datasets.
    """
    normalized_model = model_name.lower()
    verbose_console = not verbose_off
    verbose_log = not logging_off
    try:
        execute_cli(
            main_directory=main_directory,
            model_name=normalized_model,
            threshold=threshold,
            verbose_console=verbose_console,
            verbose_log=verbose_log,
            generate_mip=generate_mip,
            lions_pride=lions_pride,
        )
    except click.ClickException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise click.ClickException(str(exc)) from exc


def lion(input_data: str | tuple[numpy.ndarray, tuple[float, float, float]],
         model_name: str, output_dir: str = None, accelerator: str = None, threshold: float | None = None) -> str | numpy.ndarray | SimpleITK.Image:
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

    :param threshold: Optional threshold value applied to the resulting segmentations.
    :type threshold: Optional[float]

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
                  output_manager: system.OutputManager | None, threshold: float | None = None, generate_mip: bool = False):
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

    modality_files = []
    for prefix in ('PT_', 'PT-'):
        modality_files = file_utilities.get_files(subject, prefix, ('.nii', '.nii.gz'))
        if modality_files:
            break

    if not modality_files:
        message = (
            f"No files matching the expected prefix 'PT_' or 'PT-' were found for subject {subject_name}. "
            "Skipping this subject."
        )
        output_manager.message(message, style="warning", icon=":warning:")
        output_manager.log_update(f"   X {message}")
        return subject_peak_performance

    file_path = modality_files[0]
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

            if threshold is not None:
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

                output_manager.spinner_update(
                    f"[{subject_index + 1}/{number_of_subjects}] Fused MIP of PET image and tumor mask calculated "
                    f"for {os.path.basename(subject)}!"
                )
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
        f"[{subject_index + 1}/{number_of_subjects}] Prediction done for {subject_name} using {len(model_names)} model(s)"
        f" | Elapsed time: {round(elapsed_time / 60, 1)} min"
    )
    time.sleep(1)
    output_manager.log_update(
        f' Prediction done for {subject_name} using {len(model_names)} model: {", ".join(model_names)}!'
        f' | Elapsed time: {round(elapsed_time / 60, 1)} min')

    return subject_peak_performance


if __name__ == '__main__':
    main()
