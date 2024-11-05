#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ image conversion
---------------

This module contains functions that are responsible for converting images to NIfTI format.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""

import contextlib
import io
import os
import re
import unicodedata
import SimpleITK
import dicom2nifti
import pydicom
from lionz import system
from lionz import file_utilities


def non_nifti_to_nifti(input_path: str, output_directory: str = None) -> None:
    """
    Converts any image format known to ITK to NIFTI

        :param input_path: The path to the directory or filename to convert to nii.gz.
        :type input_path: str

        :param output_directory: Optional. The output directory to write the image to. If not specified, the output image will be written to the same directory as the input image.
        :type output_directory: str

        :return: None
        :rtype: None

        :raises: FileNotFoundError if the input path does not exist.

        Usage:
        This function can be used to convert any image format known to ITK to NIFTI. If the input path is a directory, the function will convert all images in the directory to NIFTI format. If the input path is a file, the function will convert the file to NIFTI format. The output image will be written to the specified output directory, or to the same directory as the input image if no output directory is specified.
    """
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist.")
        return

    # Processing a directory
    if os.path.isdir(input_path):
        dicom_info = create_dicom_lookup(input_path)
        nifti_dir = dcm2niix(input_path)
        rename_and_convert_nifti_files(nifti_dir, dicom_info)
        return

    # Processing a file
    if os.path.isfile(input_path):
        # Ignore hidden or already processed files
        _, filename = os.path.split(input_path)
        if filename.startswith('.') or filename.endswith(('.nii.gz', '.nii')):
            return
        else:
            output_image = SimpleITK.ReadImage(input_path)
            output_image_basename = f"{os.path.splitext(filename)[0]}.nii"

    if output_directory is None:
        output_directory = os.path.dirname(input_path)

    output_image_path = os.path.join(output_directory, output_image_basename)
    SimpleITK.WriteImage(output_image, output_image_path)


def standardize_to_nifti(parent_dir: str, output_manager: system.OutputManager) -> None:
    """
    Converts all non-NIfTI images in a parent directory and its subdirectories to NIfTI format.

    :param parent_dir: The path to the parent directory containing the images to convert.
    :type parent_dir: str
    :param output_manager: The output manager to handle console and log output.
    :type output_manager: system.OutputManager
    :return: None
    """
    # Get a list of all subdirectories in the parent directory
    subjects = os.listdir(parent_dir)
    subjects = [subject for subject in subjects if os.path.isdir(os.path.join(parent_dir, subject))]

    # Convert all non-NIfTI images in each subdirectory to NIfTI format
    progress = output_manager.create_progress_bar()
    with progress:
        task = progress.add_task("[white] Processing subjects...", total=len(subjects))
        for subject in subjects:
            subject_path = os.path.join(parent_dir, subject)
            if os.path.isdir(subject_path):
                images = os.listdir(subject_path)
                for image in images:
                    image_path = os.path.join(subject_path, image)
                    path_is_valid = os.path.isdir(image_path) | os.path.isfile(image_path)
                    path_is_valid = path_is_valid & ("lionz" not in os.path.basename(image_path))
                    if path_is_valid:
                        non_nifti_to_nifti(image_path, output_manager)
            else:
                continue
            progress.update(task, advance=1, description=f"[white] Processing {subject}...")


def dcm2niix(input_path: str) -> str:
    """
    Converts DICOM images into NIfTI images using dcm2niix.

    :param input_path: The path to the folder containing the DICOM files to convert.
    :type input_path: str
    :return: The path to the folder containing the converted NIfTI files.
    :rtype: str
    """
    output_dir = os.path.dirname(input_path)

    # Redirect standard output and standard error to discard output
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dicom2nifti.convert_directory(input_path, output_dir, compression=False, reorient=True)

    return output_dir


def remove_accents(unicode_filename: str) -> str:
    """
    Removes accents and special characters from a Unicode filename.

    :param unicode_filename: The Unicode filename to clean.
    :type unicode_filename: str
    :return: The cleaned filename.
    :rtype: str
    """
    try:
        unicode_filename = str(unicode_filename).replace(" ", "_")
        cleaned_filename = unicodedata.normalize('NFKD', unicode_filename).encode('ASCII', 'ignore').decode('ASCII')
        cleaned_filename = re.sub(r'[^\w\s-]', '', cleaned_filename.strip().lower())
        cleaned_filename = re.sub(r'[-\s]+', '-', cleaned_filename)
        return cleaned_filename
    except:
        return unicode_filename


def is_dicom_file(filename: str) -> bool:
    """
    Checks if a file is a DICOM file.

    :param filename: The path to the file to check.
    :type filename: str
    :return: True if the file is a DICOM file, False otherwise.
    :rtype: bool
    """
    try:
        pydicom.dcmread(filename)
        return True
    except pydicom.errors.InvalidDicomError:
        return False


def create_dicom_lookup(dicom_dir: str) -> dict:
    """
    Create a lookup dictionary from DICOM files.

    :param dicom_dir: The directory where DICOM files are stored.
    :type dicom_dir: str
    :return: A dictionary where the key is the anticipated filename that dicom2nifti will produce and
             the value is the modality of the DICOM series.
    :rtype: dict
    """
    dicom_info = {}

    for filename in os.listdir(dicom_dir):
        full_path = os.path.join(dicom_dir, filename)
        if is_dicom_file(full_path):
            ds = pydicom.dcmread(full_path, force=True)

            series_number = ds.SeriesNumber if 'SeriesNumber' in ds else None
            series_description = ds.SeriesDescription if 'SeriesDescription' in ds else None
            sequence_name = ds.SequenceName if 'SequenceName' in ds else None
            protocol_name = ds.ProtocolName if 'ProtocolName' in ds else None
            series_instance_UID = ds.SeriesInstanceUID if 'SeriesInstanceUID' in ds else None
            modality = ds.Modality
            if modality == "PT":
                suv_parameters = {'weight[kg]': ds.PatientWeight,
                          'total_dose[MBq]': (
                                      float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose) / 1000000)}
                units = ds.Units
                if units == "CNTS":
                    suv_converstion_factor = ds[0x7053, 0x1000].value

                else:
                    suv_converstion_factor = None
            else:
                suv_parameters = None
                units = None
                suv_converstion_factor = None

            if series_number is not None:
                base_filename = remove_accents(series_number)
                if series_description is not None:
                    anticipated_filename = f"{base_filename}_{remove_accents(series_description)}.nii"
                elif sequence_name is not None:
                    anticipated_filename = f"{base_filename}_{remove_accents(sequence_name)}.nii"
                elif protocol_name is not None:
                    anticipated_filename = f"{base_filename}_{remove_accents(protocol_name)}.nii"
            else:
                anticipated_filename = f"{remove_accents(series_instance_UID)}.nii"

            dicom_info[anticipated_filename] = (modality, suv_parameters, units, suv_converstion_factor)

    return dicom_info


def rename_and_convert_nifti_files(nifti_dir: str, dicom_info: dict) -> None:
    """
    Rename NIfTI files based on a lookup dictionary.

    :param nifti_dir: The directory where NIfTI files are stored.
    :type nifti_dir: str
    :param dicom_info: A dictionary where the key is the anticipated filename that dicom2nifti will produce and
                       the value is the modality of the DICOM series.
    :type dicom_info: dict
    """
    for filename in os.listdir(nifti_dir):
        if filename.endswith('.nii'):
            modality, suv_parameters, units, suv_conversion_factor = dicom_info.get(filename, (None, None, None, None))
            if modality:
                new_filename = f"{modality}_{filename}"
                file_path = os.path.join(nifti_dir, filename)
                if suv_parameters:
                    convert_bq_to_suv(file_path, file_path, suv_parameters, units, suv_conversion_factor)
                os.rename(file_path, os.path.join(nifti_dir, new_filename))
                del dicom_info[filename]


def convert_bq_to_suv(bq_image: str, out_suv_image: str, suv_parameters: dict, image_unit: str,
                      suv_scale_factor) -> None:
    """
    Convert a becquerel PET image to SUV image
    :param bq_image: Path to a becquerel PET image to convert to SUV image (can be NRRD, NIFTI, ANALYZE
    :param out_suv_image: Name of the SUV image to be created (preferrably with a path)
    :param suv_parameters: A dictionary with the SUV parameters (weight in kg, dose in mBq)
    :param image_unit: A string indicating the unit of the PET image ('CNTS' or 'BQML')
    :param suv_scale_factor: A number contained in the dicom tag [7053, 1000] for converting CNTS PT images to SUV ones
    """

    if image_unit == 'BQML':
        total_dose = suv_parameters["total_dose[MBq]"]
        suv_denominator = (total_dose / suv_parameters["weight[kg]"]) * 1000  # Units in kBq/mL
        suv_convertor = 1 / suv_denominator
        cmd_to_run = f"{file_utilities.get_c3d_path()} {bq_image} -scale {suv_convertor} -o {out_suv_image}"
        os.system(cmd_to_run)

    elif image_unit == 'CNTS':
        suv_convertor = float(suv_scale_factor)
        cmd_to_run = f"{file_utilities.get_c3d_path()} {bq_image} -scale {suv_convertor} -o {out_suv_image}"
        os.system(cmd_to_run)

    else:
        print('Please specify a conversion factor.')