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
import multiprocessing as mp
import concurrent.futures
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


def standardize_subject(parent_dir: str, subject: str):
    subject_path = os.path.join(parent_dir, subject)
    if os.path.isdir(subject_path):
        images = os.listdir(subject_path)
        for image in images:
            image_path = os.path.join(subject_path, image)
            path_is_valid = os.path.isdir(image_path) or os.path.isfile(image_path)
            path_is_valid = path_is_valid and ("moosez" not in os.path.basename(image_path))
            if path_is_valid:
                non_nifti_to_nifti(image_path)
    else:
        return


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

        mp_context = mp.get_context('spawn')
        max_workers = mp.cpu_count()//4 if mp.cpu_count() > 4 else 1
        max_workers = max_workers if max_workers <= 32 else 32
        output_manager.log_update(f"Number of workers: {max_workers}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
            futures = []
            for subject in subjects:
                futures.append(executor.submit(standardize_subject, parent_dir, subject))

            for _ in concurrent.futures.as_completed(futures):
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
                DICOM_PET_parameters = get_DICOM_PET_parameters(full_path)
                corrected_activity = compute_corrected_activity(DICOM_PET_parameters)
                suv_parameters = {'weight[kg]': ds.PatientWeight,
                                  'total_dose[MBq]': (float(ds.RadiopharmaceuticalInformationSequence[0].RadionuclideTotalDose) / 1000000),
                                  'total_dose[MBq]_corrected': corrected_activity / 1000000}
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


def convert_bq_to_suv(bq_image: str, out_suv_image: str, suv_parameters: dict, image_unit: str, suv_scale_factor) -> None:
    """
    Convert a becquerel or counts PET image to SUV image using SimpleITK.
    :param bq_image: Path to a becquerel PET image to convert to SUV image (can be NRRD, NIFTI, ANALYZE)
    :param out_suv_image: Name of the SUV image to be created (preferably with a path)
    :param suv_parameters: A dictionary with the SUV parameters (weight in kg, dose in mBq)
    :param image_unit: A string indicating the unit of the PET image ('CNTS' or 'BQML')
    :param suv_scale_factor: A number contained in the DICOM tag [7053,1000] for converting CNTS PET images to SUV
    """
    # Read input image
    image = SimpleITK.ReadImage(bq_image, SimpleITK.sitkFloat32)

    # Determine the scale factor for BQML images
    if image_unit.upper() == 'BQML':
        # use corrected dose if available, else raw dose
        total_dose = (
            suv_parameters.get("total_dose[MBq]_corrected")
            or suv_parameters["total_dose[MBq]"]
        )
        weight = suv_parameters["weight[kg]"]
        # denominator in kBq/mL
        suv_denominator = (total_dose / weight) * 1000.0
        suv_convertor = 1.0 / suv_denominator

    # Or for counts images, use the provided scale factor
    elif image_unit.upper() == 'CNTS':
        suv_convertor = float(suv_scale_factor)

    else:
        raise ValueError(f"Unsupported image unit: '{image_unit}'")

    # Apply scaling via a ShiftScale filter
    ss_filter = SimpleITK.ShiftScaleImageFilter()
    ss_filter.SetShift(0.0)
    ss_filter.SetScale(suv_convertor)
    suv_image = ss_filter.Execute(image)

    # Write out the SUV image
    SimpleITK.WriteImage(suv_image, out_suv_image)


# SUV computation is based on the guidelines of the Quantitative Imaging Biomarkers Alliance, mainly taken from:
# - https://qibawiki.rsna.org/index.php/Standardized_Uptake_Value_(SUV)
# - https://qibawiki.rsna.org/images/6/62/SUV_vendorneutral_pseudocode_happypathonly_20180626_DAC.pdf
def get_DICOM_PET_parameters(dicom_file_path: str) -> dict:
    """
    Get DICOM parameters from DICOM tags using pydicom
    :param dicom_file_path: Path to the DICOM file to get the SUV parameters from
    :return: DICOM_parameters, a dictionary with DICOM parameters
    """
    ds = pydicom.dcmread(dicom_file_path, stop_before_pixels=True)
    DICOM_parameters = {'PatientWeight': tag_to_float(ds.get('PatientWeight', None)),
                        'AcquisitionDate': ds.get('AcquisitionDate', None),
                        'AcquisitionTime': ds.get('AcquisitionTime', None),
                        'SeriesTime': ds.get('SeriesTime', None),
                        'DecayFactor': tag_to_float(ds.get('DecayFactor', None)),
                        'DecayCorrection': ds.get('DecayCorrection', None),
                        'RadionuclideTotalDose': None,
                        'RadiopharmaceuticalStartTime': None,
                        'RadionuclideHalfLife': None}

    if 'RadiopharmaceuticalInformationSequence' in ds:
        radiopharmaceutical_information = ds.RadiopharmaceuticalInformationSequence[0]
        DICOM_parameters['RadionuclideTotalDose'] = tag_to_float(radiopharmaceutical_information.get('RadionuclideTotalDose', None))
        DICOM_parameters['RadiopharmaceuticalStartTime'] = radiopharmaceutical_information.get('RadiopharmaceuticalStartTime', None)
        DICOM_parameters['RadionuclideHalfLife'] = tag_to_float(radiopharmaceutical_information.get('RadionuclideHalfLife', None))
    return DICOM_parameters


def tag_to_float(tag: str) -> float | None:
    if tag is None:
        return None
    return float(tag)


def tag_to_time_seconds(tag: str) -> int | None:
    if tag is None:
        return None
    time = tag.split('.')[0]
    hours, minutes, seconds = int(time[0:2]), int(time[2:4]), int(time[4:6])
    time_seconds = hours * 3600 + minutes * 60 + seconds
    return time_seconds


def get_time_difference_seconds(time_1: str, time_2: str) -> int | None:
    time_1_seconds = tag_to_time_seconds(time_1)
    time_2_seconds = tag_to_time_seconds(time_2)
    if time_1_seconds is None or time_2_seconds is None:
        return None

    time_difference_seconds = time_1_seconds - time_2_seconds
    return time_difference_seconds


def compute_corrected_activity(patient_parameters: dict) -> float | None:
    radiopharmaceutical_start_time = patient_parameters['RadiopharmaceuticalStartTime']
    series_time = patient_parameters['SeriesTime']
    injection_to_scan_time = get_time_difference_seconds(series_time, radiopharmaceutical_start_time)
    radionuclide_total_dose = patient_parameters['RadionuclideTotalDose']
    radionuclide_half_life = patient_parameters['RadionuclideHalfLife']

    if injection_to_scan_time is None or radionuclide_half_life is None:
        if radionuclide_total_dose is None:
            return None
        else:
            return radionuclide_total_dose

    decay_corrected_activity = radionuclide_total_dose * pow(2.0, -(injection_to_scan_time / radionuclide_half_life))
    return decay_corrected_activity