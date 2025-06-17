#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ image processing
---------------

This module contains functions that are responsible for image processing in LION.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""
import os
import csv
import SimpleITK as sitk
import dask.array as da
import numpy as np
import cv2
import imageio


from scipy.ndimage import rotate
from skimage import exposure
from dask.distributed import Client

from lionz.constants import CHUNK_THRESHOLD_RESAMPLING, FRAME_DURATION
from lionz import system
from typing import Tuple


class ImageResampler:
    @staticmethod
    def chunk_along_axis(axis: int) -> int:
        """
        Determines the maximum number of evenly-sized chunks that the axis can be split into.
        Each chunk is at least of size CHUNK_THRESHOLD.

        :param axis: Length of the axis.
        :type axis: int
        :return: The maximum number of evenly-sized chunks.
        :rtype: int
        :raises ValueError: If axis is negative or if CHUNK_THRESHOLD is less than or equal to 0.
        """
        # Check for negative input values
        if axis < 0:
            raise ValueError('Axis must be non-negative')

        if CHUNK_THRESHOLD_RESAMPLING <= 0:
            raise ValueError('CHUNK_THRESHOLD must be greater than 0')

        # If the axis is smaller than the threshold, it cannot be split into smaller chunks
        if axis < CHUNK_THRESHOLD_RESAMPLING:
            return 1

        # Determine the maximum number of chunks that the axis can be split into
        split = axis // CHUNK_THRESHOLD_RESAMPLING

        # Reduce the number of chunks until axis is evenly divisible by split
        while axis % split != 0:
            split -= 1

        return split

    @staticmethod
    def resample_chunk_SimpleITK(image_chunk: da.array, input_spacing: tuple, interpolation_method: int,
                                 output_spacing: tuple, output_size: tuple) -> da.array:
        """
        Resamples a dask array chunk.

        :param image_chunk: The chunk (part of an image) to be resampled.
        :type image_chunk: da.array
        :param input_spacing: The original spacing of the chunk (part of an image).
        :type input_spacing: tuple
        :param interpolation_method: SimpleITK interpolation type.
        :type interpolation_method: int
        :param output_spacing: Spacing of the newly resampled chunk.
        :type output_spacing: tuple
        :param output_size: Size of the newly resampled chunk.
        :type output_size: tuple
        :return: The resampled chunk (part of an image).
        :rtype: da.array
        """
        sitk_image_chunk = sitk.GetImageFromArray(image_chunk)
        sitk_image_chunk.SetSpacing(input_spacing)
        input_size = sitk_image_chunk.GetSize()

        if all(x == 0 for x in input_size):
            return image_chunk

        resampled_sitk_image = sitk.Resample(sitk_image_chunk, output_size, sitk.Transform(),
                                             interpolation_method,
                                             sitk_image_chunk.GetOrigin(), output_spacing,
                                             sitk_image_chunk.GetDirection(), 0.0, sitk_image_chunk.GetPixelIDValue())

        resampled_array = sitk.GetArrayFromImage(resampled_sitk_image)
        return resampled_array


    @staticmethod
    def resample_image_SimpleITK_DASK_array(sitk_image: sitk.Image, interpolation: str,
                                            output_spacing: tuple = (1.5, 1.5, 1.5),
                                            output_size: tuple = None) -> np.array:
        if interpolation == 'nearest':
            interpolation_method = sitk.sitkNearestNeighbor
        elif interpolation == 'linear':
            interpolation_method = sitk.sitkLinear
        elif interpolation == 'bspline':
            interpolation_method = sitk.sitkBSpline
        else:
            raise ValueError('The interpolation method is not supported.')
        input_spacing = sitk_image.GetSpacing()
        input_size = sitk_image.GetSize()
        input_chunks = [axis / ImageResampler.chunk_along_axis(axis) for axis in input_size]
        input_chunks_reversed = list(reversed(input_chunks))
        image_dask = da.from_array(sitk.GetArrayViewFromImage(sitk_image), chunks=input_chunks_reversed)
        if output_size is not None:
            output_spacing = [input_spacing[i] * (input_size[i] / output_size[i]) for i in range(len(input_size))]
        output_chunks = [round(input_chunks[i] * (input_spacing[i] / output_spacing[i])) for i in
                         range(len(input_chunks))]
        output_chunks_reversed = list(reversed(output_chunks))
        result = da.map_blocks(ImageResampler.resample_chunk_SimpleITK, image_dask, input_spacing, interpolation_method,
                               output_spacing, output_chunks, chunks=output_chunks_reversed, meta=np.array(()),
                               dtype=np.float32)
        return result.compute()


    @staticmethod
    def resample_segmentation(reference_image: sitk.Image, segmentation_image: sitk.Image):
        resampled_sitk_image = sitk.Resample(segmentation_image, reference_image.GetSize(), sitk.Transform(),
                                             sitk.sitkNearestNeighbor,
                                             reference_image.GetOrigin(), reference_image.GetSpacing(),
                                             reference_image.GetDirection(), 0.0, segmentation_image.GetPixelIDValue())
        return resampled_sitk_image


def mip_3d(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Creates a Maximum Intensity Projection (MIP) of a 3D image.

    :param img: The input image.
    :type img: numpy.ndarray
    :param angle: The angle to rotate the image by.
    :type angle: float
    :return: The MIP of the rotated image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    # Rotate the image
    rot_img = rotate(img, angle, axes=(1, 2), reshape=False)

    # Create Maximum Intensity Projection along the first axis
    mip = np.max(rot_img, axis=1)

    # Invert the mip
    mip_inverted = np.max(mip) - mip

    # Rotate MIP 90 degrees anti-clockwise
    mip_flipped = np.flip(mip_inverted, axis=0)

    return mip_flipped


def normalize_img(img: np.ndarray) -> np.ndarray:
    """
    Normalizes an image to its maximum intensity.

    :param img: The input image.
    :type img: numpy.ndarray
    :return: The normalized image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    # Normalize the image to its maximum intensity
    img = img / np.max(img)

    return img


def equalize_hist(img: np.ndarray) -> np.ndarray:
    """
    Applies histogram equalization to an image.

    :param img: The input image.
    :type img: numpy.ndarray
    :return: The equalized image as numpy.ndarray.
    :rtype: numpy.ndarray
    """
    img_eq = exposure.equalize_adapthist(img)

    return img_eq


def create_rotational_mip_gif(pet_img: sitk.Image, mask_img: sitk.Image, gif_path: str, output_manager: system.OutputManager, rotation_step: int = 5,
                              output_spacing: Tuple[int, int, int] = (2, 2, 2)) -> None:
    """
    Creates a Maximum Intensity Projection (MIP) GIF of a PET image and its corresponding mask, rotating the image by a specified angle at each step.

    :param pet_path: The path to the PET image file.
    :type pet_path: str
    :param mask_path: The path to the mask image file.
    :type mask_path: str
    :param gif_path: The path to save the output GIF file.
    :type gif_path: str
    :param rotation_step: The angle to rotate the image by at each step.
    :type rotation_step: int
    :param output_spacing: The output voxel spacing of the resampled image.
    :type output_spacing: Tuple[int, int, int]
    :return: None
    :rtype: None
    """
    mask_array = sitk.GetArrayFromImage(mask_img)

    if np.all(mask_array == 0):  # Check if the mask is empty
        output_manager.log_update(f"Warning: The mask at is empty. Processing PET image without mask overlay.")
        mask_overlay = False
    else:
        mask_overlay = True

    # Resample the images
    resampler = ImageResampler()
    pet_img_resampled = resampler.resample_image_SimpleITK_DASK_array(pet_img, "linear", output_spacing)
    mask_img_resampled = resampler.resample_image_SimpleITK_DASK_array(mask_img, "nearest", output_spacing) if mask_overlay else None


    # Normalize the PET image
    pet_img_resampled = normalize_img(pet_img_resampled)

    # Apply histogram equalization to PET image
    pet_img_resampled = equalize_hist(pet_img_resampled)

    # Create color versions of the images
    pet_img_color = np.stack((pet_img_resampled, pet_img_resampled, pet_img_resampled), axis=-1)  # RGB
    mask_img_color = np.stack((0.5 * mask_img_resampled, np.zeros_like(mask_img_resampled), 0.5 * mask_img_resampled),
                              axis=-1) if mask_overlay else None  # RGB, purple color

    # Create a Dask client with default settings
    client = Client()

    # Scatter the data to the workers
    pet_img_color_future = client.scatter(pet_img_color, broadcast=True)
    mask_img_color_future = client.scatter(mask_img_color, broadcast=True) if mask_overlay else None

    # Create MIPs for a range of angles and store them
    angles = list(range(0, 360, rotation_step))
    pet_mip_images_futures = client.map(mip_3d, [pet_img_color_future] * len(angles), angles)
    mask_mip_images_futures = client.map(mip_3d, [mask_img_color_future] * len(angles), angles) if mask_overlay else []

    # Gather the images
    pet_mip_images = client.gather(pet_mip_images_futures)
    mask_mip_images = client.gather(mask_mip_images_futures) if mask_overlay else []

    if mask_overlay:
        # Blend the PET and mask MIPs
        overlay_mip_images = [cv2.addWeighted(pet_mip, 0.7, mask_mip.astype(pet_mip.dtype), 0.3, 0)
                              for pet_mip, mask_mip in zip(pet_mip_images, mask_mip_images)]
    else:
        overlay_mip_images = pet_mip_images

    # Normalize the image array to 0-255
    mip_images = [(255 * (im - np.min(im)) / (np.max(im) - np.min(im))).astype(np.uint8) for im in overlay_mip_images]

    # Save as gif
    imageio.mimsave(gif_path, mip_images, duration=FRAME_DURATION)

    # Cleanup
    client.close()
    del pet_img_resampled, mask_img_resampled, pet_img_color, mask_img_color, pet_mip_images, mask_mip_images, overlay_mip_images, mip_images


def compute_tumor_metrics(pet_path: str, mask_path: str, output_manager: system.OutputManager):
    # Load images
    mask_img = sitk.ReadImage(mask_path)
    pet_img = sitk.ReadImage(pet_path)
    # Convert images to numpy arrays
    mask_array = sitk.GetArrayFromImage(mask_img)
    pet_array = sitk.GetArrayFromImage(pet_img)

    # Check if the mask is empty
    if np.all(mask_array == 0):
        output_manager.console_update(f"Warning: The mask at {mask_path} contains no tumor regions.")
        return 0, 0  # Return 0 for both tumor volume and average intensity

    # Compute voxel volume
    spacing = mask_img.GetSpacing()
    voxel_volume = np.prod(spacing)

    # Calculate tumor volume
    tumor_voxel_count = np.sum(mask_array)  # assuming tumor is labeled with 1
    tumor_volume = (tumor_voxel_count * voxel_volume) / 1000  # convert to cm^3

    # Calculate average PET intensity within the tumor
    tumor_intensity_values = pet_array[mask_array == 1]
    average_intensity = np.mean(tumor_intensity_values)

    return tumor_volume, average_intensity


def save_metrics_to_csv(tumor_volume, avg_intensity, output_file):
    # Check if file exists to decide whether to write headers
    write_header = not os.path.exists(output_file)

    with open(output_file, 'a', newline='') as csvfile:
        fieldnames = ['Tumor Volume (cm^3)', 'Average PET Intensity (Bq/ml)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerow({'Tumor Volume (cm^3)': tumor_volume, 'Average PET Intensity (Bq/ml)': avg_intensity})


def threshold_segmentation_sitk(pet_image: sitk.Image, segmentation: sitk.Image, intensity_threshold: int) -> sitk.Image:
    """
    Thresholds the segmentation to only contain voxels with intensity higher than a specified value in the PET image.

    Parameters:
    pet_image_path (str): Array of the PET image.
    segmentation_path (str): Predicted segmentation.
    intensity_threshold (int): The intensity threshold. Voxels with PET intensity higher than this value will be kept.
    """
    suv_mask = sitk.BinaryThreshold(pet_image, lowerThreshold=intensity_threshold, upperThreshold=float('inf'), insideValue=1, outsideValue=0)
    combined_mask = sitk.And(suv_mask, segmentation)
    thresholded_segmentation = sitk.Mask(segmentation, combined_mask)
    return thresholded_segmentation