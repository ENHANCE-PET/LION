#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LIONZ image prediction
---------------

This module contains functions that are responsible for predicting tumors from images.

LIONZ stands for Lesion segmentatION, a sophisticated solution for lesion segmentation tasks in medical imaging datasets.

.. moduleauthor:: Lalith Kumar Shiyam Sundar <lalith.shiyamsundar@meduniwien.ac.at>
.. versionadded:: 0.1.0
"""
import logging
import os
import subprocess

import SimpleITK as sitk
import nibabel as nib
import numpy as np

from lionz.file_utilities import get_files
from lionz.image_processing import ImageResampler, threshold_segmentation
from lionz.resources import MODELS, map_model_name_to_task_number, RULES, TRACER_WORKFLOWS
from lionz.constants import INTERPOLATION, NNUNET_RESULTS_FOLDER


class ImagePreprocessor:
    def __init__(self, workflow_dir: str, model_config: dict):
        self.workflow_dir = workflow_dir
        self.model_config = model_config

    def _get_voxel_volume(self, image_path: str) -> float:
        image_obj = nib.load(image_path)
        return np.prod(image_obj.header["pixdim"][1:4])

    def _resample_image(self, image_path: str, target_voxel_spacing: list, interpolation: str = 'linear',
                        output_size: tuple = None) -> None:
        sitk_image = sitk.ReadImage(image_path)
        resampled_image = ImageResampler.resample_image_SimpleITK_DASK(sitk_image, interpolation,
                                                                       tuple(target_voxel_spacing), output_size)
        sitk.WriteImage(resampled_image, image_path)

    def _resample_image_to_reference(self, moving_image_path: str, reference_image_path: str,
                                     target_voxel_spacing: list):
        reference_image = sitk.ReadImage(reference_image_path)
        moving_image = sitk.ReadImage(moving_image_path)
        resampled_image = ImageResampler.reslice_identity(reference_image, moving_image, moving_image_path)
        sitk.WriteImage(resampled_image, moving_image_path)

    def _infer_workflow_directories(self, model_name: str):
        all_workflows = os.listdir(self.workflow_dir)
        return [wf for wf in all_workflows if wf.startswith(model_name)]

    def preprocess_workflow(self, model_name: str):
        workflows = self._infer_workflow_directories(model_name)

        for idx, workflow_name in enumerate(workflows):
            workflow_path = os.path.join(self.workflow_dir, workflow_name)
            images = get_files(workflow_path, wildcard=".nii.gz")
            num_images = len(images)

            target_voxel_spacing = self.model_config[model_name][idx].get("voxel_spacing", ["1", "1", "1"])

            if num_images == 1:
                self._resample_image(images[0], target_voxel_spacing, interpolation=INTERPOLATION)
            elif num_images > 1:
                max_voxel_image = max(images, key=self._get_voxel_volume)
                self._resample_image(max_voxel_image, target_voxel_spacing, interpolation=INTERPOLATION)
                for image in images:
                    if image != max_voxel_image:
                        self._resample_image_to_reference(image, max_voxel_image, target_voxel_spacing)


def predict_tumor(workflow_dir: str, model_name: str, output_dir: str, accelerator: str, thresholding: bool = False):
    # Preprocess the images
    workflowPreprocessor = ImagePreprocessor(workflow_dir, MODELS)
    workflowPreprocessor.preprocess_workflow(model_name)

    # Get the workflow-task mapping
    workflow_to_task_map = map_model_name_to_task_number(model_name)

    mask_path = None

    for workflow_name, task_number in workflow_to_task_map.items():
        # Predict using the current workflow
        current_workflow_dir = os.path.join(workflow_dir, f"{model_name}_{workflow_name}")
        trainer = [model["trainer"] for model in MODELS[model_name] if task_number in model["directory"]][0]
        plans = [model["plans"] for model in MODELS[model_name] if task_number in model["directory"]][0]
        configuration = [model["configuration"] for model in MODELS[model_name] if task_number in model["directory"]][0]
        command = f'nnUNetv2_predict -i {current_workflow_dir} -o {output_dir} -d {task_number} -c {configuration}' \
                  f' -f all -tr {trainer} --disable_tta -device {accelerator} -p {plans}'
        os.environ["nnUNet_results"] = NNUNET_RESULTS_FOLDER
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        mask_path = get_files(output_dir, '.nii.gz')[0]

        # Retain only the tumor label
        tumor_label = TRACER_WORKFLOWS[model_name]['workflows'][workflow_name]["tumor_label"]
        prediction = nib.load(mask_path)
        prediction_data = prediction.get_fdata()
        prediction_data[prediction_data != tumor_label] = 0
        prediction_data[prediction_data == tumor_label] = 1
        if thresholding:
            pet_image = get_files(current_workflow_dir, '.nii.gz')[0]
            threshold_value = TRACER_WORKFLOWS[model_name]['workflows'][workflow_name]["threshold"]
            prediction_data = threshold_segmentation(pet_image, prediction_data, threshold_value)
        new_prediction = nib.Nifti1Image(prediction_data, prediction.affine, prediction.header)
        nib.save(new_prediction, mask_path)

        # Decide what to do based on rules
        action = get_next_action(model_name, workflow_name, mask_path)
        logging.info(f" Action for {workflow_name}: {action}")
        if action == "stop":
            break
        elif action == "delete_mask_and_continue" and mask_path:
            logging.info(f" Deleting mask {mask_path}")
            os.remove(mask_path)
            mask_path = None
        elif action == "continue":
            continue

    # get the nifti mask in output_dir
    resampled_segmentation_file = get_files(output_dir, '.nii.gz')[0]
    return resampled_segmentation_file


def get_next_action(model_name, workflow_name, mask_path):
    rule_details = RULES.get(model_name, {}).get(workflow_name, {})
    rule_func_data = rule_details.get('rule_func')

    if not rule_func_data:
        return "continue"

    # Unpack the function and its arguments
    rule_func, kwargs = rule_func_data

    # Check if the rule_func is callable before attempting to call it
    rule_result = None
    if callable(rule_func):
        rule_result = rule_func(mask_path, **kwargs)
        logging.info(f"Rule result for {workflow_name}: {rule_result}")
    else:
        logging.info(f"WARNING: function {rule_func} is not callable!")

    if rule_result:
        return rule_details.get('action_on_true', 'continue')
    else:
        return rule_details.get('action_on_false', 'continue')


def post_process(reference_file: str, moving_segmentation: str, output_segmentation: str):
    reference_image = sitk.ReadImage(reference_file)
    moving_image = sitk.ReadImage(moving_segmentation)
    resampled_image = ImageResampler.reslice_identity(reference_image, moving_image, moving_segmentation,
                                                      is_label_image=True)
    sitk.WriteImage(resampled_image, output_segmentation)
