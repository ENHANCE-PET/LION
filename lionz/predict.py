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

import sys
import torch
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from lionz import models
from typing import Tuple, List, Dict, Iterator


def initialize_predictor(model: models.Model, accelerator: str) -> nnUNetPredictor:
    """
    Initializes the model for prediction.

    :param model: The model object.
    :type model: Model
    :param accelerator: The accelerator for prediction.
    :type accelerator: str
    :return: The initialized predictor object.
    :rtype: nnUNetPredictor
    """
    device = torch.device(accelerator)
    predictor = nnUNetPredictor(allow_tqdm=False, device=device)
    predictor.initialize_from_trained_model_folder(model.configuration_directory, use_folds=("all",))
    return predictor


def process_case(preprocessor, chunk: np.ndarray, chunk_properties: Dict, predictor: nnUNetPredictor) -> Dict:
    data, seg, prop = preprocessor.run_case_npy(chunk,
                                          None,
                                          chunk_properties,
                                          predictor.plans_manager,
                                          predictor.configuration_manager,
                                          predictor.dataset_json)

    data_tensor = torch.from_numpy(data).contiguous()
    if predictor.device == "cuda":
        data_tensor = data_tensor.pin_memory()

    return {'data': data_tensor, 'data_properties': prop, 'ofile': None}


def preprocessing_iterator_from_array(image_array: np.ndarray, image_properties: dict, predictor: nnUNetPredictor) -> iter:

    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=predictor.verbose)
    processed_case = tuple([process_case(preprocessor, image_array, image_properties, predictor)])
    iterator = iter(processed_case)

    return iterator


def predict_from_array_by_iterator(image_array: np.ndarray, model: models.Model, accelerator: str, nnunet_log_filename: str = None) -> np.ndarray:
    image_array = image_array[None, ...]

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    nnunet_log_file = None
    if nnunet_log_filename is not None:
        nnunet_log_file = open(nnunet_log_filename, "a")
        sys.stdout = nnunet_log_file
        sys.stderr = nnunet_log_file

    try:
        predictor = initialize_predictor(model, accelerator)
        image_properties = {
            'spacing': model.voxel_spacing
        }

        iterator = preprocessing_iterator_from_array(image_array, image_properties, predictor)
        segmentations = predictor.predict_from_data_iterator(iterator)
        segmentations = [segmentation[None, ...] for segmentation in segmentations]
        combined_segmentations = np.squeeze(segmentations)
        combined_segmentations[combined_segmentations != model.tumor_label] = 0
        combined_segmentations[combined_segmentations == model.tumor_label] = 1

        return combined_segmentations

    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        if nnunet_log_filename is not None and nnunet_log_file is not None:
            nnunet_log_file.close()

