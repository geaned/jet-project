import logging
from typing import List, Optional
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import segmentation_models_pytorch as smp

from output_info import set_negative_entry
from utils import download_if_file_not_present

from scipy.optimize import minimize
from sklearn.cluster import DBSCAN


SCALING_DENOMINATOR = 10000

def optimal_angle(mask, clusterization=False):
    def mask_variance(angle):
        proj = np.sin(angle) * x_proj + np.cos(angle) * y_proj
        return (proj ** 2).mean() - (proj.mean()) ** 2
    
    change_size = int(np.sqrt(mask.sum() / SCALING_DENOMINATOR))
    if change_size > 1:
        mask = mask[::change_size,::change_size]
    
    indexes = np.nonzero(mask)
    
    if clusterization:
        indexes = np.array(indexes).T
        clustered = DBSCAN(eps=3, min_samples=2).fit(indexes)
        classes = np.array(clustered.labels_)
        most_frequent_class = np.argmax(np.bincount(classes))
        indexes = indexes[np.where(classes == most_frequent_class)]
        indexes = indexes.T
        indexes = (list(indexes[0]), list(indexes[1]))
    
    x_proj = np.repeat([np.arange(mask.shape[1])], mask.shape[0], axis=0)[indexes]
    y_proj = np.repeat([np.arange(mask.shape[0])], mask.shape[1], axis=0).T[indexes]
    angle_in_radians = minimize(mask_variance, x0=[1], bounds=[[-10, 10]]).x[0]

    # angle in range [-90, 90) degrees and another angle for flipped image
    reduced_angle_in_degrees = (angle_in_radians / np.pi * 180 + 90) % 180 - 90
    return reduced_angle_in_degrees, reduced_angle_in_degrees + 180

def detect_text(images_by_file_path, model_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # download rotation model if not present
    download_if_file_not_present('1ziaSS_upk7VHgS6jE1QnN_USSIUzCZNK', model_path)
    model = torch.load(model_path, map_location=torch.device(DEVICE))

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # get filenames and output progress
    masks_by_file_path = {}
    for file_path in images_by_file_path:
        print(f'Evaluating text mask on {file_path}...')
        image_shape = images_by_file_path[file_path].shape
        image = cv2.resize(preprocessing_fn(images_by_file_path[file_path]), (640, 640))
        
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
        pr_mask = (model.predict(x_tensor) > 0.5).squeeze().cpu()

        masks_by_file_path[file_path] = cv2.resize(
            np.array(
                pr_mask,
                dtype=float,
            ),
            (image_shape[0], image_shape[1]),
        )

    return masks_by_file_path

def rotate_to_horizontal(file_paths: List[str], result_folder, model_path, logging_dataframe: Optional[pd.DataFrame] = None):
    print('Parsing crops...')
    images_by_file_path = {}
    for file_path in file_paths:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_by_file_path[file_path] = image

    print('Looking for text masks on crops...')
    masks_by_file_path = detect_text(images_by_file_path, model_path)

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f'Rotating and saving crop to {file_path}... ', end='')

        if masks_by_file_path[file_path].sum() == 0:
            reason_for_removal = 'Text area not found'
            print(f'Failed! {reason_for_removal}')

            if logging_dataframe is not None:
                set_negative_entry(logging_dataframe, file_name, reason_for_removal)
    
            continue
        print()

        first_angle, second_angle = optimal_angle(masks_by_file_path[file_path], clusterization=True)
        flipped_file_name = file_name.replace('.png', '_flipped.png')
        cv2.imwrite(
            os.path.join(result_folder, file_name),
            imutils.rotate_bound(cv2.cvtColor(images_by_file_path[file_path], cv2.COLOR_RGB2BGR), angle=first_angle),
        )
        cv2.imwrite(
            os.path.join(result_folder, flipped_file_name),
            imutils.rotate_bound(cv2.cvtColor(images_by_file_path[file_path], cv2.COLOR_RGB2BGR), angle=second_angle),
        )
