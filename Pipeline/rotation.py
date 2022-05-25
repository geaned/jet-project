from typing import List, Optional
import cv2
import imutils
import numpy as np
import pandas as pd
import os
import torch
import segmentation_models_pytorch as smp
from detection_box_array import BasicCropData, DetectionBoxDataArray

from output_info import set_negative_entry
from pipeline_utils import create_folder_if_necessary
from pipeline_utils import download_if_file_not_present

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
    return reduced_angle_in_degrees

def detect_text(detection_box_data_arrays: List[DetectionBoxDataArray], model_path: str):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # download rotation model if not present
    download_if_file_not_present('1ziaSS_upk7VHgS6jE1QnN_USSIUzCZNK', model_path)
    model = torch.load(model_path, map_location=torch.device(DEVICE))

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # get text mask for every image tensor
    masks_found = []
    for detection_box_data_array in detection_box_data_arrays:
        cur_masks_array = []

        for idx, detection_box_data in enumerate(detection_box_data_array.box_array):
            print(f'Evaluating text mask on {detection_box_data_array.get_file_name_for_data_box(idx)}...')
            image_dims = detection_box_data.get_absolute_dimensions()
            image = cv2.resize(preprocessing_fn(detection_box_data.img_tensor), (640, 640))
            
            x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0).permute(0, 3, 1, 2).to(torch.float32)
            pr_mask = (model.predict(x_tensor) > 0.5).squeeze().cpu()

            parsed_mask = cv2.resize(
                np.array(
                    pr_mask,
                    dtype=float,
                ),
                (int(image_dims[0]), int(image_dims[1])),
            )
            cur_masks_array.append(parsed_mask)
        
        masks_found.append(cur_masks_array)

    return masks_found

def rotate_to_horizontal(detection_box_data_arrays: List[DetectionBoxDataArray], model_path, masks_folder: Optional[str] = None, logging_dataframe: Optional[pd.DataFrame] = None) -> List[BasicCropData]:
    if masks_folder is not None:
        create_folder_if_necessary(masks_folder)

    print(f'Looking for text masks on crops{" and saving masks" if masks_folder is not None else ""}...')
    masks_for_arrays = detect_text(detection_box_data_arrays, model_path)

    rotated_crops_with_data = []
    for detection_box_data_array, masks_for_array in zip(detection_box_data_arrays, masks_for_arrays):
        good_indices = []
        opt_angles = []

        for idx, (detection_box_data, mask) in enumerate(zip(detection_box_data_array.box_array, masks_for_array)):
            crop_name = detection_box_data_array.get_file_name_for_data_box(idx)

            print(f'Checking crop {crop_name}... ', end='')
            if mask.sum() == 0:
                reason_for_removal = 'Text area not found'
                print(f'Failed! {reason_for_removal}')

                if logging_dataframe is not None:
                    set_negative_entry(logging_dataframe, crop_name, reason_for_removal)
                
                continue
            print()

            good_indices.append(idx)

            if masks_folder is not None:
                cv2.imwrite(
                    os.path.join(masks_folder, crop_name),
                    mask * 255
                )

            opt_angle = optimal_angle(mask, clusterization=True)
            opt_angles.append(opt_angle)

        detection_box_data_array.box_array = [detection_box_data_array.box_array[idx] for idx in good_indices]

        for idx, detection_box_data in enumerate(detection_box_data_array.box_array):
            rotated_crop = imutils.rotate_bound(detection_box_data.img_tensor, angle=opt_angles[idx])
            rotated_crops_with_data.append(BasicCropData(detection_box_data.index, detection_box_data_array.img_name, rotated_crop))
    
    return rotated_crops_with_data
