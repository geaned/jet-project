from typing import List, Set, Optional, Tuple
import pandas as pd
import os
import cv2
import torch
import imutils
from pipeline_utils import create_folder_if_necessary

from detection_box_array import BasicCropData, DetectionBoxData, DetectionBoxDataArray
from output_info import set_negative_entry
from quality_metrics import global_check_before_detection
from quality_metrics import global_check_after_detection
from quality_metrics import local_check_after_detection


IOU_THRESHOLD = 0.3

def get_preemptively_globally_good_images(images_file_paths: List[str], image_tensors: torch.Tensor, logging_dataframe: Optional[pd.DataFrame] = None) -> Set[str]:
    good_image_paths = []
    good_image_tensors = []

    for file_path, image_tensor in zip(images_file_paths, image_tensors):
        file_name = os.path.basename(file_path)
        print(f'Checking image quality for {file_name}... ', end='')
        is_image_good, reason_for_bad = global_check_before_detection(image_tensor)

        if is_image_good:
            print('Passed!')
            good_image_paths.append(file_path)
            good_image_tensors.append(image_tensor)
        else:
            print(f'Failed! Reason: {reason_for_bad}')
            if logging_dataframe is not None:
                set_negative_entry(logging_dataframe, file_name, reason_for_bad)
    
    return good_image_paths, good_image_tensors

def get_latest_detection_label_paths(detect_folder: str) -> List[str]:
    latest_exp = max(
        os.listdir(detect_folder),
        key=lambda file_name: int(file_name.replace('exp', '') if file_name != 'exp' else '1'),
    )

    latest_detection_folder = os.path.join(detect_folder, latest_exp, "labels")
    print(f'Checking detection results folder {latest_detection_folder}...')
    return [os.path.join(latest_detection_folder, file_name) for file_name in os.listdir(latest_detection_folder)]

def organize_and_filter_detection_results(images_names: List[str], detection_tensors: List[torch.Tensor], image_tensors: List[torch.Tensor], apply_local_filter: bool = False, logging_dataframe: Optional[pd.DataFrame] = None) -> List[DetectionBoxDataArray]:
    detection_box_data_arrays = []

    print('Organizing detection data...')
    for image_name, detection_tensor, image_tensor in zip(images_names, detection_tensors, image_tensors):
        current_detection_box_data_array = []

        sorted_detection_list = sorted(list(detection_tensor.cpu().numpy()), key=lambda elem: elem[1])
        for idx, detection_box_line in enumerate(sorted_detection_list):
            left, top, right, bottom, confidence, num_class = detection_box_line
            image_tensor_cropped = image_tensor[int(top):int(bottom), int(left):int(right)]
            current_detection_box_data_array.append(DetectionBoxData(idx, int(num_class), image_tensor_cropped, left, top, right, bottom, confidence))

        current_data_array = DetectionBoxDataArray(image_name, image_tensor.shape[0], image_tensor.shape[1], current_detection_box_data_array)

        if not apply_local_filter:
            detection_box_data_arrays.append(current_data_array)
            continue

        print(f'Checking detection results from {image_name}... ', end='')
        is_image_good_after_detection, reason_for_bad_after_detection = global_check_after_detection(current_data_array)

        if is_image_good_after_detection:
            print('Passed!')
            detection_box_data_arrays.append(current_data_array)
        else:
            print(f'Failed! Reason: {reason_for_bad_after_detection}')
            if logging_dataframe is not None:
                set_negative_entry(logging_dataframe, image_name, reason_for_bad_after_detection)

    return detection_box_data_arrays

def save_cropped_images(destination_folder: str, detection_box_data_arrays: List[DetectionBoxDataArray]):
    create_folder_if_necessary(destination_folder)

    print('Saving cropped rods...')
    for detection_box_data_array in detection_box_data_arrays:
        for idx, detection_box_data in enumerate(detection_box_data_array.box_array):
            cv2.imwrite(
                os.path.join(destination_folder, detection_box_data_array.get_file_name_for_data_box(idx)),
                cv2.cvtColor(detection_box_data.img_tensor, cv2.COLOR_RGB2BGR)
            )

def get_crop_image_names_from_array(detection_box_data_arrays: List[DetectionBoxDataArray]) -> List[str]:
    found_names = []

    for detection_box_data_array in detection_box_data_arrays:
        found_names += [detection_box_data_array.get_file_name_for_data_box(idx) for idx in range(len(detection_box_data_array.box_array))]

    return found_names

def leave_only_good_crops(detection_box_data_arrays: List[DetectionBoxDataArray], logging_dataframe: Optional[pd.DataFrame] = None):
    for detection_box_data_array in detection_box_data_arrays:
        good_indices = []

        image_width, image_height = detection_box_data_array.img_width, detection_box_data_array.img_height
        for idx, detection_box_data in enumerate(detection_box_data_array.box_array):
            image_name = detection_box_data_array.get_file_name_for_data_box(idx)
            print(f'Checking local rod {idx} quality for {image_name}... ', end='')

            is_crop_good, reason_for_bad_crop = local_check_after_detection(detection_box_data, image_width, image_height)
            if is_crop_good:
                print('Passed!')
                good_indices.append(idx)
            else:
                print(f'Failed! Reason: {reason_for_bad_crop}')
                if logging_dataframe is not None:
                    set_negative_entry(logging_dataframe, image_name, reason_for_bad_crop)

        detection_box_data_array.box_array = [detection_box_data_array.box_array[idx] for idx in good_indices]

    return detection_box_data_arrays

def make_flipped_crop_array(crops_data_array: List[BasicCropData]) -> List[BasicCropData]:
    flipped_crops_data = []

    for crops_data in crops_data_array:
        flipped_crops_data.append(
            BasicCropData(
                crops_data.index,
                crops_data.img_name.replace('.png', '_flipped.png'),
                imutils.rotate_bound(crops_data.img_tensor, angle=180)
            )
        )

    return flipped_crops_data

def get_intersection_area_of_two_boxes(a: DetectionBoxData, b: DetectionBoxData) -> float:
    top_left_a, bottom_right_a = a.get_top_left_and_bottom_right()
    top_left_b, bottom_right_b = b.get_top_left_and_bottom_right()

    top_left_new = (max(top_left_a[0], top_left_b[0]), max(top_left_a[1], top_left_b[1]))
    bottom_right_new = (min(bottom_right_a[0], bottom_right_b[0]), min(bottom_right_a[1], bottom_right_b[1]))

    if top_left_new[0] >= bottom_right_new[0] or top_left_new[1] >= bottom_right_new[1]:
        return 0

    return (bottom_right_new[1] - top_left_new[1]) * (bottom_right_new[0] - top_left_new[0])

def get_iou_of_two_boxes(a: DetectionBoxData, b: DetectionBoxData) -> float:
    int_area = get_intersection_area_of_two_boxes(a, b)
    union_area = a.get_area() + b.get_area() - int_area

    return int_area / union_area

def remove_overlapping_bounding_boxes_by_iou(detection_box_data_arrays: List[DetectionBoxDataArray]) -> List[DetectionBoxDataArray]:
    print('Resolving overlapping detection boxes...')

    filtered_detection_box_data_arrays = []
    for detection_box_data_array in detection_box_data_arrays:
        excluded_indices = set()

        for i in range(len(detection_box_data_array.box_array)):
            for j in range(i + 1, len(detection_box_data_array.box_array)):
                first_detection_box_data = detection_box_data_array.box_array[i]
                second_detection_box_data = detection_box_data_array.box_array[j]

                iou = get_iou_of_two_boxes(first_detection_box_data, second_detection_box_data)
                if iou > IOU_THRESHOLD:
                    excluded_indices.add(i if first_detection_box_data.confidence < second_detection_box_data.confidence else j)
    
        filtered_detection_box_data_array = DetectionBoxDataArray(
            detection_box_data_array.img_name,
            detection_box_data_array.img_width,
            detection_box_data_array.img_height,
            [detection_box_data_array.box_array[i] for i in range(len(detection_box_data_array.box_array)) if i not in excluded_indices]
        )
        filtered_detection_box_data_arrays.append(filtered_detection_box_data_array)

    return filtered_detection_box_data_arrays

def select_more_confident_data_arrays(detection_box_data_arrays: List[DetectionBoxDataArray], detection_box_data_arrays_flipped: List[DetectionBoxDataArray], crops_data_array: List[BasicCropData], crops_data_array_flipped: List[BasicCropData], logging_dataframe: Optional[pd.DataFrame] = None) -> List[DetectionBoxDataArray]:
    more_confident_box_data_arrays = []
    more_confident_crops_data_array = []

    for detection_box_data_array, detection_box_data_array_flipped, crops_data, crops_data_flipped in zip(detection_box_data_arrays, detection_box_data_arrays_flipped, crops_data_array, crops_data_array_flipped):
        print(f'Choosing between {detection_box_data_array.img_name} and {detection_box_data_array_flipped.img_name}... ', end='')
        first_confidence_sum = sum(box_array.confidence for box_array in detection_box_data_array.box_array)
        second_confidence_sum = sum(box_array.confidence for box_array in detection_box_data_array_flipped.box_array)
        first_box_array_items_amount = len(detection_box_data_array.box_array)
        second_box_array_items_amount = len(detection_box_data_array_flipped.box_array)

        if first_box_array_items_amount == 0 and second_box_array_items_amount == 0:
            reason_for_bad = 'Text strings not found'

            print(reason_for_bad)
            if logging_dataframe is not None:
                set_negative_entry(logging_dataframe, detection_box_data_array.img_name, reason_for_bad)
            continue

        print(f'Cumulative confidence {first_confidence_sum:.2f} ({first_box_array_items_amount} text strings) vs. {second_confidence_sum:.2f} ({second_box_array_items_amount} text strings)')
        if first_confidence_sum > second_confidence_sum:
            more_confident_box_data_arrays.append(detection_box_data_array)
            more_confident_crops_data_array.append(crops_data)
        else:
            detection_box_data_array_flipped.img_name = detection_box_data_array.img_name
            crops_data_flipped.img_name = crops_data.img_name
            more_confident_box_data_arrays.append(detection_box_data_array_flipped)
            more_confident_crops_data_array.append(crops_data_flipped)
    
    return more_confident_box_data_arrays, more_confident_crops_data_array

def save_crops(destination_folder: str, crops_data_array: List[BasicCropData]):
    create_folder_if_necessary(destination_folder)

    print('Saving final crops...')
    for crops_data in crops_data_array:
        cv2.imwrite(
            os.path.join(destination_folder, crops_data.get_file_name_for_crop_box()),
            cv2.cvtColor(crops_data.img_tensor, cv2.COLOR_RGB2BGR)
        )

def group_and_write_strings_to_text_files(destination_folder: str, detection_box_data_arrays: List[DetectionBoxDataArray]):
    create_folder_if_necessary(destination_folder)

    print('Writing found strings to text files...')
    for detection_box_data_array in detection_box_data_arrays:
        label_name = os.path.join(destination_folder, detection_box_data_array.img_name.replace('.png', '.txt'))
        with open(label_name, 'w') as text_results:
            for found_string in detection_box_data_array.merge_digits_into_strings():
                # left, top, right, bottom = box
                text_results.write(f'{found_string}\n')
