from typing import List, Set
from PIL import Image
import os

from detection_box_array import DetectionBoxData, DetectionBoxDataArray
from quality_metrics import global_check_before_detection
from quality_metrics import global_check_after_detection
from quality_metrics import local_check_after_detection

def create_folder_if_necessary(folder_name: str):
    try:
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created, saving cropped rods...')
    except FileExistsError:
        print(f'Folder "{folder_name}" exists, saving cropped rods...')

def get_preemptively_globally_good_images_names(images_file_paths: List[str]) -> Set[str]:
    good_image_names = set()

    for file_path in images_file_paths:
        print(f'Checking image quality for {file_path}... ', end='')
        is_image_good, reason_for_bad = global_check_before_detection(file_path)
        print(f'{"Passed!" if is_image_good else "Failed! Reason: "} {reason_for_bad}')

        if is_image_good:
            good_image_names.add(file_path.split('/')[-1])
    
    return good_image_names

def get_latest_detection_label_paths(detect_folder: str) -> List[str]:
    latest_exp = max(
        os.listdir(detect_folder),
        key=lambda file_name: int(file_name.replace('exp', '') if file_name != 'exp' else '1'),
    )

    latest_detection_folder = os.path.join(detect_folder, latest_exp, "labels")
    print(f'Checking rod detection results folder {latest_detection_folder}...')
    return [os.path.join(latest_detection_folder, file_name) for file_name in os.listdir(latest_detection_folder)]

def get_detection_box_data_for_actually_globally_good_images(rod_detection_results_label_paths: List[str], preemptively_good_image_names: Set[str]) -> List[DetectionBoxDataArray]:
    good_detection_box_data_arrays = []

    for file_path in rod_detection_results_label_paths:
        print(f'Checking detection results from {file_path}... ', end='')
        current_image_name = file_path.split('/')[-1].replace('.txt', '.png')

        # get detection data for images that passed tests before detection
        if current_image_name in preemptively_good_image_names:
            with open(file_path) as label_file:
                current_detection_box_data_array = []

                while True:
                    line_values = label_file.readline().split()
                    if not line_values:
                        break

                    class_num, center_x, center_y, width, height = int(line_values[0]), float(line_values[1]), float(line_values[2]), float(line_values[3]), float(line_values[4])
                    current_detection_box_data_array.append(DetectionBoxData(class_num, center_x, center_y, width, height))

        current_data_array = DetectionBoxDataArray(current_image_name, current_detection_box_data_array)
        is_image_good_after_detection, reason_for_bad_after_detection = global_check_after_detection(current_data_array)

        # finally run global quality evaultion on after-detection features
        if is_image_good_after_detection:
            print('Passed!')
            good_detection_box_data_arrays.append(current_data_array)
        else:
            print(f'Failed! Reason: {reason_for_bad_after_detection}')
    
    return good_detection_box_data_arrays

def save_cropped_images(images_folder: str, good_detection_box_data_arrays: List[DetectionBoxDataArray]):
    for detection_box_data_array in good_detection_box_data_arrays:
        img_path = os.path.join(images_folder, detection_box_data_array.img_name)
        print(f'Saving cropped rods on image {img_path}...')
        current_image = Image.open(img_path)
        image_width, image_height = current_image.size
        for idx, detection_box_data in enumerate(detection_box_data_array.box_array):
            center_x, center_y = detection_box_data.get_absolute_center(image_width, image_height)
            box_width, box_height = detection_box_data.get_absolute_dimensions(image_width, image_height)

            left, right = int(center_x - box_width / 2), int(center_x + box_width / 2)
            top, bottom = int(center_y - box_height / 2), int(center_y + box_height / 2)

            current_crop = current_image.crop((left, top, right, bottom))
            current_crop.save(os.path.join('crop_results', detection_box_data_array.img_name.replace('.png', f'_{idx}.png')))

def get_locally_good_crops_paths(good_detection_box_data_arrays: List[DetectionBoxDataArray]) -> List[str]:
    good_crops_for_rotation = []
    for good_detection_box_data_array in good_detection_box_data_arrays:
        for idx, good_detection_box_data in enumerate(good_detection_box_data_array.box_array):
            image_path = os.path.join('crop_results', good_detection_box_data_array.img_name)
            possible_crop_path = image_path.replace('.png', f'_{idx}.png')
            print(f'Checking local rod {idx} quality for {image_path}... ', end='')


            is_crop_good, reason_for_bad_crop = local_check_after_detection(possible_crop_path, good_detection_box_data)
            if is_crop_good:
                print('Passed!')
                good_crops_for_rotation.append(possible_crop_path)
            else:
                print(f'Failed! Reason: {reason_for_bad_crop}')
    
    return good_crops_for_rotation
