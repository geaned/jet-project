import os
import gc
import cv2
import time
import torch
import argparse

from parsing import get_preemptively_globally_good_images
from parsing import organize_and_filter_detection_results
from parsing import save_cropped_images
from parsing import get_crop_image_names_from_array
from parsing import leave_only_good_crops
from parsing import make_flipped_crop_array
from parsing import remove_overlapping_bounding_boxes_by_iou
from parsing import select_more_confident_data_arrays
from parsing import save_crops
from parsing import group_and_write_strings_to_text_files
from pipeline_utils import download_if_file_not_present
from output_info import make_base_dataframe_for_paths
from output_info import write_dataframe_sorted_by_name
from output_info import get_full_stats_from_dataframe
from rotation import rotate_to_horizontal


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--source', nargs=1, required=True, help='Source folder with images in .png format')
arg_parser.add_argument('--crops', default=False, action='store_true')
arg_parser.add_argument('--no-rotations', default=False, action='store_true')
exec_args = arg_parser.parse_args()

IMAGES_FOLDER = exec_args.source[0]
YOLO_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'yolov5')
ROD_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rod_detection')
DIGIT_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Digit_detection')
ROTATION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rotation')

CROP_RESULT_FOLDER = 'crops'
ROTATION_RESULT_FOLDER = 'results'
STRING_RESULT_FOLDER = 'strings'

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

detect_folder = os.path.join(YOLO_FOLDER, "runs/detect")

start_time = time.time()

# find images
images_file_paths = [os.path.join(IMAGES_FOLDER, file_name) for file_name in os.listdir(IMAGES_FOLDER)]
images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in images_file_paths]

# make dataframe for image quality
image_quality_dataframe = make_base_dataframe_for_paths(images_file_paths)

# run global quality evaluation
preemptively_good_image_paths, preemptively_good_images = get_preemptively_globally_good_images(images_file_paths, images, logging_dataframe=image_quality_dataframe)

del images
gc.collect()

# download rod detection model if not present
rod_detection_model_path = os.path.join(ROD_DETECTION_FOLDER, "rod_weights.pt")
download_if_file_not_present('1UZQ1YW9Fap2ESzfMgYhUhC4SdTUZK3mw', rod_detection_model_path)

# run detection
rod_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rod_detection_model_path, device=DEVICE)
rod_detection_model.conf = 0.7

rod_detection_results = rod_detection_model(preemptively_good_images, size=1080)
rod_detection_results.print()

rod_detection_results_tensors = rod_detection_results.xyxy

# organize rod detection data into special classes
preemptively_good_image_names = [os.path.basename(image_path) for image_path in preemptively_good_image_paths]
actually_good_detection_box_data_arrays = organize_and_filter_detection_results(preemptively_good_image_names, rod_detection_results_tensors, preemptively_good_images, apply_local_filter=True, logging_dataframe=image_quality_dataframe)

del preemptively_good_images
gc.collect()

# save cropped images for gathered detection box data and quality dataframe
if exec_args.crops:
    save_cropped_images(CROP_RESULT_FOLDER, actually_good_detection_box_data_arrays)
write_dataframe_sorted_by_name(image_quality_dataframe, 'image_quality.csv')

# make dataframe for crop quality
crop_file_paths = get_crop_image_names_from_array(actually_good_detection_box_data_arrays)
crop_quality_dataframe = make_base_dataframe_for_paths(crop_file_paths)

# perform local quality check for each crop
good_for_rotation_crops_arrays = leave_only_good_crops(actually_good_detection_box_data_arrays, logging_dataframe=crop_quality_dataframe)

del actually_good_detection_box_data_arrays
gc.collect()

# rotate crops to make text horizontal
model_path = os.path.join(ROTATION_FOLDER, 'text_segmentation_model.pth')
rotated_crops_array = rotate_to_horizontal(good_for_rotation_crops_arrays, model_path, logging_dataframe=crop_quality_dataframe)

del good_for_rotation_crops_arrays
gc.collect()

# make crops array for flipped crops
rotated_crops_array_flipped = make_flipped_crop_array(rotated_crops_array)

# download digit detection model if not present
digit_detection_model_path = os.path.join(DIGIT_DETECTION_FOLDER, "digit_weights.pt")
download_if_file_not_present('1KTLSkFf8-VuwtPRArxLQINzUhKhY2Qle', digit_detection_model_path)

# prepare data for detection (for flipped as well)
crop_images_names = [rotated_crop.get_file_name_for_crop_box() for rotated_crop in rotated_crops_array]
rotated_crops = [rotated_crop.img_tensor for rotated_crop in rotated_crops_array]
crop_images_names_flipped = [rotated_crop.get_file_name_for_crop_box() for rotated_crop in rotated_crops_array_flipped]
rotated_crops_flipped = [rotated_crop.img_tensor for rotated_crop in rotated_crops_array_flipped]

# detect digits (for flipped as well)
digit_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=digit_detection_model_path, device=DEVICE)

digit_detection_results = digit_detection_model(rotated_crops, size=640)
digit_detection_results.print()
digit_detection_results_flipped = digit_detection_model(rotated_crops_flipped, size=640)
digit_detection_results_flipped.print()

digit_detection_results_tensors = digit_detection_results.xyxy
digit_detection_results_flipped_tensors = digit_detection_results_flipped.xyxy

# organize digit detection data into special classes (for flipped as well)
rotated_crops_detection_box_data_arrays = organize_and_filter_detection_results(crop_images_names, digit_detection_results_tensors, rotated_crops)
rotated_crops_detection_box_data_arrays_flipped = organize_and_filter_detection_results(crop_images_names_flipped, digit_detection_results_flipped_tensors, rotated_crops_flipped)

# resolve digit intersections (for flipped as well)
filtered_rotated_crops_detection_box_data_arrays = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays)
filtered_rotated_crops_detection_box_data_arrays_flipped = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays_flipped)

# filter out least possible rotation among each pair for a non-flipped image and a flipped image
more_confident_detection_box_data_arrays, more_confident_crop_data = select_more_confident_data_arrays(filtered_rotated_crops_detection_box_data_arrays, filtered_rotated_crops_detection_box_data_arrays_flipped, rotated_crops_array, rotated_crops_array_flipped, logging_dataframe=crop_quality_dataframe)

del filtered_rotated_crops_detection_box_data_arrays
del filtered_rotated_crops_detection_box_data_arrays_flipped
del rotated_crops_array
del rotated_crops_array_flipped
gc.collect()

# save more confident rotated crops
if not exec_args.no_rotations:
    save_crops(ROTATION_RESULT_FOLDER, more_confident_crop_data)

# merge found digits into string and output and save crop quality dataframe
group_and_write_strings_to_text_files(STRING_RESULT_FOLDER, more_confident_detection_box_data_arrays)
write_dataframe_sorted_by_name(crop_quality_dataframe, 'crop_quality.csv')

finish_time = time.time()

print('-------------- IMAGE QUALITY --------------')
get_full_stats_from_dataframe(image_quality_dataframe, type='image')
print('-------------- CROPS QUALITY --------------')
get_full_stats_from_dataframe(crop_quality_dataframe, type='crop')
print('-------------------------------------------')
print(f'The whole pipeline took {finish_time - start_time:.2f} seconds!')
