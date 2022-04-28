import os
import time
import argparse

from parsing import get_preemptively_globally_good_images_names
from parsing import get_latest_detection_label_paths
from parsing import get_detection_box_data_for_filtered_images
from parsing import create_folder_if_necessary
from parsing import save_cropped_images
from parsing import get_locally_good_crops_paths
from parsing import remove_overlapping_bounding_boxes_by_iou
from parsing import group_and_write_strings_to_text_files
from rotation import rotate_to_horizontal


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--source', nargs=1, required=True, help='Source folder with images in .png format')
exec_args = arg_parser.parse_args()

IMAGES_FOLDER = exec_args.source[0]
# IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets_not_split/data_all/img')
YOLO_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'yolov5')
ROD_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rod_detection')
DIGIT_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Digit_detection')
ROTATION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rotation')

CROP_RESULT_FOLDER = 'crop_results'
ROTATION_RESULT_FOLDER = 'rotation_results'
STRING_RESULT_FOLDER = 'found_strings'

detect_folder = os.path.join(YOLO_FOLDER, "runs/detect")

start_time = time.time()

# run global quality evaluation
images_file_paths = [os.path.join(IMAGES_FOLDER, file_name) for file_name in os.listdir(IMAGES_FOLDER)]
preemptively_good_image_names = get_preemptively_globally_good_images_names(images_file_paths)

# run detection
os.system(f'python {os.path.join(YOLO_FOLDER, "detect.py")} --source {IMAGES_FOLDER} --img 1080 --weights {os.path.join(ROD_DETECTION_FOLDER, "rod_weights.pt")} --conf-thres 0.7 --save-txt')

# find and gather label files
rod_detection_results_label_paths = get_latest_detection_label_paths(detect_folder)

# organize rod detection data into special classes
actually_good_detection_box_data_arrays = get_detection_box_data_for_filtered_images(rod_detection_results_label_paths, preemptively_good_image_names)

# make a folder for cropped images
create_folder_if_necessary(CROP_RESULT_FOLDER)

# save cropped images for gathered detection box data
save_cropped_images(IMAGES_FOLDER, CROP_RESULT_FOLDER, actually_good_detection_box_data_arrays)

# perform local quality check for each crop
good_crops_for_rotation = get_locally_good_crops_paths(CROP_RESULT_FOLDER, actually_good_detection_box_data_arrays)

# make a folder for rotated rods
create_folder_if_necessary(ROTATION_RESULT_FOLDER)

# rotate crops to make text horizontal
model_path = os.path.join(ROTATION_FOLDER, 'text_detection_model.pth')
rotate_to_horizontal(good_crops_for_rotation, ROTATION_RESULT_FOLDER, model_path)

# detect digits
os.system(f'python {os.path.join(YOLO_FOLDER, "detect.py")} --source {ROTATION_RESULT_FOLDER} --img 640 --weights {os.path.join(DIGIT_DETECTION_FOLDER, "digit_weights.pt")} --save-txt --save-conf')

# once again, gather labels and
# organize rod detection data into special classes
digit_detection_results_label_paths = get_latest_detection_label_paths(detect_folder)
rotated_crops_detection_box_data_arrays = get_detection_box_data_for_filtered_images(digit_detection_results_label_paths, with_confidence=True)

# resolve digit intersections
filtered_rotated_crops_detection_box_data_arrays = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays)

# make a folder for rotated rods
create_folder_if_necessary(STRING_RESULT_FOLDER)

# merge found digits into string and output
group_and_write_strings_to_text_files(filtered_rotated_crops_detection_box_data_arrays, STRING_RESULT_FOLDER)

finish_time = time.time()
print(f'The whole pipeline took {finish_time - start_time} seconds!')
