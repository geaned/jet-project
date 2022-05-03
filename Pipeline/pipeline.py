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
from parsing import filter_out_less_probable_data_arrays
from parsing import group_and_write_strings_to_text_files
from output_info import make_base_dataframe_for_paths
from output_info import write_dataframe_sorted_by_name
from output_info import get_full_stats_from_dataframe
from rotation import rotate_to_horizontal
from utils import download_if_file_not_present


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--source', nargs=1, required=True, help='Source folder with images in .png format')
exec_args = arg_parser.parse_args()

IMAGES_FOLDER = exec_args.source[0]
YOLO_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'yolov5')
ROD_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rod_detection')
DIGIT_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Digit_detection')
ROTATION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rotation')

CROP_RESULT_FOLDER = 'crops'
ROTATION_RESULT_FOLDER = 'results'
STRING_RESULT_FOLDER = 'strings'

detect_folder = os.path.join(YOLO_FOLDER, "runs/detect")

start_time = time.time()

# find images
images_file_paths = [os.path.join(IMAGES_FOLDER, file_name) for file_name in os.listdir(IMAGES_FOLDER)]

# make dataframe for image quality
image_quality_dataframe = make_base_dataframe_for_paths(images_file_paths)

# run global quality evaluation
preemptively_good_image_names = get_preemptively_globally_good_images_names(images_file_paths, logging_dataframe=image_quality_dataframe)

# download rod detection model if not present
rod_detection_model_path = os.path.join(ROD_DETECTION_FOLDER, "rod_weights.pt")
download_if_file_not_present('1UZQ1YW9Fap2ESzfMgYhUhC4SdTUZK3mw', rod_detection_model_path)

# run detection
os.system(f'python {os.path.join(YOLO_FOLDER, "detect.py")} --source {IMAGES_FOLDER} --img 1080 --weights {rod_detection_model_path} --conf-thres 0.7 --save-txt')

# find and gather rod label files from last detection
rod_detection_results_label_paths = get_latest_detection_label_paths(detect_folder)

# organize rod detection data into special classes
actually_good_detection_box_data_arrays = get_detection_box_data_for_filtered_images(rod_detection_results_label_paths, preemptively_good_image_names, logging_dataframe=image_quality_dataframe)

# make a folder for cropped images
create_folder_if_necessary(CROP_RESULT_FOLDER)

# save cropped images for gathered detection box data and quality dataframe
save_cropped_images(IMAGES_FOLDER, CROP_RESULT_FOLDER, actually_good_detection_box_data_arrays)
write_dataframe_sorted_by_name(image_quality_dataframe, 'image_quality.csv')

# make dataframe for crop quality
crop_quality_dataframe = make_base_dataframe_for_paths([os.path.join(CROP_RESULT_FOLDER, file_name) for file_name in os.listdir(CROP_RESULT_FOLDER)])

# perform local quality check for each crop
good_crops_for_rotation = get_locally_good_crops_paths(CROP_RESULT_FOLDER, actually_good_detection_box_data_arrays, logging_dataframe=crop_quality_dataframe)

# make a folder for rotated rods
create_folder_if_necessary(ROTATION_RESULT_FOLDER)

# rotate crops to make text horizontal
model_path = os.path.join(ROTATION_FOLDER, 'text_segmentation_model.pth')
rotate_to_horizontal(good_crops_for_rotation, ROTATION_RESULT_FOLDER, model_path, logging_dataframe=crop_quality_dataframe)

# download digit detection model if not present
digit_detection_model_path = os.path.join(DIGIT_DETECTION_FOLDER, "digit_weights.pt")
download_if_file_not_present('1KTLSkFf8-VuwtPRArxLQINzUhKhY2Qle', digit_detection_model_path)

# detect digits
os.system(f'python {os.path.join(YOLO_FOLDER, "detect.py")} --source {ROTATION_RESULT_FOLDER} --img 640 --weights {digit_detection_model_path} --save-txt --save-conf')

# find and gather digit label files from last detection
digit_detection_results_label_paths = get_latest_detection_label_paths(detect_folder)

# organize digit detection data into special classes
rotated_crops_detection_box_data_arrays = get_detection_box_data_for_filtered_images(digit_detection_results_label_paths, with_confidence=True)

# resolve digit intersections
filtered_rotated_crops_detection_box_data_arrays = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays)

# filter out least possible rotation among each pair for a flipped and non-flipped image
more_confident_detection_box_data_arrays = filter_out_less_probable_data_arrays(ROTATION_RESULT_FOLDER, filtered_rotated_crops_detection_box_data_arrays, logging_dataframe=crop_quality_dataframe)

# make a folder for rotated rods
create_folder_if_necessary(STRING_RESULT_FOLDER)

# merge found digits into string and output and save crop quality dataframe
group_and_write_strings_to_text_files(more_confident_detection_box_data_arrays, STRING_RESULT_FOLDER)
write_dataframe_sorted_by_name(crop_quality_dataframe, 'crop_quality.csv')

finish_time = time.time()

print('-------------- IMAGE QUALITY --------------')
get_full_stats_from_dataframe(image_quality_dataframe, type='image')
print('-------------- CROPS QUALITY --------------')
get_full_stats_from_dataframe(crop_quality_dataframe, type='crop')
print('-------------------------------------------')
print(f'The whole pipeline took {finish_time - start_time} seconds!')
