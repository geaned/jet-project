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
from pipeline_utils import download_if_file_not_present
from output_info import make_base_dataframe_for_paths
from output_info import write_dataframe_sorted_by_name
from output_info import get_full_stats_from_dataframe
from rotation import rotate_to_horizontal

RECOGNITION_BENCHMARK_FOLDER = os.path.join(os.path.dirname(__file__), 'deep-text-recognition-benchmark')
if os.path.exists(RECOGNITION_BENCHMARK_FOLDER):
    os.rename(RECOGNITION_BENCHMARK_FOLDER, os.path.join(os.path.dirname(__file__), 'deep_text_recognition_benchmark'))

from number_recognition import get_rosetta_predictions_and_confs
from number_recognition import write_strings_to_text_files

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--source', nargs=1, required=True, help='Source folder with images in .png format')
arg_parser.add_argument('--masks', default=False, action='store_true')
arg_parser.add_argument('--crops', default=False, action='store_true')
arg_parser.add_argument('--no-rotations', default=False, action='store_true')
exec_args = arg_parser.parse_args()

IMAGES_FOLDER = exec_args.source[0]
ROD_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rod_detection')
OCR_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'OCR')
ROTATION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rotation')

CROP_RESULT_FOLDER = 'crops'
MASKS_RESULT_FOLDER = 'masks' if exec_args.masks else None
ROTATION_RESULT_FOLDER = 'results'
STRING_RESULT_FOLDER = 'strings'
TEXT_STRINGS_IMAGES_PATH = 'text_strings'

DEVICE = 0 if torch.cuda.is_available() else 'cpu'

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
rotated_crops_array = rotate_to_horizontal(good_for_rotation_crops_arrays, model_path, masks_folder=MASKS_RESULT_FOLDER, logging_dataframe=crop_quality_dataframe)

del good_for_rotation_crops_arrays
gc.collect()

# make crops array for flipped crops
rotated_crops_array_flipped = make_flipped_crop_array(rotated_crops_array)

# download text strings detection model if not present
text_strings_detection_model_path = os.path.join(OCR_FOLDER, "ocr_weights.pt")
download_if_file_not_present('1h0r5XWEjtyc2-Dk0x6w-6jP0BveRTsUA', text_strings_detection_model_path)

# prepare data for detection (for flipped as well)
crop_images_names = [rotated_crop.get_file_name_for_crop_box() for rotated_crop in rotated_crops_array]
rotated_crops = [rotated_crop.img_tensor for rotated_crop in rotated_crops_array]
crop_images_names_flipped = [rotated_crop.get_file_name_for_crop_box() for rotated_crop in rotated_crops_array_flipped]
rotated_crops_flipped = [rotated_crop.img_tensor for rotated_crop in rotated_crops_array_flipped]

# detect text strings (for flipped as well)
text_strings_detection_model = torch.hub.load(
    'ultralytics/yolov5', 'custom', path=text_strings_detection_model_path, device=DEVICE,
)
text_strings_detection_model.conf = 0.6

text_strings_detection_results = text_strings_detection_model(rotated_crops, size=640)
text_strings_detection_results.print()
text_strings_detection_results_flipped = text_strings_detection_model(rotated_crops_flipped, size=640)
text_strings_detection_results_flipped.print()

text_strings_detection_results_tensors = text_strings_detection_results.xyxy
text_strings_detection_results_flipped_tensors = text_strings_detection_results_flipped.xyxy

# organize text strings detection data into special classes (for flipped as well)
rotated_crops_detection_box_data_arrays = organize_and_filter_detection_results(
    crop_images_names, text_strings_detection_results_tensors, rotated_crops,
)
rotated_crops_detection_box_data_arrays_flipped = organize_and_filter_detection_results(
    crop_images_names_flipped, text_strings_detection_results_flipped_tensors, rotated_crops_flipped,
)

# resolve text strings intersections (for flipped as well)
filtered_rotated_crops_detection_box_data_arrays = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays)
filtered_rotated_crops_detection_box_data_arrays_flipped = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays_flipped)

# filter out the least possible rotation among each pair for a non-flipped image and a flipped image
more_confident_detection_box_data_arrays, more_confident_crop_data = select_more_confident_data_arrays(filtered_rotated_crops_detection_box_data_arrays, filtered_rotated_crops_detection_box_data_arrays_flipped, rotated_crops_array, rotated_crops_array_flipped, logging_dataframe=crop_quality_dataframe)

del filtered_rotated_crops_detection_box_data_arrays
del filtered_rotated_crops_detection_box_data_arrays_flipped
del rotated_crops_array
del rotated_crops_array_flipped
gc.collect()

# save more confident rotated crops
if not exec_args.no_rotations:
    save_crops(ROTATION_RESULT_FOLDER, more_confident_crop_data)

# download rosetta recognition model if not present
ROSETTA_MODEL_PATH = os.path.join(OCR_FOLDER, "rosetta_with_gans.pth")
download_if_file_not_present('1fEfZfqRdz8Hb5IkplM2mxVPxiR1LBIDc', ROSETTA_MODEL_PATH)

# save cropped text strings
save_cropped_images(TEXT_STRINGS_IMAGES_PATH, more_confident_detection_box_data_arrays)

# recognize serial numbers and output them
rosetta_preds = get_rosetta_predictions_and_confs(ROSETTA_MODEL_PATH, TEXT_STRINGS_IMAGES_PATH)
write_strings_to_text_files(STRING_RESULT_FOLDER, rosetta_preds)

# output predicted serial numbers and save crop quality dataframe
write_dataframe_sorted_by_name(crop_quality_dataframe, 'crop_quality.csv')

finish_time = time.time()

print('-------------- IMAGE QUALITY --------------')
get_full_stats_from_dataframe(image_quality_dataframe, type='image')
print('-------------- CROPS QUALITY --------------')
get_full_stats_from_dataframe(crop_quality_dataframe, type='crop')
print('-------------------------------------------')
print(f'The whole pipeline took {finish_time - start_time:.2f} seconds!')
