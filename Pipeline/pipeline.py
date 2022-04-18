import os
import argparse
from PIL import Image

from detection_box_array import DetectionBoxData, DetectionBoxDataArray
from quality_metrics import global_check_before_detection, global_check_after_detection
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

images_files = [os.path.join(IMAGES_FOLDER, file_name) for file_name in os.listdir(IMAGES_FOLDER)]
good_image_names = set()

# run global quality evaluation
for file_path in images_files:
    print(f'Checking image quality for {file_path}... ', end='')
    is_image_good, reason_for_bad = global_check_before_detection(file_path)
    print(f'{"Passed!" if is_image_good else "Failed! Reason: "} {reason_for_bad}')

    if is_image_good:
        good_image_names.add(file_path.split('/')[-1])

# run detection
os.system(f'python {os.path.join(YOLO_FOLDER, "detect.py")} --source {IMAGES_FOLDER} --img 1080 --weights {os.path.join(ROD_DETECTION_FOLDER, "rod_weights.pt")} --conf-thres 0.7 --save-txt')

# find and gather label files
detect_folder = os.path.join(YOLO_FOLDER, "runs/detect")
latest_exp = max(
    os.listdir(detect_folder),
    key=lambda file_name: int(file_name.replace('exp', '') if file_name != 'exp' else '1'),
)
rod_detection_results_folder = os.path.join(detect_folder, latest_exp, "labels")
print(f'Checking rod detection results folder {rod_detection_results_folder}...')

rod_detection_results_files = [os.path.join(rod_detection_results_folder, file_name) for file_name in os.listdir(rod_detection_results_folder)]

# organize detection data into special classes
good_detection_box_data_arrays = []
for file_path in rod_detection_results_files:
    print(f'Checking detection results from {file_path}... ', end='')
    current_image_name = file_path.split('/')[-1].replace('.txt', '.png')
    if current_image_name in good_image_names:
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

    # run global quality evaultion on after-detection features
    if is_image_good_after_detection:
        print('Passed!')
        good_detection_box_data_arrays.append(current_data_array)
    else:
        print(f'Failed! Reason: {reason_for_bad_after_detection}')

# make a folder for cropped_images
try:
    os.makedirs('crop_results')
    print('Folder "crop_results" created, saving cropped rods...')
except FileExistsError:
    print('Folder "crop_results" exists, saving cropped rods...')

# save cropped images
for detection_box_data_array in good_detection_box_data_arrays:
    img_path = os.path.join(IMAGES_FOLDER, detection_box_data_array.img_name)
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

# perform local quality check for each crop
# ...

# make a folder for cropped_images
try:
    os.makedirs('rotation_results')
    print('Folder "rotation_results" created, rotating rods...')
except FileExistsError:
    print('Folder "rotation_results" exists, rotating rods...')

# rotate crops to make text horizontal
rotate_to_horizontal(
    os.listdir('crop_results'),
    os.path.join(ROTATION_FOLDER, 'text_detection_model.pth'),
)
