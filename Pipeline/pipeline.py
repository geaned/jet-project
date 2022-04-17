import os
from PIL import Image

from detection_box_array import DetectionBoxData, DetectionBoxDataArray


def global_quality_check():
    pass


IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'datasets_not_split/data_all/img')
YOLO_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'yolov5')
ROD_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rod_detection')
DIGIT_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Digit_detection')
ROTATION_FOLDER = os.path.join(os.path.dirname(__file__), os.pardir, 'Rotation')

# run detection
os.system(f'python {os.path.join(YOLO_FOLDER, "detect.py")} --source {IMAGES_FOLDER} --img 1080 --weights {os.path.join(ROD_DETECTION_FOLDER, "rod_weights.pt")} --save-txt')

# find and gather label files
detect_folder = os.path.join(YOLO_FOLDER, "runs/detect")
rod_detection_results_folder = os.path.join(detect_folder, os.listdir(detect_folder)[-1], "labels") # actually should search for max index
print(f'Checking rod detection results folder {rod_detection_results_folder}')

rod_detection_results_files = [os.path.join(rod_detection_results_folder, file_name) for file_name in os.listdir(rod_detection_results_folder)]

# organized data into special classes
detection_box_data_arrays = []
for file_name in rod_detection_results_files:
    with open(file_name) as label_file:
        current_detection_box_data_array = []
        while True:
            line_values = label_file.readline().split()
            if not line_values:
                break

            class_num, center_x, center_y, width, height = int(line_values[0]), float(line_values[1]), float(line_values[2]), float(line_values[3]), float(line_values[4])
            current_detection_box_data_array.append(DetectionBoxData(class_num, center_x, center_y, width, height))

    detection_box_data_arrays.append(DetectionBoxDataArray(file_name.split('/')[-1].replace('.txt', '.png'), current_detection_box_data_array))

# make a folder for cropped_images
try:
    os.makedirs('crop_results')
    print('Folder "crop_results" created, saving cropped rods...')
except FileExistsError:
    print('Folder "crop_results" exists, saving cropped rods...')

# save cropped images
for detection_box_data_array in detection_box_data_arrays:
    img_path = os.path.join(IMAGES_FOLDER, detection_box_data_array.img_name)
    print(f'Saving cropped rods on image {img_path}')
    current_image = Image.open(img_path)
    image_width, image_height = current_image.size
    for idx, detection_box_data in enumerate(detection_box_data_array.box_array):
        center_x, center_y = detection_box_data.get_absolute_center(image_width, image_height)
        box_width, box_height = detection_box_data.get_absolute_dimensions(image_width, image_height)

        left, right = int(center_x - box_width / 2), int(center_x + box_width / 2)
        top, bottom = int(center_y - box_height / 2), int(center_y + box_height / 2)

        current_crop = current_image.crop((left, top, right, bottom))
        current_crop.save(os.path.join('crop_results', detection_box_data_array.img_name.replace('.png', f'_{idx}.png')))

# perform global quality check
global_quality_check()
