import streamlit as st
from PIL import Image, ImageDraw
import torchvision.transforms as tt
import os
import sys
import cv2
import torch

sys.path.append('Pipeline')

from Pipeline.parsing import organize_and_filter_detection_results
from Pipeline.parsing import get_crop_image_names_from_array
from Pipeline.parsing import leave_only_good_crops
from Pipeline.parsing import make_flipped_crop_array
from Pipeline.parsing import remove_overlapping_bounding_boxes_by_iou
from Pipeline.parsing import select_more_confident_data_arrays
from Pipeline.pipeline_utils import download_if_file_not_present
from Pipeline.quality_metrics import global_check_after_detection
from Pipeline.quality_metrics import global_check_before_detection
from Pipeline.output_info import make_base_dataframe_for_paths
from Pipeline.rotation import rotate_to_horizontal


def show_rod_detection_results(rod_detection_df):
    st.header('Rod detection results')
    st.dataframe(rod_detection_df.pandas().xyxy[0].drop(columns=['class', 'name']).rename(columns={'xmin': 'Left', 'ymin': 'Top', 'xmax': 'Right', 'ymax': 'Bottom', 'confidence': 'Confidence'}))
    rod_image = tt.ToPILImage()(cv2.cvtColor(image_tensor, cv2.COLOR_BGR2RGB))
    canvas_rod_image = ImageDraw.Draw(rod_image)

    for crop in rod_detection_df.xyxy[0]:
        left, top, right, bottom, _, _ = crop
        canvas_rod_image.rectangle([(left, top), (right, bottom)], outline='red', width=10)

    st.image(rod_image)

def show_quality_verdict(quality_df):
    st.header('Quality verdict')
    st.dataframe(quality_df.drop(columns=['Name', 'Good']))


DEVICE = 0 if torch.cuda.is_available() else 'cpu'

st.header('Serial number detection')
st.write('Select an image and choose image area via sidebar')

IMAGES_FOLDER = os.path.join(os.path.dirname(__file__), 'showcase_images')
ROD_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), 'Rod_detection')
DIGIT_DETECTION_FOLDER = os.path.join(os.path.dirname(__file__), 'Digit_detection')
ROTATION_FOLDER = os.path.join(os.path.dirname(__file__), 'Rotation')

found_images_names = [os.path.join(IMAGES_FOLDER, image_name) for image_name in sorted(os.listdir(IMAGES_FOLDER))]
chosen_image_path = st.sidebar.selectbox('Choose an image', found_images_names, index=0, format_func=os.path.basename)

image = Image.open(chosen_image_path)
canvas_image = ImageDraw.Draw(image)
left, right = st.sidebar.slider(label='Horizontal bounds', min_value=0, max_value=image.size[0], value=(0, image.size[0]))
top, bottom = st.sidebar.slider(label='Vertical bounds', min_value=0, max_value=image.size[1], value=(0, image.size[1]))

canvas_image.rectangle([(left, top), (right, bottom)], outline='blue', width=10)

st.image(image)

image_tensor = cv2.imread(chosen_image_path)[top:bottom, left:right]

if st.sidebar.button('Run the pipeline'):
    if image_tensor.size == 0:
        st.error('Crop with zero area selected')
        st.stop()

    with st.spinner('Processing...'):
        is_preemptively_good, reason_for_preemptively_bad = global_check_before_detection(image_tensor)
        if not is_preemptively_good:
            st.error(f'Image did not pass preemptive global quality evaluation! Reason: {reason_for_preemptively_bad}')
            st.stop()

        rod_detection_model_path = os.path.join(ROD_DETECTION_FOLDER, "rod_weights.pt")
        download_if_file_not_present('1UZQ1YW9Fap2ESzfMgYhUhC4SdTUZK3mw', rod_detection_model_path)

        rod_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=rod_detection_model_path, device=DEVICE)
        rod_detection_model.conf = 0.7

        rod_detection_results = rod_detection_model([image_tensor], size=1080)
        rod_detection_results_tensors = rod_detection_results.xyxy

        actually_good_detection_box_data_arrays = organize_and_filter_detection_results(
            [os.path.basename(chosen_image_path)],
            rod_detection_results_tensors,
            [image_tensor],
            apply_local_filter=False
        )

        is_finally_good, reason_for_finally_bad = global_check_after_detection(actually_good_detection_box_data_arrays[0])
        if not is_finally_good:
            st.error(f'Image did not pass final global quality evaluation! Reason: {reason_for_finally_bad}')
            show_rod_detection_results(rod_detection_results)
            st.stop()

        crop_file_paths = get_crop_image_names_from_array(actually_good_detection_box_data_arrays)
        crop_quality_dataframe = make_base_dataframe_for_paths(crop_file_paths)

        good_for_rotation_crops_arrays = leave_only_good_crops(actually_good_detection_box_data_arrays, logging_dataframe=crop_quality_dataframe)

        if len(good_for_rotation_crops_arrays[0].box_array) == 0:
            st.error(f'None of the rods passed local quality evaluation!')
            show_rod_detection_results(rod_detection_results)
            show_quality_verdict(crop_quality_dataframe)
            st.stop()

        model_path = os.path.join(ROTATION_FOLDER, 'text_segmentation_model.pth')
        rotated_crops_array = rotate_to_horizontal(good_for_rotation_crops_arrays, model_path, logging_dataframe=crop_quality_dataframe)

        if len(rotated_crops_array) == 0:
            print(f'Could not detect text on any of the rods!')
            show_rod_detection_results(rod_detection_results)
            show_quality_verdict(crop_quality_dataframe)
            st.stop()

        rotated_crops_array_flipped = make_flipped_crop_array(rotated_crops_array)

        digit_detection_model_path = os.path.join(DIGIT_DETECTION_FOLDER, "digit_weights.pt")
        download_if_file_not_present('1KTLSkFf8-VuwtPRArxLQINzUhKhY2Qle', digit_detection_model_path)

        crop_images_names = [rotated_crop.get_file_name_for_crop_box() for rotated_crop in rotated_crops_array]
        rotated_crops = [rotated_crop.img_tensor for rotated_crop in rotated_crops_array]
        crop_images_names_flipped = [rotated_crop.get_file_name_for_crop_box() for rotated_crop in rotated_crops_array_flipped]
        rotated_crops_flipped = [rotated_crop.img_tensor for rotated_crop in rotated_crops_array_flipped]

        digit_detection_model = torch.hub.load('ultralytics/yolov5', 'custom', path=digit_detection_model_path, device=DEVICE)

        digit_detection_results = digit_detection_model(rotated_crops, size=640)
        digit_detection_results.print()
        digit_detection_results_flipped = digit_detection_model(rotated_crops_flipped, size=640)
        digit_detection_results_flipped.print()

        digit_detection_results_tensors = digit_detection_results.xyxy
        digit_detection_results_flipped_tensors = digit_detection_results_flipped.xyxy

        rotated_crops_detection_box_data_arrays = organize_and_filter_detection_results(crop_images_names, digit_detection_results_tensors, rotated_crops)
        rotated_crops_detection_box_data_arrays_flipped = organize_and_filter_detection_results(crop_images_names_flipped, digit_detection_results_flipped_tensors, rotated_crops_flipped)


        filtered_rotated_crops_detection_box_data_arrays = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays)
        filtered_rotated_crops_detection_box_data_arrays_flipped = remove_overlapping_bounding_boxes_by_iou(rotated_crops_detection_box_data_arrays_flipped)

        more_confident_detection_box_data_arrays, more_confident_crop_data = select_more_confident_data_arrays(filtered_rotated_crops_detection_box_data_arrays, filtered_rotated_crops_detection_box_data_arrays_flipped, rotated_crops_array, rotated_crops_array_flipped, logging_dataframe=crop_quality_dataframe)

    st.success('Done!')
    
    show_rod_detection_results(rod_detection_results)
    show_quality_verdict(crop_quality_dataframe)

    for detection_box_data_array, crop_data in zip(more_confident_detection_box_data_arrays, more_confident_crop_data):
        if sum([data_box.confidence for data_box in detection_box_data_array.box_array]) == 0:
            continue

        st.header(detection_box_data_array.img_name)
        st.image(crop_data.img_tensor, channels='BGR')
        st.write(
            'With an average digit-wise confidence of',
            float('{0:.4f}'.format(
                sum([detection_box_data_array.box_array[idx].confidence for idx in range(len(detection_box_data_array.box_array))]) / len(detection_box_data_array.box_array), 'found'
            ))
        )
        st.write('Found', detection_box_data_array.merge_digits_into_strings())
