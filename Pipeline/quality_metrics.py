import os
import cv2

from typing import Tuple

import numpy as np
import scipy.ndimage as snd

from detection_box_array import DetectionBoxDataArray, DetectionBoxData

GLOBAL_HIGHLIGHT_LOWER_THRESHOLD = 0.804
GLOBAL_BLUR_UPPER_THRESHOLD = 35.992
RODS_ON_PERIPHERY_UPPER_THRESHOLD = 0.094
TOO_MANY_RODS_LOWER_THRESHOLD = 30
LOCAL_HIGHLIGHT_LOWER_THRESHOLD = 0.912
LOCAL_BLUR_UPPER_THRESHOLD = 688.078
LOCAL_ROTATED_TOO_MUCH_UPPER_THRESHOLD = 0.811
SMALL_ROD_SQUARE_UPPER_THRESHOLD = 0.0058

def sta6_optimized(gray_img: np.ndarray, conv_size: int = 5, stride: int = 1):
    image_intensity = gray_img / 255

    mean_kernel = np.ones((conv_size, conv_size)) / conv_size ** 2
    image_mean_intensity = snd.convolve(image_intensity, mean_kernel)
    image_mean_intensity = image_mean_intensity[(conv_size - 1) // 2:image_mean_intensity.shape[0] - conv_size // 2,
                           (conv_size - 1) // 2:image_mean_intensity.shape[1] - conv_size // 2]
    image_intensity = image_intensity[(conv_size - 1) // 2:image_intensity.shape[0] - conv_size // 2,
                      (conv_size - 1) // 2:image_intensity.shape[1] - conv_size // 2]

    image_intensity = image_intensity[::stride, ::stride]
    image_mean_intensity = image_mean_intensity[::stride, ::stride]

    sta6 = np.mean(np.power(image_intensity - image_mean_intensity, 2))

    return sta6

def global_highlight(gray_img: np.ndarray) -> bool:
    """returns True, if image is too bright"""
    intensity = gray_img / 255
    mean_intensity = np.mean(intensity)

    return mean_intensity > GLOBAL_HIGHLIGHT_LOWER_THRESHOLD

def global_too_blurry(gray_img: np.ndarray) -> bool:
    """returns True, if image is too blurry"""
    return sta6_optimized(gray_img) * 1e7 <= GLOBAL_BLUR_UPPER_THRESHOLD

def global_check_before_detection(path_to_image):
    image = cv2.imread(path_to_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if global_highlight(gray):
        return False, 'Image is too bright'
    if global_too_blurry(gray):
        return False, 'Image is too blurry'

    return True, ''

def weight_rod_pixels_sum(detection_box_data_array: DetectionBoxDataArray) -> int:
    ans_sum = 0

    for detection_box_data in detection_box_data_array.box_array:
        bouding_box = detection_box_data.get_data()['bounding_box']
        dist = np.sqrt((bouding_box['center_x'] - 0.5) ** 2 + (bouding_box['center_y'] - 0.5) ** 2)
        ans_sum += np.cos(dist) * bouding_box['width'] * bouding_box['height']

    return ans_sum

def global_rods_on_periphery(detection_box_data_array: DetectionBoxDataArray) -> bool:
    """returns True, if rods are on the image periphery"""
    return weight_rod_pixels_sum(detection_box_data_array) < RODS_ON_PERIPHERY_UPPER_THRESHOLD

def global_without_rods(detection_box_data_array: DetectionBoxDataArray) -> bool:
    """returns True, if our image has 0 rods"""
    return len(detection_box_data_array.box_array) == 0

def global_too_many_rods(detection_box_data_array: DetectionBoxDataArray) -> bool:
    """returns True, if our image has too many rods"""
    return len(detection_box_data_array.box_array) >= TOO_MANY_RODS_LOWER_THRESHOLD

def global_check_after_detection(detection_box_data_array: DetectionBoxDataArray) -> Tuple[bool, str]:
    if global_without_rods(detection_box_data_array):
        return False, 'Image has 0 rods'
    if global_too_many_rods(detection_box_data_array):
        return False, 'Image has too many rods'
    if global_rods_on_periphery(detection_box_data_array):
        return False, 'Rods are on the image periphery'
    
    return True, ''

def local_highlight(gray_img: np.ndarray) -> bool:
    """returns True, if cropped image is too bright"""
    intensity = gray_img / 255
    mean_intensity = np.mean(intensity)
    return mean_intensity > LOCAL_HIGHLIGHT_LOWER_THRESHOLD

def local_too_blurry(gray_img: np.ndarray) -> bool:
    """returns True, if cropped image is too blurry"""
    return sta6_optimized(gray_img) * 1e7 <= LOCAL_BLUR_UPPER_THRESHOLD

def local_rotated_too_much(gray_img: np.ndarray) -> bool:
    """returns True, if rod image is rotated too much"""
    height, width = gray_img.shape
    size_ratio = min(height, width) / max(height, width)
    return size_ratio < LOCAL_ROTATED_TOO_MUCH_UPPER_THRESHOLD

def local_small_rod_square(height: float, width: float) -> bool:
    """
    Args: height and width are float from 0 to 1
    Returns: True if rod square is too small
    """
    return height * width < SMALL_ROD_SQUARE_UPPER_THRESHOLD

def local_check_after_detection(path_to_image, detection_box: DetectionBoxData) -> Tuple[bool, str]:
    image = cv2.imread(path_to_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    data = detection_box.get_data()['bounding_box']
    height, width = data['height'], data['width']
    if local_highlight(gray):
        return False, 'Image is too bright'
    if local_too_blurry(gray):
        return False, 'Image is too blurry'
    if local_rotated_too_much(gray):
        return False, 'Crop area is too thin'
    if local_small_rod_square(height, width):
        return False, 'Crop square is too small'

    return True, ''
