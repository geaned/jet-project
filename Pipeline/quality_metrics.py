from typing import Tuple
import numpy as np
import cv2
import torch
import scipy.ndimage as snd

from detection_box_array import DetectionBoxDataArray, DetectionBoxData

GLOBAL_HIGHLIGHT_LOWER_THRESHOLD = 0.804
GLOBAL_BLUR_UPPER_THRESHOLD = 35.992
RODS_ON_PERIPHERY_UPPER_THRESHOLD = 0.078
TOO_MANY_RODS_LOWER_THRESHOLD = 30
LOCAL_HIGHLIGHT_LOWER_THRESHOLD = 0.912
LOCAL_BLUR_UPPER_THRESHOLD = 510.477
LOCAL_ROTATED_TOO_MUCH_UPPER_THRESHOLD = 0.777
SMALL_ROD_SQUARE_UPPER_THRESHOLD = 0.0058

def sta6_optimized(gray_img: torch.Tensor, conv_size: int = 5, stride: int = 1):
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

def global_highlight(gray_img: torch.Tensor, lower_threshold: float) -> bool:
    """returns True, if image is too bright"""
    intensity = gray_img / 255
    mean_intensity = np.mean(intensity)

    return mean_intensity > lower_threshold

def global_too_blurry(gray_img: torch.Tensor, upper_threshold: float) -> bool:
    """returns True, if image is too blurry"""
    return sta6_optimized(gray_img) * 1e7 <= upper_threshold

def global_check_before_detection(image_tensor: torch.Tensor):
    gray_tensor = cv2.cvtColor(image_tensor, cv2.COLOR_BGR2GRAY)

    if global_highlight(gray_tensor, GLOBAL_HIGHLIGHT_LOWER_THRESHOLD):
        return False, 'Image is too bright'
    if global_too_blurry(gray_tensor, GLOBAL_BLUR_UPPER_THRESHOLD):
        return False, 'Image is too blurry'

    return True, ''

def weight_rod_pixels_sum(detection_box_data_array: DetectionBoxDataArray) -> int:
    ans_sum = 0

    image_width, image_height = detection_box_data_array.get_image_dimensions()
    for detection_box_data in detection_box_data_array.box_array:
        center_x, center_y = detection_box_data.get_center()
        box_width, box_height = detection_box_data.get_absolute_dimensions()

        relative_x_offset = abs((center_x - image_width / 2) / image_width)
        relative_y_offset = abs((center_y - image_height / 2) / image_height)

        dist = np.sqrt(relative_x_offset ** 2 + relative_y_offset ** 2)
        ans_sum += np.cos(dist) * (box_width / image_width) * (box_height / image_height)

    return ans_sum

def global_rods_on_periphery(detection_box_data_array: DetectionBoxDataArray, upper_threshold: float) -> bool:
    """returns True, if rods are on the image periphery"""
    return weight_rod_pixels_sum(detection_box_data_array) < upper_threshold

def global_without_rods(detection_box_data_array: DetectionBoxDataArray) -> bool:
    """returns True, if our image has no rods"""
    return len(detection_box_data_array.box_array) == 0

def global_too_many_rods(detection_box_data_array: DetectionBoxDataArray, lower_threshold: float) -> bool:
    """returns True, if our image has too many rods"""
    return len(detection_box_data_array.box_array) >= lower_threshold

def global_check_after_detection(detection_box_data_array: DetectionBoxDataArray) -> Tuple[bool, str]:
    if global_without_rods(detection_box_data_array):
        return False, 'Image has no rods'
    if global_too_many_rods(detection_box_data_array, TOO_MANY_RODS_LOWER_THRESHOLD):
        return False, 'Image has too many rods'
    if global_rods_on_periphery(detection_box_data_array, RODS_ON_PERIPHERY_UPPER_THRESHOLD):
        return False, 'The rods are on the image periphery'
    
    return True, ''

def local_highlight(gray_img: np.ndarray, lower_threshold: float) -> bool:
    """returns True, if cropped image is too bright"""
    intensity = gray_img / 255
    mean_intensity = np.mean(intensity)
    return mean_intensity > lower_threshold

def local_too_blurry(gray_img: np.ndarray, upper_threshold: float) -> bool:
    """returns True, if cropped image is too blurry"""
    return sta6_optimized(gray_img) * 1e7 <= upper_threshold

def local_rotated_too_much(width: int, height: int, upper_threshold: float) -> bool:
    """returns True, if rod image is rotated too much"""
    size_ratio = min(width, height) / max(width, height)
    return size_ratio < upper_threshold

def local_small_rod_square(relative_width: float, relative_height: float, upper_threshold: float) -> bool:
    """
    Args: height and width are float from 0 to 1
    Returns: True if rod square is too small
    """
    return relative_width * relative_height < upper_threshold

def local_check_after_detection(detection_box: DetectionBoxData, width: float, height: float) -> Tuple[bool, str]:
    gray_tensor = cv2.cvtColor(detection_box.img_tensor, cv2.COLOR_RGB2GRAY)
    box_width, box_height = detection_box.get_absolute_dimensions()

    if local_highlight(gray_tensor, LOCAL_HIGHLIGHT_LOWER_THRESHOLD):
        return False, 'Crop image is too bright'

    if local_too_blurry(gray_tensor, LOCAL_BLUR_UPPER_THRESHOLD):
        return False, 'Crop image is too blurry'

    if local_rotated_too_much(box_width, box_height, LOCAL_ROTATED_TOO_MUCH_UPPER_THRESHOLD):
        return False, 'Crop area is too thin'

    if local_small_rod_square(box_width / width, box_height / height, SMALL_ROD_SQUARE_UPPER_THRESHOLD):
        return False, 'Crop square is too small'

    return True, ''
