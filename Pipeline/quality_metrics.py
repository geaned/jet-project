import os
import cv2

import numpy as np
import scipy.ndimage as snd

from detection_box_array import DetectionBoxDataArray

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

    return mean_intensity > 0.804

def global_too_blurry(gray_img: np.ndarray) -> bool:
    """returns True, if image is too blurry"""
    return sta6_optimized(gray_img) * 1e7 <= 35.992

def global_check_before_detection(path_to_image):
    image = cv2.imread(path_to_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if global_highlight(gray):
        return False, 'Image is too bright'
    if global_too_blurry(gray):
        return False, 'Image is too blurry'

    return True, ''

def weight_rod_pixels_sum(detection_box_data_array):
    ans_sum = 0

    for detection_box_data in detection_box_data_array.box_array:
        bouding_box = detection_box_data.get_data()['bounding_box']
        dist = np.sqrt((bouding_box['center_x'] - 0.5) ** 2 + (bouding_box['center_y'] - 0.5) ** 2)
        ans_sum += np.cos(dist) * bouding_box['width'] * bouding_box['height']

    return ans_sum

def global_rods_on_periphery(detection_box_data_array) -> bool:
    """returns True, if rods are on the image periphery"""
    return weight_rod_pixels_sum(detection_box_data_array) < 0.094

def global_without_rods(detection_box_data_array) -> bool:
    """returns True, if our image has 0 rods"""
    return len(detection_box_data_array.box_array) == 0

def global_too_many_rods(detection_box_data_array) -> bool:
    """returns True, if our image has too many rods"""
    return len(detection_box_data_array.box_array) >= 30

def global_check_after_detection(detection_box_data_array):
    if global_without_rods(detection_box_data_array):
        return False, 'Image has 0 rods'
    if global_too_many_rods(detection_box_data_array):
        return False, 'Image has too many rods'
    if global_rods_on_periphery(detection_box_data_array):
        return False, 'Rods are on the image periphery'
    
    return True, ''