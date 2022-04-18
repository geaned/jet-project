import os
import cv2

import numpy as np
import scipy.ndimage as snd


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

def weight_rod_pixels_sum(txt_file_path):
    ans_sum = 0
    with open(txt_file_path, 'r') as f:
        for line in f.readlines():
            arr = [float(s) for s in line.split()[1:]]
            x_center, y_center, width, height = arr
            dist = np.sqrt((x_center - 0.5) ** 2 + (y_center - 0.5) ** 2)
            ans_sum += np.cos(dist) * width * height
    return ans_sum

def global_rods_on_periphery(txt_file_path) -> bool:
    """returns True, if rods are on the image periphery"""
    return weight_rod_pixels_sum(txt_file_path) < 0.094

def global_without_rods(txt_file_path) -> bool:
    """returns True, if our image has 0 rods"""
    with open(txt_file_path, 'r') as f:
        return len(f.readlines()) == 0

def global_too_many_rods(txt_file_path) -> bool:
    """returns True, if our image has too many rods"""
    with open(txt_file_path, 'r') as f:
        return len(f.readlines()) >= 30

def global_check_after_detection(path_to_txt_file):
    if global_without_rods(path_to_txt_file):
        return False, 'Image has 0 rods'
    if global_too_many_rods(path_to_txt_file):
        return False, 'Image has too many rods'
    if global_rods_on_periphery(path_to_txt_file):
        return False, 'Rods are on the image periphery'
