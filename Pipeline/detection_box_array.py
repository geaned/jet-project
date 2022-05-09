from typing import Tuple, List
import numpy as np
import torch


NEW_STRING_HEIGHT_DIFFERENCE_THRESHOLD = 0.8

class BasicCropData():
    def __init__(self, idx: int, img_name: str, img_tensor: torch.Tensor):
        self.img_name = img_name
        self.index = idx
        self.img_tensor = img_tensor

    def get_file_name_for_crop_box(self) -> str:
        return self.img_name.replace('.png', f'_{self.index}.png')    

class DetectionBoxData():
    def __init__(self, idx: int, class_num: int, img_tensor: torch.Tensor, left: float, top: float, right: float, bottom: float, confidence: float):
        self.index = idx
        self.class_num = class_num
        self.img_tensor = img_tensor
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom
        self.confidence = confidence
    
    def get_data(self) -> dict:
        return {
            'class_num': self.class_num,
            'bounding_box': {
                'left': self.left,
                'top': self.top,
                'right': self.right,
                'bottom': self.bottom,
                'confidence': self.confidence,
            },
        }
    
    def __str__(self) -> str:
        return str(self.get_data())

    def get_boundaries_coordinates(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (int(self.left), int(self.top), int(self.right), int(self.bottom))

    def get_top_left_and_bottom_right(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.left, self.top), (self.right, self.bottom)

    def get_area(self) -> float:
        return (self.right - self.left) * (self.bottom - self.top)

    def get_center(self) -> Tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def get_absolute_dimensions(self) -> Tuple[float, float]:
        return ((self.right - self.left), (self.bottom - self.top))

class DetectionBoxDataArray():
    def __init__(self, img_name: str, width: int, height: int, box_array: List[DetectionBoxData]):
        self.img_name = img_name
        self.img_width = width
        self.img_height = height
        self.box_array = box_array

    def __str__(self) -> str:
        return f'{self.img_name}: {[detection_box.get_data() for detection_box in self.box_array]}'

    def get_file_name_for_data_box(self, box_idx: int) -> str:
        return self.img_name.replace('.png', f'_{self.box_array[box_idx].index}.png')

    def get_image_dimensions(self):
        return (self.img_width, self.img_height)

    # may be rewritten with extra data
    def merge_digits_into_strings(self) -> List[str]:
        average_digit_height = np.mean([detection_box.get_absolute_dimensions()[1] for detection_box in self.box_array])

        item_not_parsed = [True]*len(self.box_array)

        found_strings = []

        while sum(item_not_parsed) > 0:
            pivotal_digit_idx = min(
                [idx for idx in range(len(self.box_array)) if item_not_parsed[idx]],
                key=lambda idx: self.box_array[idx].get_center()[1],
            )
            pivotal_element = self.box_array[pivotal_digit_idx]
            item_not_parsed[pivotal_digit_idx] = False

            current_row_idxs = [pivotal_digit_idx]
            for idx, other_element in enumerate(self.box_array):
                if abs(pivotal_element.get_center()[1] - other_element.get_center()[1]) < NEW_STRING_HEIGHT_DIFFERENCE_THRESHOLD * average_digit_height and item_not_parsed[idx]:
                    current_row_idxs.append(idx)
                    item_not_parsed[idx] = False
            
            # left = min([self.box_array[idx].left for idx in current_row_idxs])
            # top = min([self.box_array[idx].top for idx in current_row_idxs])
            # right = max([self.box_array[idx].right for idx in current_row_idxs])
            # bottom = min([self.box_array[idx].bottom for idx in current_row_idxs])

            sorted_row_idxs = sorted(current_row_idxs, key=lambda idx: self.box_array[idx].get_center()[0])
            new_found_string = ''.join([str(self.box_array[idx].class_num) for idx in sorted_row_idxs])

            # found_strings.append(((left, top, right, bottom), new_found_string))
            found_strings.append(new_found_string)
        
        return found_strings
