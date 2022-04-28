from typing import Tuple, List, Optional
import numpy as np


NEW_STRING_HEIGHT_DIFFERENCE_THRESHOLD = 0.8

class DetectionBoxDataArray():
    def __init__(self, img_name, box_array):
        self.img_name = img_name
        self.box_array = box_array

    # may be rewritten with extra data
    def merge_digits_into_strings(self) -> List[str]:
        average_digit_height = np.mean([detection_box.height for detection_box in self.box_array])

        item_not_parsed = [True]*len(self.box_array)

        found_strings = []

        while sum(item_not_parsed) > 0:
            pivotal_digit_idx = min(
                [idx for idx in range(len(self.box_array)) if item_not_parsed[idx]],
                key=lambda idx: self.box_array[idx].center_y,
            )
            pivotal_element = self.box_array[pivotal_digit_idx]
            item_not_parsed[pivotal_digit_idx] = False

            current_row_idxs = [pivotal_digit_idx]
            for idx, other_element in enumerate(self.box_array):
                if abs(pivotal_element.center_y - other_element.center_y) < NEW_STRING_HEIGHT_DIFFERENCE_THRESHOLD * average_digit_height and item_not_parsed[idx]:
                    current_row_idxs.append(idx)
                    item_not_parsed[idx] = False
            
            sorted_row_idxs = sorted(current_row_idxs, key=lambda idx: self.box_array[idx].center_x)
            new_found_string = ''.join([str(self.box_array[idx].class_num) for idx in sorted_row_idxs])

            found_strings.append(new_found_string)
        
        return found_strings


class DetectionBoxData():
    def __init__(self, class_num, center_x, center_y, width, height, confidence: Optional[int] = None):
        self.class_num = class_num
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
        self.confidence = confidence
    
    def get_data(self) -> dict:
        return {
            'class_num': self.class_num,
            'bounding_box': {
                'center_x': self.center_x,
                'center_y': self.center_y,
                'width': self.width,
                'height': self.height,
            },
        }
    
    def __str__(self) -> str:
        return str(self.get_data())

    def get_top_left_and_bottom_right(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        return (self.center_x - self.width / 2, self.center_y - self.height / 2), (self.center_x + self.width / 2, self.center_y + self.height / 2)

    def get_relative_area(self) -> float:
        return self.width * self.height

    def get_absolute_center(self, image_width, image_height) -> Tuple[float, float]:
        return (self.center_x * image_width, self.center_y * image_height)

    def get_absolute_dimensions(self, image_width, image_height) -> Tuple[float, float]:
        return (self.width * image_width, self.height * image_height)
