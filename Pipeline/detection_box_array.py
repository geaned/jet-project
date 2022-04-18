from typing import Tuple


class DetectionBoxDataArray():
    def __init__(self, img_name, box_array):
        self.img_name = img_name
        self.box_array = box_array


class DetectionBoxData():
    def __init__(self, class_num, center_x, center_y, width, height):
        self.class_num = class_num
        self.center_x = center_x
        self.center_y = center_y
        self.width = width
        self.height = height
    
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

    def get_absolute_center(self, image_width, image_height) -> Tuple[int, int]:
        return (self.center_x * image_width, self.center_y * image_height)

    def get_absolute_dimensions(self, image_width, image_height) -> Tuple[int, int]:
        return (self.width * image_width, self.height * image_height)
