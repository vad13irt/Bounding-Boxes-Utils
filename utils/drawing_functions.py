import cv2
from general_functions import transform_bounding_boxes


def draw_bounding_boxes(image, bounding_boxes, source_format="pascal_voc", color=(0, 255, 255), thickness=1):
    image_with_bboxes = image.copy()
    bounding_boxes = transform_bounding_boxes(bounding_boxes, source_format=source_format, target_format="pascal_voc", rounding=True)

    for bounding_box in bounding_boxes:
        x_min, y_min, x_max, y_max = bounding_box.astype(int)
        image_with_bboxes = cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)
        
    return image_with_bboxes