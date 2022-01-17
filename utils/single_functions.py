import numpy as np
from format_functions import from_pascal_voc, from_coco, from_yolo


def transform_bounding_box(bounding_box, source_format="pascal_voc", target_format="pascal_voc", rounding=True):
    methods = {
        "pascal_voc": from_pascal_voc,
        "coco": from_coco,
        "yolo": from_yolo,
    }
    
    assert source_format in methods, f"'source_format' must be one of ['pascal_voc', 'yolo', 'coco'], but given '{source_format}'."
    assert target_format in methods, f"'target_format' must be one of ['pascal_voc', 'yolo', 'coco'], but given '{target_format}'."
    
    from_method = methods.get(source_format, from_pascal_voc)
        
    transformed_bounding_box = from_method(bounding_box=bounding_box, target_format=target_format, rounding=rounding)
    return transformed_bounding_box


def normalize_bounding_box(bounding_box, width, height, source_format="pascal_voc"):
    assert width >= 0, f"'width' must be in range (0, +inf), but given {width}."
    assert height >= 0, "'height' must be in range (0, +inf), but given {height}."
    

    x_min, y_min, x_max, y_max = transform_bounding_box(bounding_box=bounding_box, source_format=source_format, target_format="pascal_voc")
    
    normalized_x_max = x_max / width
    normalized_x_min = x_min / width
    
    normalized_y_max = y_max / height
    normalized_y_min = y_min / height
    
    normalized_bounding_box = np.array((normalized_x_min, normalized_y_min, normalized_x_max, normalized_y_max)) 
    normalized_bounding_box = transform_bounding_box(bounding_box=normalized_bounding_box, source_format="pascal_voc", target_format=source_format, rounding=False)
    
    return normalized_bounding_box