import numpy as np


def from_pascal_voc(bounding_box, target_format="pascal_voc", rounding=True):
    assert target_format in ("yolo", "coco"), f"'target_format' must be one of ['yolo', 'coco'], but given '{target_format}'."
    x_min, y_min, x_max, y_max = bounding_box
        
    width = x_max - x_min
    height = y_max - y_min
        
    half_width = width / 2
    half_height = height / 2
        
    if target_format == "coco":
        formated_bounding_box = [x_min, y_min, width, height]
            
    elif target_format == "yolo":
        x_center = x_max / 2
        y_center = y_max / 2
            
        formated_bounding_box = [x_center, y_center, width, height]
            
    else:
        formated_bounding_box = bounding_box
            
    formated_bounding_box = np.array(formated_bounding_box)
    
    if rounding:
        formated_bounding_box = formated_bounding_box.round()
    
    return formated_bounding_box


def from_coco(bounding_box, target_format="pascal_voc", rounding=True):
    assert target_format in ("pascal_voc", "yolo"), f"'target_format' must be one of ['pascal_voc', 'yolo'], but given '{target_format}'."
    
    x_min, y_min, width, height = bounding_box 
        
    x_max = x_min + width
    y_max = y_min + height
        
    if target_format == "pascal_voc":
        formated_bounding_box = [x_min, y_min, x_max, y_max]
            
    elif target_format == "yolo":
        x_center = x_max / 2
        y_center = y_max / 2
            
        formated_bounding_box = [x_center, y_center, width, height]
            
    else:
        formated_bounding_box = bounding_box
            
    formated_bounding_box = np.array(formated_bounding_box)
    
    if rounding:
        formated_bounding_box = formated_bounding_box.round()
        
    return formated_bounding_box
    

def from_yolo(bounding_box, target_format="pascal_voc", rounding=True):
    assert target_format in ("pascal_voc", "coco"), f"'target_format' must be one of ['pascal_voc', 'coco'], but given '{target_format}'."
    
    x_center, y_center, width, height = bounding_box
        
    half_width = width / 2
    half_height = height / 2
        
    x_max = x_center + half_width
    x_min = x_center - half_width
    y_max = y_center + half_height
    y_min = y_center - half_height
        
    if target_format == "pascal_voc":
        formated_bounding_box = [x_min, y_min, x_max, y_max]
            
    elif target_format == "coco":
        formated_bounding_box = [x_min, y_min, width, height]
        
    else:
        formated_bounding_box = bounding_box
        
    formated_bounding_box = np.array(formated_bounding_box)
    
    if rounding:
        formated_bounding_box = formated_bounding_box.round()
    
    return formated_bounding_box