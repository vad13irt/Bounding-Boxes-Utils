import numpy as np
from single_functions import transform_bounding_box, normalize_bounding_box, unnormalize_bounding_box


def transform_bounding_boxes(bounding_boxes, source_format="pascal_voc", target_format="pascal_voc", rounding=True):
    transformed_bounding_boxes = []
    for bounding_box in bounding_boxes:
        transformed_bounding_box = transform_bounding_box(bounding_box, source_format=source_format, target_format=target_format, rounding=True)
        transformed_bounding_boxes.append(transformed_bounding_box)
        
    transformed_bounding_boxes = np.array(transformed_bounding_boxes)
    
    return transformed_bounding_boxes
        

def normalize_bounding_boxes(bounding_boxes, width, height, source_format="pascal_voc"):
    normalized_bounding_boxes = []
    
    for bounding_box in bounding_boxes:
        normalized_bounding_box = normalize_bounding_box(bounding_box=bounding_box, 
                                                         width=width, 
                                                         height=height, 
                                                         source_format=source_format)
        
        normalized_bounding_boxes.append(normalized_bounding_box)

    normalized_bounding_boxes = np.array(normalized_bounding_boxes)
    return normalized_bounding_boxes
        

def unnormalize_bounding_boxes(bounding_boxes, width, height, source_format="pascal_voc"):
    unnormalized_bounding_boxes = []
    
    for bounding_box in bounding_boxes:
        unnormalized_bounding_box = unnormalize_bounding_box(bounding_box=bounding_box, 
                                                             width=width, 
                                                             height=height, 
                                                             source_format=source_format)
        
        unnormalized_bounding_boxes.append(unnormalized_bounding_box)

    unnormalized_bounding_boxes = np.array(unnormalized_bounding_boxes)
    return unnormalized_bounding_boxes