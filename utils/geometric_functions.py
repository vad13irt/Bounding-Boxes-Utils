import numpy as np
from single_functions import transform_bounding_box
from scipy.spatial  import distance


def get_bounding_box_area(bounding_box, source_format="pascal_voc"):
    if source_format != "pascal_voc":
        bounding_box = transform_bounding_box(bounding_box=bounding_box, 
                                              source_format=source_format, 
                                              target_format="pascal_voc", 
                                              rounding=True)
    
    x_min, y_min, x_max, y_max = bounding_box
    
    top_rib_length = distance.euclidean((x_min, y_max), (x_max, y_max))
    left_rib_length = distance.euclidean((x_min, y_min), (x_min, y_max))
    
    area = np.float(top_rib_length * left_rib_length)
    return area



def get_bounding_box_coordinates(bounding_box, source_format="pascal_voc"):
    if source_format != "pascal_voc":
        bounding_box = transform_bounding_box(bounding_box=bounding_box, 
                                              source_format=source_format, 
                                              target_format="pascal_voc", 
                                              rounding=True)
        
    x_min, y_min, x_max, y_max = bounding_box
    
    left_bottom_coordinates = (x_min, y_min)
    left_top_coordinates = (x_min, y_max)
    right_bottom_coordinates = (x_max, y_min)
    right_top_coordinates = (x_max, y_max)
    
    coordinates = np.array([left_top_coordinates, right_top_coordinates, right_bottom_coordinates, left_bottom_coordinates])
    
    return coordinates