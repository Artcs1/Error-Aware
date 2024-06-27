from PIL import Image                                      # (pip install Pillow)
import numpy as np                                         # (pip install numpy)
from skimage import measure                                # (pip install scikit-image)
import os
import json




def get_coco_json_format():
    # Standard COCO format 
    coco_format = {
        "info": {},
        "licenses": {},
        "categories": [{}],
        "images": [{}],
        "annotations": [{}]
    }

    return coco_format


