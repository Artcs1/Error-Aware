import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

import fiftyone.utils.annotations as foua


import json
import numpy as np

gt_file = "instances_val2017.json"
predictions_file = "predictions_yolov10x.json"

coco = COCO(gt_file)
coco_catIds  = coco.getCatIds();
map_coco = {}
for id_cat, coco_cat in enumerate(coco_catIds):
    map_coco[coco_cat] = id_cat  
coco_imgIds  = coco.getImgIds(catIds=[])
all_cocoimgs = coco.loadImgs(coco_imgIds) 
with open(predictions_file, 'r') as f:
    predictions_data = json.load(f)

dataset = foz.load_zoo_dataset("coco-2017", split="validation")
classes = dataset.default_classes
predictions_view = dataset.take(300, seed=51)

with fo.ProgressBar() as pb:
    for sample in pb(predictions_view):
        img_id, height, width = 0, 0, 0
        for cocoimg in all_cocoimgs:
            if cocoimg['file_name'] == sample.filepath.split('/')[-1]:
                img_id = cocoimg['id']
                height = cocoimg['height']
                width  = cocoimg['width']
                break
        detections = []
        for pred in predictions_data:
            if pred['image_id'] == img_id:
                rel_box = [pred['bbox'][0]/width, pred['bbox'][1]/height, pred['bbox'][2]/width, pred['bbox'][3]/height]
                score = pred['score']
                category = map_coco[pred['category_id']]
                annIds  = coco.getAnnIds(imgIds=[img_id], catIds=[pred['category_id']], iscrowd = 0)
                anns    = coco.loadAnns(annIds)
                g = [g['bbox'] for g in anns]
                d = [pred['bbox']]
                matrix =  coco_mask.iou(g, d, [False])
                if matrix == []:
                    iou = 0
                else:
                    iou = np.max(matrix)
                if pred['score'] > 0.3:
                    detections.append(fo.Detection(label=classes[category], bounding_box=rel_box, confidence=iou*100))

        sample["predictions"] = fo.Detections(detections=detections)
        sample.save()

config = fo.AppConfig(
    {
        "font_size": 60,
        "bbox_linewidth": 5,
        "show_all_confidences": True,
        "per_object_label_colors": False,
    }
)

# Render the annotated image
session = fo.launch_app(predictions_view, config = config)
session.wait(-1)

