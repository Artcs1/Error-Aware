from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from munkres import Munkres
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

import argparse

from utils import *
from tqdm import tqdm


dataset_path = '/home/jeffri/Desktop/'
def qualitative_images(ann, coco, imgIds,  num_classes, name_categories, dataset, show_images = True):

    if dataset == 'coco':
        fake_to_real = [0, 3, 62, 47, 1, 10] # coco 90 classes to 80 classes
    else:
        fake_to_real = [0, 1, 2, 3, 4, 5] # classes of the original dataset

    m = Munkres()
    enumerator = 0
    for img in tqdm(imgIds):
        for cat in range(1,num_classes+1,1):

            annIds            = ann.getAnnIds(imgIds=[img], catIds=[cat], iscrowd = 0)
            corrected_anns    = ann.loadAnns(annIds)

            cocoIds           = coco.getAnnIds(imgIds=[img], catIds=[cat], iscrowd = 0)
            original_anns     = coco.loadAnns(cocoIds)

            g = [g['bbox'] for g in original_anns]
            d = [d['bbox'] for d in corrected_anns]

            file_name = ann.loadImgs(img)[0]['file_name']

            if os.path.exists(dataset_path):
                if dataset == 'coco':
                    if args.split == 'test': 
                        I = cv2.imread(dataset_path+'/coco/images/val2017/'+ file_name)
                    else:
                        I = cv2.imread(dataset_path+'/coco/images/train2017/'+ file_name)
                if dataset == 'google':
                    I = cv2.imread(dataset_path+'/google/images_v4/train/'+ file_name)
                if dataset == 'dota':
                    I = cv2.imread(dataset_path+'/DOTA/train/images/'+ file_name)
            else:
                I = np.zeros((int(ann.loadImgs(img)[0]['height']),int(ann.loadImgs(img)[0]['width']),3))

 
            for img_id in range(len(g)):

                image_coco_id = 0
                for coco_val in all_cocoimgs_gt:
                    if coco_val['file_name'] == file_name:
                        image_coco_id = coco_val['id']
                        break

                preds = [pred['bbox'] for pred in predictions_data if pred['image_id'] == image_coco_id and pred['category_id'] == fake_to_real[cat] and pred['score']>0.3]
                if preds == []:
                    continue

                D = [d[img_id]]
                G = [g[img_id]]

                D_ious = coco_mask.iou(D,preds, [0] * len(preds))
                D_amax = np.argmax(np.array(D_ious))
                G_ious = coco_mask.iou(G,preds, [0] * len(preds))
                G_amax = np.argmax(np.array(G_ious))

                if D_amax == G_amax and D_ious[0][D_amax] != 0.0:
                    cv2.rectangle(I, (int(preds[G_amax][0]), int(preds[G_amax][1])) , (int(preds[G_amax][0] + preds[G_amax][2]), int(preds[G_amax][1] + preds[G_amax][3])) , (255,0,0), 2)
                    cv2.rectangle(I, (int(d[img_id][0]), int(d[img_id][1])) , (int(d[img_id][0] + d[img_id][2]), int(d[img_id][1] + d[img_id][3])) , (0,255,0), 2)
                    cv2.rectangle(I, (int(g[img_id][0]), int(g[img_id][1])) , (int(g[img_id][0] + g[img_id][2]), int(g[img_id][1] + g[img_id][3])) , (0,0,255), 2)

                    cv2.putText(I, 'IoU = '+str(round(D_ious[0][D_amax]*100,2)), (int(d[img_id][0]), int(d[img_id][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(I, 'IoU = '+str(round(G_ious[0][G_amax]*100,2)), (int(g[img_id][0] + g[img_id][2]+5), int(g[img_id][1] + g[img_id][3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if show_images==False:
                cv2.imwrite(args.show_predict[:-5]+'/'+ file_name+'_'+str(enumerator)+'.png', I)
                enumerator+=1
            else:
                cv2.imshow('image.png', I)
                cv2.waitKey()
                cv2.destroyAllWindows()


def main():

    parser = argparse.ArgumentParser(description='estadisticas')
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--split', default='test')
    parser.add_argument('--show_images', default=False)
    parser.add_argument('--show_predict', default=None)
    global args
    args = parser.parse_args()

    corrected_file = './output/'+args.dataset+'_original_'+args.split+'.json'
    original_file =  './output/'+args.dataset+'_original_'+args.split+'.json'

    if args.dataset == 'coco':
       name_categories = ['', 'car', 'chair', 'cup', 'person','traffic light']
    elif args.dataset == 'google':
        name_categories = ['', 'building', 'car', 'dog', 'flower','person']
    elif args.dataset == 'dota':
        name_categories = ['', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']



    ann = COCO(corrected_file)
    coco = COCO(original_file)    

    cats = ann.loadCats(ann.getCatIds())
    classes = cats
    num_classes = len(cats)

    coco_cats = coco.loadCats(coco.getCatIds())

    catIds   = ann.getCatIds();
    imgIds   = ann.getImgIds(catIds=[]);
    all_imgs = ann.loadImgs(imgIds)

    coco_catIds  = coco.getCatIds();
    coco_imgIds  = coco.getImgIds(catIds=[])
    all_cocoimgs = coco.loadImgs(coco_imgIds) 


    path = args.show_predict[:-5]
    if not os.path.exists(path):
        os.mkdir(args.show_predict[:-5])

    gt_file = "instances_val2017.json"
    predictions_file = args.show_predict

    coco_gt = COCO(gt_file)
    coco_catIds_gt  = coco_gt.getCatIds();

    coco_imgIds_gt  = coco_gt.getImgIds(catIds=[])
    global all_cocoimgs_gt
    all_cocoimgs_gt = coco_gt.loadImgs(coco_imgIds_gt) 

    global predictions_data
    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)

    qualitative_images(ann, coco, imgIds, num_classes, name_categories, args.dataset, args.show_images)

if __name__ == '__main__':
    main()   

