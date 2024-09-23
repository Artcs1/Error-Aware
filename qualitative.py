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



def get_correspondences(all_imgs, all_cocoimgs, imgIds, coco_imgIds):

    map_imgs = np.zeros(imgIds[len(imgIds)-1]+1) 

    for img in tqdm(all_imgs):
        
        for coco_i in all_cocoimgs:
            if img['file_name'] == coco_i['file_name']:
                map_imgs[img['id']] = coco_i['id']
    
    return map_imgs


def get_statistics(ann, coco, imgIds, num_classes, map_imgs, name_categories, dataset, show_images = True):


    coco_format_original  = get_coco_json_format()
    coco_format_corrected = get_coco_json_format()

    info = {"description": dataset + " 2017 image 5 category - original", "url": "dummy_url", "version": "0.01", "year": 2024, "contributor": "jeffri.murrugarra","data_created": "2024/06/02"}
    coco_format_original['info'] = info
    info = {"description": dataset + " 2017 image 5 category - corrected", "url": "dummy_url", "version": "0.01", "year": 2024, "contributor": "jeffri.murrugarra","data_created": "2024/06/02"}
    coco_format_corrected['info'] = info

    licenses = { "id": 1, "url": "dummy_url", "name": "jeffri.murrugarra"}
    coco_format_original['licenses']  = licenses
    coco_format_corrected['licenses'] = licenses

    categories = ann.loadCats(ann.getCatIds())
    coco_format_original['categories']  = categories
    coco_format_corrected['categories'] = categories

    m = Munkres()

    information = [[], [], [], [], [], [], []] # ious, tipo, name, larea, small_d, tls_errors, brs_errors

    image_id = 0
    images = []

    annotation_id = 0
    annotations_original  = []
    annotations_corrected = []

    results = []

    discrepancies = 0

    coco_fake_to_coco_real = [0, 3, 62, 47, 1, 10]

    enumerator = 0

    for img in tqdm(imgIds):
        
        list_images = ann.loadImgs(img)
        list_images[0]['id'] = image_id
        images.append(list_images[0])

        for cat in range(1,num_classes+1,1):

            annIds  = ann.getAnnIds(imgIds=[img], catIds=[cat], iscrowd = 0)
            anns    = ann.loadAnns(annIds)

            cocoIds   = coco.getAnnIds(imgIds=[map_imgs[img]], catIds=[cat], iscrowd = 0)
            coco_anns = coco.loadAnns(cocoIds)

            g = [g['bbox'] for g in coco_anns]
            d = [d['bbox'] for d in anns]

            matrix = coco_mask.iou(g, d, [False])

            flag = False

            if len(matrix)!=0:

                if len(g) <= len(d):
                    indexes = m.compute(-matrix)
                else:
                    indexes = m.compute(-matrix.T)

                file_name = ann.loadImgs(img)[0]['file_name']
                if dataset == 'coco':
                    if args.split == 'test': 
                        #I = cv2.imread('/home/jeffri/Desktop/coco/images/val2017/'+ file_name)
                        I = cv2.imread('./images/val2017/'+ file_name)
                    else:
                        I = cv2.imread('/home/jeffri/Desktop/coco/images/train2017/'+ file_name)
                if dataset == 'google':
                    I = cv2.imread('/home/jeffri/Desktop/google/images_v4/train/'+ file_name)
                if dataset == 'dota':
                    I = cv2.imread('/home/datasets/DOTA/train/images/'+ file_name)

                if I is None:
                    I = np.zeros((int(ann.loadImgs(img)[0]['height']),int(ann.loadImgs(img)[0]['width']),3))

                for val in indexes:
                
                    if len(g) <= len(d):
                        orig_ind = val[0]
                        corr_ind = val[1]
                    else:
                        orig_ind = val[1]
                        corr_ind = val[0]

                    if matrix[orig_ind][corr_ind] > 0.1:

                        I_copy = I

                        orig_ann = coco_anns[orig_ind]
                        corr_ann = anns[corr_ind]

                        coco_area = coco_anns[orig_ind]['area']
                        anns_area  = anns[corr_ind]['area']
                        tl_coco = np.array(coco_anns[orig_ind]['bbox'][0:2])
                        tl_anns = np.array(anns[corr_ind]['bbox'][0:2])
                        tl_error = np.sum(abs(tl_coco-tl_anns))

                        br_coco = np.array(coco_anns[orig_ind]['bbox'][2:4]) + tl_coco
                        br_anns = np.array(anns[corr_ind]['bbox'][2:4]) + tl_anns
                        br_error = np.sum(abs(br_coco-br_anns))
                        small_dimension = np.min(coco_anns[orig_ind]['bbox'][2:4])

                        if True:#coco_area*2 > anns_area:#(coco_area*1.15)>  anns_area and (anns_area * 1.15) > coco_area: #br_error > 200 or tl_error >200:

                            # ious, tipo, name, larea, small_d, tls_errors, brs_errors
                            information[0].append(matrix[orig_ind][corr_ind])
                            information[1].append('hbb')
                            information[2].append(name_categories[cat])
                            information[4].append(small_dimension)
                            information[5].append(br_error)
                            information[6].append(tl_error)
                            area = coco_area

                            if area <= 32*32:
                                information[3].append('small')
                            elif area <= 96*96:
                                information[3].append('medium')
                            else:
                                information[3].append('large') 

                            if matrix[orig_ind][corr_ind] < 1.0:


                                image_coco_id = 0
                                for coco_val in all_cocoimgs_gt:

                                    if coco_val['file_name'] == file_name:
                                        image_coco_id = coco_val['id']
                                        break

                                preds = [pred['bbox'] for pred in predictions_data if pred['image_id'] == image_coco_id and pred['category_id'] == coco_fake_to_coco_real[cat] and pred['score']>0.3]

                                if preds == []:
                                    continue

                                D = [d[corr_ind]]
                                G = [g[orig_ind]]

                                D_ious = coco_mask.iou(D,preds, [0] * len(preds))
                                D_amax = np.argmax(np.array(D_ious))
                                G_ious = coco_mask.iou(G,preds, [0] * len(preds))
                                G_amax = np.argmax(np.array(G_ious))

                                if D_amax == G_amax and D_ious[0][D_amax] != 0.0:
                                    cv2.rectangle(I_copy, (int(preds[G_amax][0]), int(preds[G_amax][1])) , (int(preds[G_amax][0] + preds[G_amax][2]), int(preds[G_amax][1] + preds[G_amax][3])) , (255,0,0), 2)

                                    cv2.rectangle(I_copy, (int(d[corr_ind][0]), int(d[corr_ind][1])) , (int(d[corr_ind][0] + d[corr_ind][2]), int(d[corr_ind][1] + d[corr_ind][3])) , (0,255,0), 2)
                                    cv2.rectangle(I_copy, (int(g[orig_ind][0]), int(g[orig_ind][1])) , (int(g[orig_ind][0] + g[orig_ind][2]), int(g[orig_ind][1] + g[orig_ind][3])) , (0,0,255), 2)

                                    cv2.putText(I_copy, 'IoU = '+str(round(D_ious[0][D_amax]*100,2)), (int(d[corr_ind][0]), int(d[corr_ind][1]-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    cv2.putText(I_copy, 'IoU = '+str(round(G_ious[0][G_amax]*100,2)), (int(g[orig_ind][0] + g[orig_ind][2]+5), int(g[orig_ind][1] + g[orig_ind][3])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                            orig_ann['id'] = annotation_id
                            corr_ann['id'] = annotation_id

                            orig_ann['image_id'] = image_id
                            corr_ann['image_id'] = image_id

                            orig_ann['segmentation'] = [[10., 20., 25. ,30., 15., 20. , 26. ,30.]]
                            corr_ann['segmentation'] = [[10., 20., 25. ,30., 15., 20. , 26. ,30.]]

                            annotations_original.append(orig_ann)
                            annotations_corrected.append(corr_ann)

                            results.append({"image_id": image_id, "category_id": cat, "bbox": corr_ann['bbox'], "score": 1.0})
                            annotation_id+=1
                                    
                if show_images:
                    cv2.imwrite(args.show_predict[:-5]+'/'+ file_name+'_'+str(enumerator)+'.png', I_copy)
                    enumerator+=1



        image_id+=1

    coco_format_original['images']  = images
    coco_format_corrected['images'] = images


    coco_format_original['annotations']  = annotations_original
    coco_format_corrected['annotations'] = annotations_corrected
    
    with open("output/"+dataset+"_original_"+args.split+".json", "w") as outfile:
        json.dump(coco_format_original, outfile)
    
    with open("output/"+dataset+"_corrected_"+args.split+".json", "w") as outfile:
        json.dump(coco_format_corrected, outfile)

    with open("output/"+dataset+"_sim_results_"+args.split+".json", "w") as outfile:
        json.dump(results, outfile)

    return information[0], information[1], information[2], information[3], information[4], information[5], information[6]# ious, tipo, name, larea, small_d, tls_errors, brs_errors

def main():

    parser = argparse.ArgumentParser(description='estadisticas')
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--split', default='coco')
    parser.add_argument('--show_images', default=False)
    parser.add_argument('--show_predict', default=None)
    global args
    args = parser.parse_args()

    if args.dataset == 'coco':
        corrected_file = './coco/instances_'+args.split+'.json'    
        original_file = './coco/instancescoco_'+args.split+'.json'
        name_categories = ['', 'car', 'chair', 'cup', 'person','traffic light']
    elif args.dataset == 'google':
        corrected_file = './google/instances_'+args.split+'.json'    
        original_file = './google/instancesgoogle_'+args.split+'.json'
        name_categories = ['', 'building', 'car', 'dog', 'flower','person']
    elif args.dataset == 'dota':
        corrected_file = './DOTA/DOTA1.5_'+args.split+'.json'    
        original_file = './DOTA/DOTA_'+args.split+'.json'
        name_categories = ['', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

    if args.show_predict != None:

        path = args.show_predict[:-5]
        if not os.path.exists(path):
            os.mkdir(args.show_predict[:-5])

        gt_file = "instances_val2017.json"
        predictions_file = args.show_predict

        coco_gt = COCO(gt_file)
        coco_catIds_gt  = coco_gt.getCatIds();

        map_coco = {}

        for id_cat, coco_cat in enumerate(coco_catIds_gt):
            map_coco[coco_cat] = id_cat  


        coco_imgIds_gt  = coco_gt.getImgIds(catIds=[])
        global all_cocoimgs_gt
        all_cocoimgs_gt = coco_gt.loadImgs(coco_imgIds_gt) 


        global predictions_data
        with open(predictions_file, 'r') as f:
            predictions_data = json.load(f)

    ann = COCO(ann_file)
    coco = COCO(coco_file)    

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

    map_imgs = get_correspondences(all_imgs, all_cocoimgs, imgIds, coco_imgIds)
    ious, tipo, name, larea, small_d, tls_errors, brs_errors = get_statistics(ann, coco, imgIds, num_classes, map_imgs, name_categories, args.dataset, args.show_images)

    if args.show_stats:
        plot_statistics(ious, name, tipo, larea, small_d, tls_errors, brs_errors)

if __name__ == '__main__':
    main()   

