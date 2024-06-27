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


def get_correspondences(all_imgs, all_cocoimgs, imgIds, coco_imgIds):

    map_imgs = np.zeros(imgIds[len(imgIds)-1]+1) 

    for img in all_imgs:
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

    dis_per_class = np.zeros(num_classes + 1)

    for img in imgIds:
        
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
                        I = cv2.imread('/home/jeffri/Desktop/coco/images/val2017/'+ file_name)
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

                                flag = True
                                cv2.rectangle(I, (int(g[orig_ind][0]), int(g[orig_ind][1])) , (int(g[orig_ind][0] + g[orig_ind][2]), int(g[orig_ind][1] + g[orig_ind][3])) , (0,0,255), 2)
                                cv2.rectangle(I, (int(d[corr_ind][0]), int(d[corr_ind][1])) , (int(d[corr_ind][0] + d[corr_ind][2]), int(d[corr_ind][1] + d[corr_ind][3])) , (0,255,0), 2)
                                dis_per_class[cat]+=1
                                discrepancies+=1

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

                if show_images and flag:
                
                    cv2.imshow('image', I)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

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

    print(discrepancies)
    print(dis_per_class)

    return information[0], information[1], information[2], information[3], information[4], information[5], information[6]# ious, tipo, name, larea, small_d, tls_errors, brs_errors


def plot_statistics(ious, name, tipo, larea, small_d, tls_errors, brs_errors):

    data = {'iou':ious, 'class':name, 'tipo':tipo}
    df = pd.DataFrame(data)
    ax = sns.boxplot(y="class", x="iou", hue="tipo",data=data, fliersize = 1)
    
    plt.show()

    data = {'iou':ious, 'class':larea, 'tipo':tipo}
    df = pd.DataFrame(data)
    ax = sns.boxplot(y="class", x="iou", hue="tipo",data=data, fliersize = 1)

    plt.show()

    plt.scatter(small_d, ious)
    plt.xlabel("Smallest Dimension")
    plt.ylabel("Iou")
    plt.show()

    plt.scatter(small_d, [-error for error in brs_errors])
    plt.scatter(small_d, tls_errors)
    plt.xlabel("Smallest Dimension")
    plt.ylabel("Top left Right Error (top), Bottom Right Error (Bottom)")
    plt.show()

    data_stats = {'iou':ious, 'br_errors:': brs_errors, 'tl_errors': tls_errors}
    df = pd.DataFrame(data_stats)
    print(df.describe())

def main():

    parser = argparse.ArgumentParser(description='estadisticas')
    parser.add_argument('--dataset', default='coco')
    parser.add_argument('--split', default='train')
    parser.add_argument('--show_images', default=False)
    parser.add_argument('--show_stats', default=False)
    global args
    args = parser.parse_args()

    if args.dataset == 'coco':
        ann_file = './coco/instances_'+args.split+'.json'    
        coco_file = './coco/instancescoco_'+args.split+'.json'
        name_categories = ['', 'car', 'chair', 'cup', 'person','traffic light']
    elif args.dataset == 'google':
        ann_file = './google/instances_'+args.split+'.json'    
        coco_file = './google/instancesgoogle_'+args.split+'.json'
        name_categories = ['', 'building', 'car', 'dog', 'flower','person']
    elif args.dataset == 'dota':
        ann_file = './DOTA/DOTA1.5_'+args.split+'.json'    
        coco_file = './DOTA/DOTA_'+args.split+'.json'
        name_categories = ['', 'plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court', 'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter', 'container-crane']

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

