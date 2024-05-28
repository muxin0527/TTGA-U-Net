import os
import cv2
import numpy as np
import json
from glob import glob
from tqdm import tqdm
import pandas as pd
from natsort import natsorted
from skimage import io
from skimage import io, measure
import argparse
import sys


sys.path.append('/home/data/Program/TTGA U-Net/second-stage/TGA U-Net/')
def main(args):
    contour = args.contour
    graphInfoList = natsorted(glob(args.graph_save_path+'/graph_info_'+str(contour)+'/json/*.json'))
    image_gt_list = natsorted(glob(args.graph_save_path+'/graph_info_'+str(contour)+'/images/*slic.png'))
    seg_list = natsorted(glob(args.cnn_Result_path+'/*seg.tif'))
    seg_prob_list = natsorted(glob(args.cnn_Result_path+'/*seg_mean_prob.tif'))
    save_path = args.gat_Result_path+'/result_'+str(contour)
    os.makedirs(save_path, exist_ok=True)

    graphInfoList = natsorted(graphInfoList)[4266:6094]
    image_gt_list = natsorted(image_gt_list)[4266:6094]
    result_list = natsorted(glob(save_path + '/result*.npy'))

    acc = dict()
    for idx, result in tqdm(enumerate(result_list)):
        filename = graphInfoList[idx]
        with open(filename, 'r') as f:
            graphOriginInfo = json.load(f)
        result_name = os.path.basename(filename)[:-5]

        data = np.load(result, allow_pickle=True)
        img_label = cv2.imread(image_gt_list[idx])
        seg = cv2.imread(seg_list[idx], cv2.IMREAD_GRAYSCALE)
        seg_src = seg.copy()
        seg_prob = cv2.imread((seg_prob_list[idx]))
        metric = dict()
        metric['TP'] = 0
        metric['FP'] = 0
        metric['TN'] = 0
        metric['FN'] = 0
        for item, label in enumerate(data):
            xy = graphOriginInfo['nodes'][item]['centroid']
            init_label = graphOriginInfo['nodes'][item]['init_label']
            location = graphOriginInfo['nodes'][item]['vessel_location']
            gt = graphOriginInfo['nodes'][item]['ground_truth']
            x = xy[0]
            y = xy[1]
            if np.argmax(label) == 0:
                if gt == 0:  # 正确分为背景  TN
                    color = (0, 255, 0)
                    radius = 2
                    metric['TN'] += 1
                else:       # 错误分为背景 假阴 FN
                    color = (255, 0, 0)
                    radius = 2
                    metric['FN'] += 1
            elif np.argmax(label) == 1:
                if gt == 1:             # TP
                    color = (0, 0, 255)
                    radius = 2
                    metric['TP'] += 1
                else:                   # FP
                    color = (255, 0, 255)
                    radius = 2
                    metric['FP'] += 1
                if init_label != 1:
                    for location_xy in location:
                        print(location_xy)
                        if seg_prob[location_xy[1], location_xy[0], 0] > seg_prob[location_xy[1], location_xy[0], 1]:
                            # opencv BGR 所以0代表静脉，2代表动脉
                            c = [0, (0, 0, 255)]
                        else:
                            c = [255, (255, 0, 0)]
                        cv2.circle(img_label, (location_xy[1], location_xy[0]), radius=1, color=(0, 255, 255),
                                   thickness=-1)
                        if location_xy[2] > 60:
                            cv2.circle(seg, (location_xy[1], location_xy[0]), radius=1, color=c[0],
                                       thickness=-1)

            cv2.circle(img_label, (y, x), radius=2, color=color, thickness=-1)

        seg_diff = np.int16(seg) - np.int16(seg_src)
        seg_diff[seg_diff != 0] = 1
        metric['Precision'] = metric['TP'] / (metric['TP'] + metric['FP'] + 1)
        metric['Recall'] = metric['TP'] / (metric['TP'] + metric['FN'])
        metric['f1'] = 2 * metric['Precision'] * metric['Recall'] / (metric['Precision'] + metric['Recall'] + 1)
        metric['accuracy'] = (metric['TP'] + metric['TN']) / (metric['TP'] + metric['TN'] + metric['FP'] + metric['FN'])
        metric['average_acc'] = (metric['TP'] / (metric['TP'] + metric['FN'])
                                 + metric['TN'] / (metric['TN'] + metric['FP']))/2
        acc[result_name] = metric
        cv2.imwrite(save_path +'/' + result_name+'.tif', img_label)
        cv2.imwrite(save_path +'/' + result_name + 'seg_gat.tif', seg)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument('--gat_Result_path', type=str, default='/home/data/Program/TTGA U-Net/second-stage/',
                        help="OCTA Result_path ")
    parser.add_argument('--cnn_Result_path', type=str, default='/home/data/Program/TTGA U-Net/first-stage/UNETR++/Result',
                        help="OCTA Result_path ")
    parser.add_argument("--graph_save_path", type=str, default='/home/data/Program/TTGA U-Net/second-stage/graph-construction/Graph',
                        help="graph save path")
    parser.add_argument("--contour", type=int, default=1, help="thickness")
    args = parser.parse_args()
    main(args)