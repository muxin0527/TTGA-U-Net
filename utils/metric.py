import os
import json
import numpy as np
from glob import glob
from tqdm import tqdm
import SimpleITK as sitk
from natsort import natsorted
import medpy.metric.binary as metric


def mean_std(metric):
    return np.mean(metric), np.std(metric)

def cal_mean_std(metric):
    dice = []
    jaccard = []
    hd95 = []
    assd = []
    precision = []
    recall = []
    specificity = []
    for item in metric.values():
        dice.append(item['dice'])
        jaccard.append(item['jaccard'])
        hd95.append(item['hd95'])
        assd.append(item['assd'])
        precision.append(item['precision'])
        recall.append(item['recall'])
        specificity.append(item['specificity'])
    dice_mean, dice_std = mean_std(dice)
    jaccard_mean, jaccard_std = mean_std(jaccard)
    hd95_mean, hd95_std = mean_std(hd95)
    assd_mean, assd_std = mean_std(assd)
    precision_mean, precision_std = mean_std(precision)
    recall_mean, recall_std = mean_std(recall)
    specificity_mean, specificity_std = mean_std(specificity)
    metric['dice_mean'] = dice_mean
    metric['dice_std'] = dice_std
    metric['jaccard_mean'] = jaccard_mean
    metric['jaccard_std'] = jaccard_std
    metric['hd95_mean'] = hd95_mean
    metric['hd95_std'] = hd95_std
    metric['assd_mean'] = assd_mean
    metric['assd_std'] = assd_std
    metric['precision_mean'] = precision_mean
    metric['precision_std'] = precision_std
    metric['recall_mean'] = recall_mean
    metric['recall_std'] = recall_std
    metric['specificity_mean'] = specificity_mean
    metric['specificity_std'] = specificity_std
    print('dice_mean:', dice_mean, 'dice_std:', dice_std)

    return metric


def seg_val(pred, gt):
    val_metric = dict()
    val_metric['dice'] = metric.dc(pred, gt)
    val_metric['jaccard'] = metric.jc(pred, gt)
    val_metric['hd95'] = metric.hd95(pred, gt)
    val_metric['assd'] = metric.assd(pred, gt)
    val_metric['precision'] = metric.precision(pred, gt)
    val_metric['recall'] = metric.recall(pred, gt)
    val_metric['specificity'] = metric.specificity(pred, gt)
    # val_metric['sensitivity'] = metric.sensitivity(pred, gt)

    return val_metric


def one_fold_metric(id):
    result_path = './Task92' + id + '_MRLiverVessel/'
    pred_path = './Task92' + id + '_MRLiverVessel/predict/*.nii.gz'
    gt_path = '/home/data/Program/unetr_plus_plus/DATASET/unetr_pp_raw/unetr_pp_raw_data/Task92' + id + '_MRLiverVessel/labelsTs/*.nii.gz'
    n_seg_classes = 2

    gt_list = natsorted(glob(gt_path))
    pred_list = natsorted(glob(pred_path))

    test_metric = dict()
    for idex, file in tqdm(enumerate(gt_list)):
        gt = sitk.ReadImage(file)
        gt_array = sitk.GetArrayFromImage(gt)

        pred = sitk.ReadImage(pred_list[idex])
        pred_array = sitk.GetArrayFromImage(pred)
        test_metric[file] = seg_val(pred_array, gt_array)

    test_metric = cal_mean_std(test_metric)
    test_metric_json = json.dumps(test_metric)
    with open(os.path.join(result_path, 'test_metric.json'), 'w') as f:
        f.write(test_metric_json)


def five_fold():
    dice, jaccard, hd95, assd, precision, recall, specificity = [], [], [], [], [], [], []
    for i in range(1, 6):
        json_path = 'Task92' + str(i) + '_MRLiverVessel/'
        with open(os.path.join(json_path, 'test_metric.json')) as f:
            data = json.load(f)

        dice.append(data['dice_mean'])
        jaccard.append(data['jaccard_mean'])
        hd95.append(data['hd95_mean'])
        assd.append(data['assd_mean'])
        precision.append(data['precision_mean'])
        recall.append(data['recall_mean'])
        specificity.append(data['specificity_mean'])

    dice_mean, dice_std = mean_std(dice)
    jaccard_mean, jaccard_std = mean_std(jaccard)
    hd95_mean, hd95_std = mean_std(hd95)
    assd_mean, assd_std = mean_std(assd)
    precision_mean, precision_std = mean_std(precision)
    recall_mean, recall_std = mean_std(recall)
    specificity_mean, specificity_std = mean_std(specificity)
    print('dice_mean:', dice_mean, 'dice_std:', dice_std)

    five_fold_metric = dict()
    five_fold_metric['dice_mean'] = dice_mean
    five_fold_metric['dice_std'] = dice_std
    five_fold_metric['jaccard_mean'] = jaccard_mean
    five_fold_metric['jaccard_std'] = jaccard_std
    five_fold_metric['hd95_mean'] = hd95_mean
    five_fold_metric['hd95_std'] = hd95_std
    five_fold_metric['assd_mean'] = assd_mean
    five_fold_metric['assd_std'] = assd_std
    five_fold_metric['precision_mean'] = precision_mean
    five_fold_metric['precision_std'] = precision_std
    five_fold_metric['recall_mean'] = recall_mean
    five_fold_metric['recall_std'] = recall_std
    five_fold_metric['specificity_mean'] = specificity_mean
    five_fold_metric['specificity_std'] = specificity_std

    five_fold_metric_json = json.dumps(five_fold_metric)
    with open('3dMRLiverVessel_metric.json', 'w') as f:
        f.write(five_fold_metric_json)

if __name__ == '__main__':

    # one fold_metric
    for id in range(1, 6):
        one_fold_metric(str(id))


    # five fold metric
    five_fold()
