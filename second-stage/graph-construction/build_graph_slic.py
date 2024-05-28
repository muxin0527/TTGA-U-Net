import os
import cv2
import json
import numpy as np
from copy import deepcopy
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from collections import Counter
import argparse

from skimage.segmentation import slic, mark_boundaries
from skimage import io, measure

import torch
import torch.nn.functional as F
import pandas as pd

class BuildGraph(object):
    def __init__(self, img_path, seg_path, gt_path, vessel_prob_path, prob_path, save_path, thickness=1):
        self.img_name = os.path.basename(img_path)
        self.image_name = os.path.basename(seg_path)[:-4]
        self.img = io.imread(img_path)
        self.seg = io.imread(seg_path)
        self.seg_gt = io.imread(gt_path)
        self.seg_vessel = io.imread(vessel_prob_path)
        self.seg_prob = np.load(prob_path)
        print("seg_prob_shape:", self.seg_prob.shape)
        self.graph_json_path = os.path.join(save_path, 'json')
        self.graph_img_path = os.path.join(save_path, 'images')
        os.makedirs(self.graph_json_path, exist_ok=True)
        os.makedirs(self.graph_img_path, exist_ok=True)
        self.graph = dict()
        self.graph['image_id'] = self.image_name
        self.graph['metric'] = {}
        self.graph['info'] = {'node_num': 0, 'link_num': 0}
        self.graph['nodes'] = list()
        self.graph['links'] = list()
        self.ratio = 0.2
        self.thickness = thickness

    def slic_sampling(self, img):
        slic_seg = slic(img, n_segments=1200, compactness=0.1, max_num_iter=50, start_label=1, channel_axis=None)
        return slic_seg

    def draw_boundaries(self, img, sampling_result):
        color = (173 / 255, 216 / 255, 230 / 255)
        boundaries_img = mark_boundaries(img, sampling_result, color=color, mode='inner')
        boundaries_img = (boundaries_img * 255).astype(np.uint8)
        return boundaries_img
        
    def graph_generator(self):
        segments = self.slic_sampling(self.seg)
        # 超像素分割结果可视化
        graph_img = self.draw_boundaries(self.seg, segments)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "slic.png"), graph_img)
        # print(graph_img.shape)
        graph_gt_img = self.draw_boundaries(self.seg_gt, segments)
        graph_vessel_img = self.draw_boundaries(self.seg_vessel, segments)
        graph_degree_img = self.draw_boundaries(self.seg, segments)

        # 统计区域信息
        region_label = measure.label(segments, connectivity=2) + 1
        region_prop = measure.regionprops(region_label)
        Sample_Nodes = []

        # 采样点信息提取，提取采样点的初始标签，真实标签，像素坐标等
        all_acc_num = 0
        for item, rp in enumerate(region_prop):
            node = dict()
            node['id'] = item
            node['centroid'] = (int(rp.centroid[0]), int(rp.centroid[1]))

            # 计算区域内血管类别
            init_elem = [self.seg[x, y] for x, y in rp.coords]
            init_counter = Counter(init_elem)
            init_label = max(init_counter, key=init_counter.get)

            # 转化为血管
            if init_label == 0:
                if init_counter[init_label] <= 0.75 * rp.area:
                    init_label = 1
            if init_label > 0:
                init_label = 1
            node['init_label'] = int(init_label)

            # 在血管图上，判断区域是否为血管
            area_value = [self.seg_vessel[x, y] for x, y in rp.coords]
            zero_count = area_value.count(0)
            vessel_count = rp.area - zero_count

            # 节点内血管像素占比大于定比例，节点的vessel特征设置为1，表示该节点有血管
            if vessel_count/rp.area >= self.ratio:  # 影响采样点密度
                node['vessel'] = 1
                node['vessel_location'] = [(int(x), int(y), int(self.seg_vessel[x, y])) for x, y in rp.coords if self.seg_vessel[x, y] != 0]
            else:
                node['vessel'] = 0

            # 计算真值
            gt_elem = [self.seg_gt[x, y] for x, y in rp.coords]
            gt_counter = Counter(gt_elem)
            gt_label = max(gt_counter, key=gt_counter.get)

            # 区域内背景像素占比小于0.75的判断为血管
            if gt_label == 0:
                if gt_counter[gt_label] <= 0.75 * rp.area:
                    gt_label = 1

            if gt_label > 0:
                gt_label = 1
            if node['init_label'] == 1:
                gt_label = 1
            node['ground_truth'] = int(gt_label)

            if node['init_label'] == node['ground_truth']:
                all_acc_num += 1
            node['area'] = int(rp.area)
            node['coords'] = [(xy[0], xy[1]) for xy in rp.coords.tolist()]
            node_elem = rp.coords
            y_list = node_elem[:, 0]
            x_list = node_elem[:, 1]

            #四周膨胀一个像素
            contour_thickness = self.thickness
            ymxm = node_elem - contour_thickness  # y-1 x-1
            ymxo = np.c_[y_list - contour_thickness, x_list]  # y-1 x
            ymxp = np.c_[y_list - contour_thickness, x_list + contour_thickness]  # y-1 x+1
            yoxm = np.c_[y_list, x_list - contour_thickness]  # y  x-1
            yoxp = np.c_[y_list, x_list + contour_thickness]  # y  x+1
            ypxm = np.c_[y_list + contour_thickness, x_list - contour_thickness]  # y+1 x-1
            ypxo = np.c_[y_list + contour_thickness, x_list]  # y+1  x
            ypxp = node_elem + contour_thickness  # y+1 x+1

            bigcoords = ymxm.tolist() + ymxo.tolist() + ymxp.tolist() + yoxm.tolist() \
                        + yoxp.tolist() + ypxm.tolist() + ypxo.tolist() + ypxp.tolist()
            bigcoords = np.array(bigcoords)

            # 使用np.where 处理小于0 大于256的数
            bigcoords[np.where(bigcoords < 0)] = 0
            bigcoords[np.where(bigcoords >= 256)] = 255

            bigcoords = set([(yx[0], yx[1]) for yx in bigcoords.tolist()])
            node_elem = set([(yx[0], yx[1]) for yx in node_elem.tolist()])
            contour = bigcoords - node_elem
            node['contour'] = list(contour)
            
            Sample_Nodes.append(node)

        chain_star = [-1 for i in range(len(Sample_Nodes))]
        chain_idx = 0

        # 确定采样点以及边
        for idx, src in enumerate(Sample_Nodes):
            for dst in Sample_Nodes:
                if src['vessel'] != 0 and dst['vessel'] != 0:
                    if len(set(src['contour']) & set(dst['coords'])) != 0:  # 两个节点有接触的话
                        if chain_star[src['id']] == -1:  # 说明graphs['nodes']里没有该节点，然后放入
                            self.graph['nodes'].append(deepcopy(src))  # 用深拷贝，因为要修改值，而append是拷贝的引用
                            self.graph['nodes'][-1]['id'] = chain_idx
                            chain_star[src['id']] = chain_idx
                            src_id = chain_idx
                            chain_idx += 1
                        else:
                            src_id = chain_star[src['id']]
                        if chain_star[dst['id']] == -1:
                            self.graph['nodes'].append(deepcopy(dst))
                            self.graph['nodes'][-1]['id'] = chain_idx
                            chain_star[dst['id']] = chain_idx
                            dst_id = chain_idx
                            chain_idx += 1
                        else:
                            dst_id = chain_star[dst['id']]

                        self.graph['links'].append({'source': src_id, 'target': dst_id})

                        # 邻接点
                        if not self.graph['nodes'][src_id].__contains__('neibor'):
                            self.graph['nodes'][src_id]['neibor'] = [src_id]  #  邻居中包含自身
                        self.graph['nodes'][src_id]['neibor'].append(dst_id)
                        
                        if not self.graph['nodes'][dst_id].__contains__('neibor'):
                            self.graph['nodes'][dst_id]['neibor'] = [dst_id]  #  邻居中包含自身
                        self.graph['nodes'][dst_id]['neibor'].append(src_id)

                        vi = self.graph['nodes'][src_id]['centroid']
                        vj = self.graph['nodes'][dst_id]['centroid']

                        # 边的联系 同为血管为紫色 一个为血管另一个不为血管为黄色， 都不是血管为绿色
                        if src['init_label'] != 0 and dst['init_label'] != 0:
                            color = (255, 127, 255)
                        elif src['init_label'] != 0 or dst['init_label'] != 0:
                            color = (255, 255, 0)
                        else:
                            color = (0, 255, 0)
                        cv2.line(graph_img, (vi[1], vi[0]), (vj[1], vj[0]), color=color, thickness=1)

                        # 把轴向的位置也加上
                        dst_x, dst_y = dst['centroid']
                        src_x, src_y = src['centroid']
                        sys_cen = [int(2*src_x-dst_x), int(2*src_y-dst_y)]
                        for kkk in range(2):
                            if sys_cen[kkk] < 0:
                                sys_cen[kkk] = 0
                            if sys_cen[kkk] >= 256:
                                sys_cen[kkk] = 255
                        src['contour'].append((sys_cen[0], sys_cen[1]))

        # 节点个数，边个数
        self.graph['info']['node_num'] = len(self.graph['nodes'])
        self.graph['info']['link_num'] = len(self.graph['links'])

        # 计算特征 ---> 平均值 角度
        for v in self.graph['nodes']:
            # v  vertex 节点
            # 计算角度
            v_y, v_x = v['centroid']
            color = (0, 255, 0)
            neibor = sorted(v['neibor'], key=lambda neibor_id: self.graph['nodes'][neibor_id]['centroid'])
            if v['init_label'] == 1:
                color = (255, 0, 255)
                new_neibor = []
                for nei in neibor:
                    if self.graph['nodes'][nei]['init_label'] == 1:
                        new_neibor.append(nei)
                neibor = new_neibor
            else:  # 潜在血管
                vessel_neibor = [v['id']]
                for nei in neibor:
                    if self.graph['nodes'][nei]['init_label'] == 1:
                        vessel_neibor.append(nei)
                if len(vessel_neibor) > 1:
                    neibor = vessel_neibor

            dy = self.graph['nodes'][neibor[-1]]['centroid'][0] - self.graph['nodes'][neibor[0]]['centroid'][0]
            dx = self.graph['nodes'][neibor[-1]]['centroid'][1] - self.graph['nodes'][neibor[0]]['centroid'][1]
            degree = round(np.arctan2(dy, dx), 4)
            v['degree'] = degree
            cv2.line(graph_degree_img, (v_x, v_y), (v_x + int(7 * np.cos(degree)), v_y + int(7 * np.sin(degree))),
                     color, thickness=1)


            v_coords = v['coords']
            img_mean = np.mean([self.img[coord[0], coord[1]] for coord in v_coords], axis=0).tolist()
            img_std = np.std([self.img[coord[0], coord[1]] for coord in v_coords
                               if self.seg_vessel[coord[0], coord[1]] != 0], axis=0).tolist()
            v['img_mean'] = img_mean
            v['img_std'] = img_std

            # 特征图特征的平均值
            cur_prob = [self.seg_prob[coord[0], coord[1]] for coord in v_coords]
            print(len(cur_prob))
            prob_mean = np.mean(cur_prob, axis=0).tolist()
            print("prob_mean_shape:", len(prob_mean))
            v['prob_mean'] = prob_mean
            v['features'] = v['prob_mean'] + [abs(np.cos(v['degree'])), int(v['init_label'])]

        # 可视化 节点 节点角度 真值
        for v in self.graph['nodes']:
            init_label = v['init_label']
            location = v['centroid']
            if init_label == 0:
                radius = 1
                color = (255, 255, 255)
            elif init_label == 1:
                radius = 2
                color = (255, 0, 0)

            cv2.circle(graph_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)
            cv2.circle(graph_degree_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)
            if init_label == 0:
                radius = 2
                color = (0, 255, 0)
            cv2.circle(graph_vessel_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)

            if v['ground_truth'] == 0:
                radius = 1
                color = (255, 255, 255)
            elif v['ground_truth'] == 1:
                radius = 2
                color = (255, 0, 0)

            cv2.circle(graph_gt_img, (int(location[1]), int(location[0])), radius=radius, color=color,
                       thickness=-1)

        # 计算精度
        node_num = len(self.graph['nodes'])
        TP, TN, FP, FN = 0, 0, 0, 0
        for v in self.graph['nodes']:
            if v['init_label'] == 0:
                if v['ground_truth'] == 0:
                    TN += 1
                else:
                    FN += 1
            else:
                if v['ground_truth'] == 0:
                    FP += 1
                else:
                    TP += 1

        self.graph['metric']['TP'] = TP  # {'TP': TP,'TN': TN, 'FP': FP, 'FN':FN}
        self.graph['metric']['FP'] = FP
        self.graph['metric']['TN'] = TN
        self.graph['metric']['FN'] = FN
        if (TP + FP) == 0:
            print('TP+FP=0')
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/images', self.img_name))
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/labels', self.img_name))
            return 1
        else:
            self.graph['metric']['Precision'] = TP / (TP + FP)

        if (TP + FN) == 0:
            print('TP+FN=0')
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/images', self.img_name))
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/labels', self.img_name))
            return 1
        else:
            self.graph['metric']['Recall'] = TP / (TP + FN)

        self.graph['metric']['F1'] = 2 * (self.graph['metric']['Recall'] * self.graph['metric']['Precision']) / (
                self.graph['metric']['Recall'] + self.graph['metric']['Precision'])
        if (TP + FP + TN + FN) == 0:
            print('TP+FP+TN+FN=0')
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/images', self.img_name))
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/labels', self.img_name))
            return 1
        else:
            self.graph['metric']['Acc'] = (TP + TN) / (TP + FP + TN + FN)

        if (FP + TN) == 0:
            print('FP+TN=0')
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/images', self.img_name))
            os.remove(os.path.join('/home/data/Program/LiverVessel_Seg/dataset/Origin/labels', self.img_name))
            return 1
        else:
            self.graph['metric']['Average_Acc'] = (TP / (TP + FN) + TN / (FP + TN)) / 2

        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph.png"), graph_img)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph_degree.png"), graph_degree_img)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph_gt.png"), graph_gt_img)
        io.imsave(os.path.join(self.graph_img_path, self.image_name + "graph_vessel.png"), graph_vessel_img)
        # print(os.path.join(self.graph_img_path, self.image_name + "node.png"))
        return chain_star
    

    def save_json(self):
        graph_json = json.dumps(self.graph)
        with open(os.path.join(self.graph_json_path, self.image_name+'.json'), 'w') as json_file:
            json_file.write(graph_json)

    def display(self, out):
        io.imshow(out)
        io.show()


class GraphDGL(object):
    def __init__(self, mode, graph_list, dgl_save_path):
        """
        将graph信息合并,转换成dgl所需要的格式
        :param mode: 'train', 'valid', 'test'
        :param graph_list: 图的节点和边信息
        :param dgl_save_path: 保存路径 'dgl_graph_info'
        """
        self.mode = mode
        self.graph_list = graph_list
        self.mode_graph = dict()
        self.mode_graph_id = list()
        self.mode_labels = list()
        self.mode_feats = list()
        self.mode_graph_path = os.path.join(dgl_save_path, mode+'_graph.json')
        self.mode_graph_id_path = os.path.join(dgl_save_path, mode+'_graph_id.npy')
        self.mode_labels_path = os.path.join(dgl_save_path, mode+'_labels.npy')
        self.mode_feats_path = os.path.join(dgl_save_path, mode + '_features.npy')

    def dataloader(self):
        self.mode_graph['directed'] = True
        self.mode_graph['multigraph'] = False
        self.mode_graph['graph'] = {}
        self.mode_graph['nodes'] = []
        self.mode_graph['links'] = []

        for item, graph in tqdm(enumerate(self.graph_list), desc=self.mode):
            start = len(self.mode_graph['nodes'])
            # mode_graph
            for node in graph['nodes']:
                self.mode_graph['nodes'].append({'id': node['id']+start})
            for link in graph['links']:
                self.mode_graph['links'].append({'source': link['source']+start,
                                                 'target': link['target']+start})
            # mode_graph_id
            self.mode_graph_id[start:start+len(graph['nodes'])] = [item+1] * len(graph['nodes'])
            # mode_labels
            label_list = [node['ground_truth'] for node in graph['nodes']]
            label_array = np.array(label_list, dtype=np.int64)
            label_encoder = F.one_hot(torch.tensor(label_array), 2)
            self.mode_labels[start:start+len(graph['nodes'])] = np.array(label_encoder)
            # model_feats
            features_list = [node['features'] for node in graph['nodes']]
            self.mode_feats[start:start+len(graph['nodes'])] = np.array(features_list)

        mode_graph_json = json.dumps(self.mode_graph)
        with open(self.mode_graph_path, 'w') as f:
            f.write(mode_graph_json)

        np.save(self.mode_graph_id_path, np.array(self.mode_graph_id))
        np.save(self.mode_labels_path, np.array(self.mode_labels))
        np.save(self.mode_feats_path, np.array(self.mode_feats))


def main(args):
    contour = args.contour
    img_path = args.img_path
    seg_path = args.seg_path
    seg_gt_path = args.seg_gt_path
    seg_vessel_path = args.seg_vessel_path
    prob_path = args.prob_path
    save_path = args.graph_save_path
    graph_save_path = save_path+'/graph_info_' + str(contour)
    dgl_save_path = save_path+'/dgl_graph_info_'
    img_path_list = natsorted(glob(img_path))
    print(len(img_path_list))
    seg_path_list = natsorted(glob(seg_path))
    print(len(seg_path_list))
    seg_gt_path_list = natsorted(glob(seg_gt_path))
    print(len(seg_gt_path_list))
    seg_vessel_path_list = natsorted(glob(seg_vessel_path))
    print(len(seg_vessel_path_list))
    prob_path_list = natsorted(glob(prob_path))
    print(len(prob_path_list))
    data_list = zip(img_path_list, seg_path_list, seg_gt_path_list, seg_vessel_path_list,
                    prob_path_list)
    graph_list = []
    graph_acc = dict()
    feats_data = list()

    nan_count = 0
    for img, seg, seg_gt, seg_vessel, prob in tqdm(data_list, desc='Graph Building>>>'):
        print(os.path.basename(img))
        graph = BuildGraph(img, seg, seg_gt, seg_vessel, prob, graph_save_path, thickness=contour)
        value = graph.graph_generator()
        if value == 1:
            nan_count += 1
            continue
        graph.save_json()
        graph_list.append(graph.graph)
        acc_dict = dict()
        acc_dict['TP'] = graph.graph['metric']['TP']
        acc_dict['FP'] = graph.graph['metric']['FP']
        acc_dict['TN'] = graph.graph['metric']['TN']
        acc_dict['FN'] = graph.graph['metric']['FN']
        acc_dict['Precision'] = graph.graph['metric']['Precision']
        acc_dict['Recall'] = graph.graph['metric']['Recall']
        acc_dict['F1'] = graph.graph['metric']['F1']
        acc_dict['Acc'] = graph.graph['metric']['Acc']
        acc_dict['Average_Acc'] = graph.graph['metric']['Average_Acc']
        graph_acc[graph.graph['image_id']] = acc_dict

        for graph_node in graph.graph['nodes']:
            feats_data.append(graph_node['features'] + [graph_node['ground_truth']])

    acc_pd = pd.DataFrame.from_dict(graph_acc)
    acc_pd.to_excel('graph_metrics.xlsx')
    feats_pd = pd.DataFrame(feats_data)
    feats_pd.to_excel('feats_data.xlsx')
    print("nan number:", nan_count)

    print('train_1 data building')
    os.makedirs(dgl_save_path + str(contour), exist_ok=True)
    print("graph_list:", len(graph_list))
    train_graph_list = graph_list[:4580]
    print("train_graph_list:", len(train_graph_list))
    train = GraphDGL('train', train_graph_list, dgl_save_path + str(contour))
    train.dataloader()
    valid_graph_list = graph_list[4580:6544]
    print("valid_graph_list:", len(valid_graph_list))
    valid = GraphDGL('valid', valid_graph_list, dgl_save_path + str(contour))
    valid.dataloader()
    test_graph_list = graph_list[4580:6544]
    print("test_graph_list:", len(test_graph_list))
    test = GraphDGL('test', test_graph_list, dgl_save_path + str(contour))
    test.dataloader()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Graph-construction')
    parser.add_argument("--contour", type=int, default=1,
                        help="Thickness of the contour")
    parser.add_argument("--img_path", type=str, default='../../../dataset/images/*.png')
    parser.add_argument("--seg_path", type=str, default='../../first-stage/UNETR++/Result/*seg.tif',
                        help="the binary segmentation results of first-stage model")
    parser.add_argument("--seg_gt_path", type=str, default='../../../dataset/Test_GT/*.tif',
                        help="segmentation ground truth")
    parser.add_argument("--seg_vessel_path", type=str, default='../../first-stage/UNETR++/Result/*seg_vessel_prob.tif',
                        help="segmentation probability map")
    parser.add_argument("--prob_path", type=str, default='../../first-stage/UNETR++/Result/*Prob.npy',
                        help="the .npy of the prediction of first-stage model")
    parser.add_argument("--graph_save_path", type=str, default='Graph',
                        help="graph save path")
    args = parser.parse_args()
    main(args)








