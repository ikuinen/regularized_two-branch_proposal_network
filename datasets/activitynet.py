import os

import h5py
import numpy as np
import torch

from datasets.base_dataset import BaseDataset, build_collate_data
from utils import iou


class ActivityNetDataset(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)

        self.num_clips = args['max_num_frames'] // args['target_stride']
        start = np.reshape(np.repeat(np.arange(0, self.num_clips)[:, np.newaxis], axis=1,
                                     repeats=self.num_clips), [-1])
        end = np.reshape(np.repeat(np.arange(1, self.num_clips + 1)[np.newaxis, :], axis=0,
                                   repeats=self.num_clips), [-1])
        self.props = np.stack([start, end], -1)

        # predefined proposals
        # idx = self.props[:, 0] < self.props[:, 1]
        # self.props = self.props[idx]
        # keep = np.zeros([self.props.shape[0]]).astype(np.bool)
        #
        # for i, (s, e) in enumerate(self.props):
        #     d = e - s
        #     if 16 < d <= 32:
        #         if s % 2 != 0:
        #             continue
        #     elif d > 32:
        #         if s % 4 != 0:
        #             continue
        #     if d in [8, 15, 24, 32, 48, 56, 64]:
        #         keep[i] = True
        # self.props = self.props[keep]
        # print(self.props)
        # print(self.props)
        # exit(0)

        # predefined proposals

        # idx = self.props[:, 0] % 2 == 0
        # self.props = self.props[idx]

        if 'is_training' in kwargs and False:
            start = np.reshape(np.repeat(np.arange(0, self.num_clips)[:, np.newaxis], axis=1,
                                         repeats=self.num_clips), [-1])
            end = np.reshape(np.repeat(np.arange(1, self.num_clips + 1)[np.newaxis, :], axis=0,
                                       repeats=self.num_clips), [-1])
            self.props = np.stack([start, end], -1)
            idx = self.props[:, 0] < self.props[:, 1]
            self.props = self.props[idx]
            idx = (self.props[:, 1] - self.props[:, 0]) <= self.num_clips // 2
            self.props = self.props[idx]
        else:
            self.props = []
            tmp = [[1], [2], [2]]
            tmp[0].extend([1] * 15)
            tmp[1].extend([1] * 7)
            tmp[2].extend([1] * 7)
            acum_layers = 0
            stride = 1
            for scale_idx, strides in enumerate(tmp):
                for i, stride_i in enumerate(strides):
                    stride = stride * stride_i
                    keep = False
                    if 'is_training' in kwargs:
                        if scale_idx == 0 and i in [7, 15]:
                            keep = True
                        elif scale_idx == 1 and (i in [3, 7]):
                            keep = True
                        elif scale_idx == 2 and (i in [3, 5, 7]):
                            keep = True
                    else:
                        if scale_idx == 0 and i in [7, 15]:
                            keep = True
                        elif scale_idx == 1 and (i in [3, 7]):
                            keep = True
                        elif scale_idx == 2 and (i in [3, 5, 7]):
                            keep = True
                    if not keep:
                        continue
                    ori_s_idxs = list(range(0, self.num_clips - acum_layers - i * stride, stride))
                    ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]

                    self.props.append(np.stack([ori_s_idxs, ori_e_idxs], -1))
                    # print(ori_s_idxs)
                    # print(ori_e_idxs)
                    # print('----')
                acum_layers += stride * (len(strides) + 1)
            self.props = np.concatenate(self.props, 0)
            self.props[:, 1] += 1

        # print(self.props)
        # exit(0)

        # if 'is_training' in kwargs:
        #     idx = self.props[:, 1] - self.props[:, 0] > 8
        #     # print(self.props[:, 1] - self.props[:, 0])
        #     self.props = self.props[idx]
        #     # print('fuck')
        print('candidate proposals', self.props.shape)
        # exit(0)

        # predefined proposals graph
        iou_predefined = True
        print('iou_predefined graph', iou_predefined)
        if iou_predefined:
            # self.props_graph = iou(self.props.tolist(), self.props.tolist())
            #
            # min_iou, max_iou = 0.45, 1.0
            # self.props_graph = (self.props_graph - min_iou) / (max_iou - min_iou)
            # idx = self.props_graph < 0
            # self.props_graph[:, :] = 1.0
            # self.props_graph = self.props_graph.astype(np.int32)
            # self.props_graph[idx] = 0
            props_iou = iou(self.props.tolist(), self.props.tolist())
            self.props_graph = np.zeros_like(props_iou).astype(np.int32)
            sort_idx = np.argsort(-props_iou, -1)
            for i in range(self.props.shape[0]):
                self.props_graph[i, sort_idx[i]] = 1
                low_idx = props_iou[i] < 0.6
                self.props_graph[i, low_idx] = 0
            # print(self.props_graph.sum(axis=-1))
            # exit(0)
        else:
            num_props = self.props.shape[0]
            self.props_graph = np.zeros([num_props, num_props]).astype(np.int32)
            for i in range(num_props):
                for j in range(num_props):
                    if abs(self.props[i, 0] - self.props[j, 0]) <= 4 and \
                            abs(self.props[i, 1] - self.props[j, 1]) <= 4:
                        self.props_graph[i, j] = 1
            print(self.props_graph.sum(axis=-1))
            exit(0)
        # print(self.props_graph[:20, :20])
        # exit(0)

        # x1 = np.exp(self.props_graph)
        # self.props_graph = (x1 / np.sum(x1, keepdims=True, axis=-1)).astype(np.float32)

        self.props_torch = torch.from_numpy(self.props)
        self.props_graph_torch = torch.from_numpy(self.props_graph)

        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'],
                                             self.props_torch, self.props_graph_torch)

    def _load_frame_features(self, vid):
        with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)


class ActivityNet(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'])

    def _load_frame_features(self, vid):
        with h5py.File(os.path.join(self.args['feature_path'], '%s.h5' % vid), 'r') as fr:
            return np.asarray(fr['feature']).astype(np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)
