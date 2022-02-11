import os

import numpy as np
import torch

from datasets.base_dataset import BaseDataset, build_collate_data
from utils import iou


class CharadesSTA(BaseDataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        super().__init__(data_path, vocab, args, **kwargs)
        self.num_clips = args['max_num_frames'] // args['target_stride']
        start = np.reshape(np.repeat(np.arange(0, self.num_clips)[:, np.newaxis], axis=1,
                                     repeats=self.num_clips), [-1])
        end = np.reshape(np.repeat(np.arange(1, self.num_clips + 1)[np.newaxis, :], axis=0,
                                   repeats=self.num_clips), [-1])
        self.props = np.stack([start, end], -1)

        # predefined proposals
        idx = self.props[:, 0] < self.props[:, 1]
        self.props = self.props[idx]

        if 'is_training' in kwargs:
            idx = (self.props[:, 1] - self.props[:, 0]) <= self.num_clips // 2
            self.props = self.props[idx]
            idx = (self.props[:, 1] - self.props[:, 0]) % 2 == 1
            self.props = self.props[idx]
            print('train candidate proposals', self.props.shape)
        else:
            print('test candidate proposals', self.props.shape)

        # predefined proposals graph
        iou_predefined = True
        print('iou_predefined graph', iou_predefined)
        if iou_predefined:
            # self.props_graph = iou(self.props.tolist(), self.props.tolist())
            # min_iou, max_iou = 0.6, 1.0
            # self.props_graph = (self.props_graph - min_iou) / (max_iou - min_iou)
            # idx = self.props_graph < 0
            # self.props_graph[:, :] = 1.0
            # self.props_graph = self.props_graph.astype(np.int32)
            # self.props_graph[idx] = 0

            props_iou = iou(self.props.tolist(), self.props.tolist())
            self.props_graph = np.zeros_like(props_iou).astype(np.int32)
            sort_idx = np.argsort(-props_iou, -1)[:, :11]
            for i in range(self.props.shape[0]):
                self.props_graph[i, sort_idx[i]] = 1
                low_idx = props_iou[i] < 0.6
                self.props_graph[i, low_idx] = 0
        else:
            num_props = self.props.shape[0]
            self.props_graph = np.zeros([num_props, num_props]).astype(np.int32)
            for i in range(num_props):
                for j in range(num_props):
                    if abs(self.props[i, 0] - self.props[j, 0]) <= 2 and \
                            abs(self.props[i, 1] - self.props[j, 1]) <= 2:
                        self.props_graph[i, j] = 1
            # print(self.props_graph.sum(axis=-1))
            # exit(0)

        self.props_torch = torch.from_numpy(self.props)
        self.props_graph_torch = torch.from_numpy(self.props_graph)

        self.collate_fn = build_collate_data(args['max_num_frames'], args['max_num_words'],
                                             args['frame_dim'], args['word_dim'],
                                             self.props_torch, self.props_graph_torch)

    def _load_frame_features(self, vid):
        return np.asarray(
            np.load(os.path.join(self.args['feature_path'], '%s.npy' % vid))).astype(np.float32)

    def collate_data(self, samples):
        return self.collate_fn(samples)
