import numpy as np
import torch
from gensim.utils import tokenize
from torch.utils.data import Dataset

from utils import load_json


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1e-10) / (union[1] - union[0] + 1e-10)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.target_stride = args['target_stride']

    def load_data(self, data):
        self.data = data

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)

        words = [w.lower() for w in tokenize(sentence)]
        words = [w for w in words if w in self.vocab]
        frames_feat = self._load_frame_features(vid)
        words_feat = [self.vocab[w].astype(np.float32) for w in words]

        num_clips = self.max_num_frames // self.target_stride
        # s_times = torch.arange(0, num_clips).float() * duration / num_clips
        # e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips

        # s_idx = int(timestamps[0] / duration * (num_clips - 1))
        # e_idx = int(timestamps[1] / duration * num_clips)
        # frame_gt = np.zeros([num_clips])
        # frame_gt[s_idx:e_idx] = 1

        props = self.props_torch.float() * duration / num_clips
        gts = torch.tensor([timestamps[0], timestamps[1]]).unsqueeze(0).expand(props.size(0), -1).float()
        map_gt = calculate_IoU_batch((props[:, 0], props[:, 1]), (gts[:, 0], gts[:, 1]))

        reg_gt = self.props_torch[torch.argmax(map_gt)]
        frame_gt = torch.zeros(num_clips).float()
        frame_gt[reg_gt[0]:reg_gt[1] - 1] = 1.0

        return {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'map_gt': map_gt,
            'reg_gt': reg_gt,
            'frame_gt': frame_gt,
            'raw': [vid, duration, timestamps, sentence]
        }


class BaseDataset2(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.target_stride = args['target_stride']

    def load_data(self, data):
        self.data = data

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)

        words = [w.lower() for w in tokenize(sentence)]
        words = [w for w in words if w in self.vocab]
        frames_feat = self._load_frame_features(vid)
        words_feat = [self.vocab[w].astype(np.float32) for w in words]

        # num_clips = self.max_num_frames // self.target_stride
        # s_times = torch.arange(0, num_clips).float() * duration / num_clips
        # e_times = torch.arange(1, num_clips + 1).float() * duration / num_clips

        props = self.props_torch.float()

        gts = torch.tensor([timestamps[0], timestamps[1]]).unsqueeze(0).expand(props.size(0), -1).float()
        map_gt = calculate_IoU_batch((props[:, 0], props[:, 1]), (gts[:, 0], gts[:, 1]))

        return {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'map_gt': map_gt,
            'raw': [vid, duration, timestamps, sentence]
        }


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim, props=None, props_graph=None):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []
        words_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_feat']), max_num_words))

        # frames_feat = np.zeros([bsz, max(frames_len), frame_dim]).astype(np.float32)
        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max(words_len), word_dim]).astype(np.float32)
        # map_gt = np.zeros([bsz, num_clips, num_clips]).astype(np.float32)
        map_gt = []
        reg_gt = []
        frame_gt = []

        rep = []
        for i, sample in enumerate(samples):
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]

            rep.append(np.mean(sample['words_feat'], axis=0))

            if len(sample['frames_feat']) < frames_feat.shape[1] and False:
                frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            else:
                # keep_idx = np.linspace(start=0, stop=len(sample['frames_feat']) - 1,
                #                        num=frames_feat.shape[1]).astype(np.int32)
                # print(sample['raw'][0], keep_idx)
                keep_idx = np.arange(0, frames_feat.shape[1] + 1) / frames_feat.shape[1] * len(sample['frames_feat'])
                keep_idx = np.round(keep_idx).astype(np.int64)
                keep_idx[keep_idx >= len(sample['frames_feat'])] = len(sample['frames_feat']) - 1
                frames_len[i] = frames_feat.shape[1]

                map_gt.append(sample['map_gt'])
                reg_gt.append(sample['reg_gt'])
                frame_gt.append(sample['frame_gt'])

                for j in range(frames_feat.shape[1]):
                    s, e = keep_idx[j], keep_idx[j + 1]
                    assert s <= e
                    if s == e:
                        frames_feat[i, j] = sample['frames_feat'][s]
                    else:
                        frames_feat[i, j] = sample['frames_feat'][s:e].mean(axis=0)
        rep = np.asarray(rep)
        # rep = rep / np.sqrt(np.sum(np.power(rep, 2), axis=1, keepdims=True))
        # dist = np.matmul(rep, rep.T)
        dist = rep[:, np.newaxis, :] - rep[np.newaxis, :, :]
        dist = np.sqrt(np.sum(np.power(dist, 2), -1))

        dist = np.exp(-dist)
        dist /= np.sum(dist, axis=-1, keepdims=True)

        neg = []
        for i in range(bsz):
            # print(i, np.argsort(dist[i]))
            # idx = np.random.choice(np.argsort(-dist[i])[:32])
            # print(dist[i])
            idx = np.random.choice(bsz, p=dist[i])
            neg.append(idx)
        neg = np.asarray(neg)
        # print(neg)
        # exit(0)
        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_len': torch.from_numpy(np.asarray(words_len)),
                'map_gt': torch.stack(map_gt, 0),
                'reg_gt': torch.stack(reg_gt, 0),
                'frame_gt': torch.stack(frame_gt, 0),
                'props': props.unsqueeze(0),
                'props_graph': props_graph.unsqueeze(0),
                'neg': neg,
            }
        })

        return batch

    return collate_data
