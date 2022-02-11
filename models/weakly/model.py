import torch
import torch.nn as nn
import torch.nn.functional as F

import models.weakly.fusion as fusion
import models.weakly.query_encoder as query_encoder
import models.weakly.scorer as scorer
import models.weakly.video_encoder as video_encoder
from models.modules import NetVLAD
from models.weakly.prop import SparsePropMaxPool


class Filter(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size = 512
        self.word_vlad = NetVLAD(cluster_size=8, feature_size=hidden_size)
        self.word_fc = nn.Linear(8 * hidden_size, hidden_size)
        self.fc_gate1 = nn.Linear(hidden_size, hidden_size)
        self.fc_gate2 = nn.Linear(hidden_size, 1)

        # self.self_gate1 = nn.Linear(hidden_size, hidden_size)
        # self.self_gate2 = nn.Linear(hidden_size, 1)

    def forward(self, frames_feat, words_feat, words_len, words_mask, **kwargs):
        frames_feat, words_feat = frames_feat.detach(), words_feat.detach()

        word_des = self.word_vlad(words_feat, words_mask, True)
        word_des = self.word_fc(word_des)

        # word_des = []
        # for i in range(words_len.size(0)):
        #     word_des.append(words_feat[i, :words_len[i]].mean(dim=0))
        # word_des = torch.stack(word_des, 0)
        word_des = word_des.unsqueeze(1)
        # word_des = word_des.expand(*frames_feat.size())
        # NET:mIoU 0.6055 | IoU@0.1 0.9683 | IoU@0.3 0.9297 | IoU@0.5 0.6561 | IoU@0.7 0.3870 | IoU@0.9 0.1048 |
        x = frames_feat * word_des
        x = torch.tanh(self.fc_gate1(x))
        x = self.fc_gate2(x)
        x1 = torch.sigmoid(x)

        # x = frames_feat
        # x = torch.tanh(self.self_gate1(x))
        # x = self.self_gate2(x)
        # x2 = torch.sigmoid(x)

        x = x1

        x = x.squeeze(-1)
        x_max, x_min = x.max(dim=-1, keepdim=True)[0], x.min(dim=-1, keepdim=True)[0]
        x = (x - x_min + 1e-10) / (x_max - x_min + 1e-10)
        x = x.unsqueeze(-1)
        # [nb, len, 1]

        return x


class WeaklyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.video_encoder = getattr(video_encoder, config['VideoEncoder']['name'])(config['VideoEncoder'])
        self.query_encoder = getattr(query_encoder, config['QueryEncoder']['name'])(config['QueryEncoder'])
        self.fusion = getattr(fusion, config['Fusion']['name'])(config['Fusion'])
        # self.fusion1 = getattr(fusion, config['Fusion']['name'])(config['Fusion'])
        self.prop = SparsePropMaxPool(config['Fusion']['SparsePropMaxPool'])
        # self.prop = DensePropMaxPool(config['Fusion']['SparsePropMaxPool'])
        self.scorer = getattr(scorer, config['Scorer']['name'])(config['Scorer'])
        # self.scorer1 = getattr(scorer, config['Scorer']['name'])(config['Scorer'])
        self.back = nn.Parameter(torch.zeros(1, 1, 512), requires_grad=False)
        # from torch.nn import init
        # init.normal_(self.back, 0, 1.0 / 512)
        self.filter_branch = config['filter_branch']
        if self.filter_branch:
            self.filter = Filter(config['Filter'])

    def forward(self, frames_feat, frames_len, words_feat, words_len, bias, get_negative=False, **kwargs):
        dropout_rate = 0.1
        frames_encoded = self.video_encoder(frames_feat)

        kwargs['props'] = kwargs['props'].squeeze(0)
        kwargs['props_graph'] = kwargs['props_graph'].squeeze(0)

        words_mask = generate_mask(words_feat, words_len)
        words_encoded = self.query_encoder(words_feat, words_len, words_mask)

        if self.filter_branch:
            weight = self.filter(frames_encoded, words_encoded, words_len, words_mask, **kwargs)
            # weight = kwargs['frame_gt'].unsqueeze(-1)
            fused_h = self.fusion(words_encoded, words_len, words_mask,
                                  frames_encoded * weight + self.back * (1 - weight))
        else:
            weight = None
            fused_h = self.fusion(words_encoded, words_len, words_mask, frames_encoded)
        fused_h = F.dropout(fused_h, dropout_rate, self.training)
        props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)
        score = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask, **kwargs)
        res = {
            'score': score,
        }

        if get_negative:
            bsz = frames_feat.size(0)
            if self.filter_branch:
                fused_h = self.fusion(words_encoded, words_len, words_mask,
                                      frames_encoded * (1 - weight) + self.back * weight)
                fused_h = F.dropout(fused_h, dropout_rate, self.training)
                props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)
                score = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask, **kwargs)
                res.update({
                    'intra_neg': {
                        'weight': weight,
                        'neg_score': score,
                    }
                })
            else:
                res.update({
                    'intra_neg': {
                        'weight': None,
                        'neg_score': None,
                    }
                })

            # frames_encoded = frames_encoded[list(reversed(range(bsz)))]
            # words_encoded = words_encoded[kwargs['neg']]
            idx = kwargs['neg']
            idx = list(reversed(range(bsz)))
            # words_encoded, words_mask, words_len = words_encoded[idx], words_mask[idx], words_len[idx]
            frames_encoded = frames_encoded[idx]
            # exit(0)
            # idx = list(reversed(range(bsz)))
            # frames_encoded = frames_encoded[idx]
            if self.filter_branch:
                weight = self.filter(frames_encoded, words_encoded, words_len, words_mask)
                fused_h = self.fusion(words_encoded, words_len, words_mask,
                                      frames_encoded * weight + self.back * (1 - weight))
            else:
                fused_h = self.fusion(words_encoded, words_len, words_mask, frames_encoded)
            fused_h = F.dropout(fused_h, dropout_rate, self.training)
            props_h, map_h, map_mask = self.prop(fused_h.transpose(1, 2), **kwargs)
            score = self.scorer(props_h=props_h, map_h=map_h, map_mask=map_mask, **kwargs)
            res.update({
                'inter_neg': {
                    'neg_score': score,
                }
            })
        return res


def generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class Filter2(nn.Module):
    def __init__(self, config):
        super().__init__()

        hidden_size = 512
        self.word_vlad = NetVLAD(cluster_size=4, feature_size=hidden_size)

        self.ws1 = nn.Linear(hidden_size, hidden_size, bias=True)
        self.ws2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wst = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, frames_feat, words_feat, words_len, words_mask, **kwargs):
        frames_feat, words_feat = frames_feat.detach(), words_feat.detach()

        k = self.word_vlad(words_feat, words_mask, False)

        q = self.ws1(frames_feat)  # [nb, len1, d]
        k = self.ws2(k)  # [nb, len2, d]

        sim = self.wst((q.unsqueeze(2) + k.unsqueeze(1)).tanh()).squeeze(-1)
        sim = F.softmax(sim, -1)

        x = sim.mean(dim=-1)
        x_max, x_min = x.max(dim=-1, keepdim=True)[0], x.min(dim=-1, keepdim=True)[0]
        x = (x - x_min + 1e-10) / (x_max - x_min + 1e-10)
        x = x.unsqueeze(-1)

        return x
