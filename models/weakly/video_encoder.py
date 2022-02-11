import torch
import torch.nn.functional as F
from torch import nn


class FrameAvgPool(nn.Module):
    def __init__(self, config):
        super(FrameAvgPool, self).__init__()
        self.vis_conv = nn.Conv1d(config['input_size'], config['hidden_size'],
                                  kernel_size=1, stride=1, padding=0)
        self.avg_pool = nn.AvgPool1d(config['kernel_size'], config['stride'])
        # self.bn = nn.BatchNorm1d(config['hidden_size'])

    def forward(self, visual_input, fast_weights=None, **kwargs):
        # visual_input = F.normalize(visual_input, dim=1)
        vis_h = self.vis_conv(visual_input.transpose(1, 2))
        vis_h = self.avg_pool(vis_h)
        # vis_h = self.bn(vis_h)
        vis_h = vis_h.transpose(1, 2)
        return vis_h

    def reset_parameters(self):
        self.vis_conv.reset_parameters()


class FrameMaxPool(nn.Module):
    def __init__(self, config):
        super(FrameMaxPool, self).__init__()
        self.vis_conv = nn.Conv1d(config['input_size'], config['hidden_size'], 1, 1)
        self.max_pool = nn.MaxPool1d(config['kernel_size'], config['stride'])

    def forward(self, visual_input):
        vis_h = self.vis_conv(visual_input.transpose(1, 2))
        vis_h = self.max_pool(vis_h)
        vis_h = vis_h.transpose(1, 2)
        # vis_h = F.normalize(vis_h, dim=-1)
        return vis_h
