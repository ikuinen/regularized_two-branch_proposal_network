import torch
import torch.nn.functional as F
from torch import nn

from models.modules import DynamicGRU, CrossGate, TanhAttention
from models.weakly.prop import SparsePropMaxPool


class BaseFusion(nn.Module):
    def __init__(self, config):
        super(BaseFusion, self).__init__()
        hidden_size = config['hidden_size']
        self.vis_conv = nn.Conv2d(hidden_size, hidden_size, 1, 1)
        self.prop = SparsePropMaxPool(config['SparsePropMaxPool'])

    def forward(self, textual_input, text_len, textual_mask,
                visual_input, visual_len, visual_mask, fast_weights=None,
                **kwargs):
        map_h, map_mask = self.prop(visual_input.transpose(1, 2))
        map_h = self.vis_conv(map_h)
        txt_h = torch.stack([textual_input[i][l - 1] for i, l in enumerate(text_len)])
        txt_h = txt_h[:, :, None, None]
        fused_h = F.normalize(txt_h * map_h) * map_mask
        return fused_h, map_mask

    def reset_parameters(self):
        self.vis_conv.reset_parameters()


class BetterFusion(nn.Module):
    def __init__(self, config):
        super(BetterFusion, self).__init__()
        hidden_size = config['hidden_size']
        self.fuse_attn = TanhAttention(hidden_size)
        self.fuse_gate = CrossGate(hidden_size)
        self.fuse_gru = DynamicGRU(hidden_size * 2, hidden_size // 2,
                                   num_layers=1, bidirectional=True, batch_first=True)

    def reset_parameters(self):
        self.fuse_attn.reset_parameters()
        self.fuse_gate.reset_parameters()
        self.fuse_gru.reset_parameters()

    def forward(self, textual_input, text_len, textual_mask, visual_input, visual_len=None, visual_mask=None, fast_weights=None,
                **kwargs):
        if fast_weights is not None:
            # fast_weights1 = get_sub_layer('fuse_attn.', fast_weights)
            # agg_txt_h, _ = self.fuse_attn(visual_input, textual_input, textual_mask, fast_weights=fast_weights1)
            # fast_weights1 = get_sub_layer('fuse_gate.', fast_weights)
            # visual_h, agg_txt_h = self.fuse_gate(visual_input, agg_txt_h, fast_weights=fast_weights1)
            # fast_weights1 = get_sub_layer('fuse_gru.', fast_weights)
            # fused_h = self.fuse_gru(torch.cat([visual_h, agg_txt_h], -1), None, fast_weights=fast_weights1)
            fused_h = None
            return None, None
        else:
            agg_txt_h, _ = self.fuse_attn(visual_input, textual_input, textual_mask)
            visual_h, agg_txt_h = self.fuse_gate(visual_input, agg_txt_h)
            x = torch.cat([visual_h, agg_txt_h], -1)
            fused_h = self.fuse_gru(x, None)
            return fused_h
