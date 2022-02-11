import torch
import torch.nn.functional as F
from torch import nn

from models.modules import DynamicGRU


def get_padded_mask_and_weight(*args):
    mask, conv = args
    masked_weight = torch.round(F.conv1d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                         stride=conv.stride, padding=conv.padding, dilation=conv.dilation))

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]  # conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight


class QueryEncoder(nn.Module):
    def __init__(self, config):
        super(QueryEncoder, self).__init__()
        self.txt_gru = DynamicGRU(config['input_size'],
                                  config['hidden_size'] // 2 if config['gru']['bidirectional'] else config[
                                      'hidden_size'],
                                  num_layers=config['gru']['num_layers'],
                                  bidirectional=config['gru']['bidirectional'], batch_first=True)

    def reset_parameters(self):
        self.txt_gru.reset_parameters()

    def forward(self, textual_input, textual_len, textual_mask, fast_weights=None, **kwargs):
        # if fast_weights is not None:
        #     fast_weights = get_sub_layer('txt_gru.', fast_weights)
        txt_h = self.txt_gru(textual_input, textual_len, fast_weights=fast_weights)
        return txt_h
