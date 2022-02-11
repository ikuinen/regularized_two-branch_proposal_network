import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import dense_to_sparse


def get_padded_mask_and_weight(*args):
    if len(args) == 2:
        mask, conv = args
        masked_weight = torch.round(F.conv2d(mask.clone().float(), torch.ones(1, 1, *conv.kernel_size).cuda(),
                                             stride=conv.stride, padding=conv.padding, dilation=conv.dilation))
    elif len(args) == 5:
        mask, k, s, p, d = args
        masked_weight = torch.round(
            F.conv2d(mask.clone().float(), torch.ones(1, 1, k, k).cuda(), stride=s, padding=p, dilation=d))
    else:
        raise NotImplementedError

    masked_weight[masked_weight > 0] = 1 / masked_weight[masked_weight > 0]  # conv.kernel_size[0] * conv.kernel_size[1]
    padded_mask = masked_weight > 0

    return padded_mask, masked_weight


class GraphConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.gcn = GCNConv(input_size, output_size)

    def _get_buffer(self, x, graph, bsz, len_):
        if not hasattr(self, 'buffer_edge_index'):
            adj_mat = graph.new_zeros(x.size(0), x.size(0))
            for i in range(bsz):
                adj_mat[i * len_:(i + 1) * len_, i * len_:(i + 1) * len_] = graph
            edge_index, edge_attr = dense_to_sparse(adj_mat)
            # print(edge_index.size(1) / bsz)
            setattr(self, 'num_edges_per_graph', edge_index.size(1) // bsz)
            setattr(self, 'buffer_edge_index', edge_index)
        total_edges = getattr(self, 'num_edges_per_graph') * bsz
        return getattr(self, 'buffer_edge_index')[:, :total_edges]

    def forward(self, x, graph):
        bsz, len_, hid_dim = x.size()
        x = x.contiguous().view(-1, hid_dim)
        edge_index = self._get_buffer(x, graph.squeeze(0), bsz, len_)
        res = x
        x = self.gcn(x, edge_index)
        x = F.relu(x) + res
        return x.contiguous().view(bsz, len_, -1)


class GATGraphConv(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        num_heads = 2
        self.gat = GATConv(input_size, output_size // num_heads, heads=num_heads)

    def _get_buffer(self, x, graph, bsz, len_):
        if not hasattr(self, 'buffer_edge_index') or True:
            adj_mat = graph.new_zeros(x.size(0), x.size(0))
            for i in range(bsz):
                adj_mat[i * len_:(i + 1) * len_, i * len_:(i + 1) * len_] = graph
            edge_index, edge_attr = dense_to_sparse(adj_mat)
            assert edge_index.size(1) % bsz == 0
            setattr(self, 'num_edges_per_graph', edge_index.size(1) // bsz)
            setattr(self, 'buffer_edge_index', edge_index)
        total_edges = getattr(self, 'num_edges_per_graph') * bsz
        return getattr(self, 'buffer_edge_index')[:, :total_edges]

    def forward(self, x, graph):
        bsz, len_, hid_dim = x.size()
        x = x.contiguous().view(-1, hid_dim)
        edge_index = self._get_buffer(x, graph.squeeze(0), bsz, len_)
        res = x
        x = self.gat(x, edge_index)
        x = F.relu(x) + res
        return x.contiguous().view(bsz, len_, -1)


class GeneralGraphConv(nn.Module):
    def __init__(self, gcn_class, **kwargs):
        super().__init__()
        self.gcn = gcn_class(**kwargs)

    def _get_buffer(self, x, graph, bsz, len_):
        if not hasattr(self, 'buffer_edge_index') or True:
            adj_mat = graph.new_zeros(x.size(0), x.size(0))
            for i in range(bsz):
                adj_mat[i * len_:(i + 1) * len_, i * len_:(i + 1) * len_] = graph
            edge_index, edge_attr = dense_to_sparse(adj_mat)
            assert edge_index.size(1) % bsz == 0
            setattr(self, 'num_edges_per_graph', edge_index.size(1) // bsz)
            setattr(self, 'buffer_edge_index', edge_index)
        total_edges = getattr(self, 'num_edges_per_graph') * bsz
        return getattr(self, 'buffer_edge_index')[:, :total_edges]

    def forward(self, x, graph):
        bsz, len_, hid_dim = x.size()
        x = x.contiguous().view(-1, hid_dim)
        edge_index = self._get_buffer(x, graph.squeeze(0), bsz, len_)
        x = x.unsqueeze(1)
        res = x
        x = self.gcn(x, edge_index)
        x = F.relu(x) + res
        return x.contiguous().view(bsz, len_, -1)


class MapGraph(nn.Module):
    def __init__(self, config):
        super(MapGraph, self).__init__()
        input_size = config['input_size']
        hidden_sizes = config['hidden_sizes']

        self.convs = nn.ModuleList()

        channel_sizes = [input_size] + hidden_sizes
        for i, d in enumerate(hidden_sizes):
            self.convs.append(GATGraphConv(channel_sizes[i], channel_sizes[i + 1]))

        self.pred_layer = nn.Linear(channel_sizes[-1], 1)

    def forward(self, props_h, props_graph, **kwargs):
        x = props_h
        for c in self.convs:
            x = c(x, props_graph)
        x = self.pred_layer(x).squeeze(-1)
        return x

    def reset_parameters(self):
        self.pred_layer.reset_parameters()


class MapConv(nn.Module):
    def __init__(self, config):
        super(MapConv, self).__init__()
        input_size = config['input_size']
        hidden_sizes = config['hidden_sizes']
        kernel_sizes = config['kernel_sizes']
        strides = config['strides']
        paddings = config['paddings']
        dilations = config['dilations']
        self.convs = nn.ModuleList()
        assert len(hidden_sizes) == len(kernel_sizes) \
               and len(hidden_sizes) == len(strides) \
               and len(hidden_sizes) == len(paddings) \
               and len(hidden_sizes) == len(dilations)
        channel_sizes = [input_size] + hidden_sizes
        for i, (k, s, p, d) in enumerate(zip(kernel_sizes, strides, paddings, dilations)):
            self.convs.append(nn.Conv2d(channel_sizes[i], channel_sizes[i + 1], k, s, p, d))
        self.pred_layer = nn.Conv2d(hidden_sizes[-1], 1, 1, 1)

    def forward(self, map_h, map_mask, props, **kwargs):
        padded_mask = map_mask
        x = map_h
        for i, pred in enumerate(self.convs):
            # print(x.size())
            # x = F.relu(pred(x))
            x = torch.relu_(pred(x))
            padded_mask, masked_weight = get_padded_mask_and_weight(padded_mask, pred)
            x = x * masked_weight
        x = self.pred_layer(x)
        # print(x.size())
        # exit(0)
        x = x[:, 0, props[:, 0], props[:, 1] - 1]
        # print(x.size())
        # exit(0)
        return x

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        self.pred_layer.reset_parameters()
