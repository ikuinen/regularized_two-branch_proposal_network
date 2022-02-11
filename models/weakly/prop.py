from torch import nn


class SparsePropMaxPool(nn.Module):
    def __init__(self, config):
        super(SparsePropMaxPool, self).__init__()
        self.num_scale_layers = config['num_scale_layers']

        self.layers = nn.ModuleList()

        for scale_idx, num_layer in enumerate(self.num_scale_layers):
            scale_layers = nn.ModuleList()
            if scale_idx == 0:
                first_layer = nn.MaxPool1d(1, 1)
            elif scale_idx == 1:
                first_layer = nn.MaxPool1d(3, 2)
            else:
                # first_layer = nn.MaxPool1d(5, 4)
                first_layer = nn.MaxPool1d(3, 2)
            rest_layers = [nn.MaxPool1d(2, 1) for _ in range(1, num_layer)]
            scale_layers.extend([first_layer] + rest_layers)
            self.layers.append(scale_layers)

    def reset_parameters(self):
        pass

    def forward(self, x, props, **kwargs):
        batch_size, hidden_size, ori_num_clips = x.size()
        acum_layers = 0
        stride = 1
        ori_map_h = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)
        ori_map_mask = x.new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
        for scale_idx, scale_layers in enumerate(self.layers):
            for i, layer in enumerate(scale_layers):
                stride = stride * layer.stride
                x = layer(x)
                ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride))
                ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]
                ori_map_h[:, :, ori_s_idxs, ori_e_idxs] = x
                ori_map_mask[:, :, ori_s_idxs, ori_e_idxs] = 1
                # print(ori_s_idxs)
                # print(ori_e_idxs)
                # print('=====================')
            acum_layers += stride * (len(scale_layers) + 1)
        props_h = ori_map_h[:, :, props[:, 0], props[:, 1] - 1]

        # ori_map_h1 = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)
        # ori_map_h1[:, :, props[:, 0], props[:, 1] - 1] = ori_map_h[:, :, props[:, 0], props[:, 1] - 1]
        ori_map_mask[:, :, props[:, 0], props[:, 1] - 1] = 1
        # ori_map_h = ori_map_h1
        # ori_map_h *= ori_map_mask
        # print(props[:, 0], props[:, 1] - 1)
        # exit(0)
        return props_h.transpose(1, 2), ori_map_h, ori_map_mask


class DensePropMaxPool(nn.Module):
    def __init__(self, config):
        super(DensePropMaxPool, self).__init__()
        num_layers = config['num_layers']
        self.layers = nn.ModuleList(
            [nn.Identity()]
            + [nn.MaxPool1d(2, stride=1) for _ in range(num_layers - 1)]
        )
        self.num_layers = num_layers

    def forward(self, x, props, **kwargs):
        batch_size, hidden_size, num_clips = x.shape
        map_h = x.new_zeros(batch_size, hidden_size, num_clips, num_clips).cuda()
        map_mask = x.new_zeros(batch_size, 1, num_clips, num_clips).cuda()

        for dig_idx, pool in enumerate(self.layers):
            x = pool(x)
            start_idxs = [s_idx for s_idx in range(0, num_clips - dig_idx, 1)]
            end_idxs = [s_idx + dig_idx for s_idx in start_idxs]
            map_h[:, :, start_idxs, end_idxs] = x
            map_mask[:, :, start_idxs, end_idxs] = 1
        props_h = map_h[:, :, props[:, 0], props[:, 1] - 1]
        # map_mask[:, :, props[:, 0], props[:, 1] - 1] = 1
        # print(props[:, 0], props[:, 1] - 1)
        # exit(0)
        return props_h.transpose(1, 2), map_h, map_mask

    # def forward(self, x, props, **kwargs):
    #     batch_size, hidden_size, ori_num_clips = x.size()
    #     acum_layers = 0
    #     stride = 1
    #     ori_map_h = x.new_zeros(batch_size, hidden_size, ori_num_clips, ori_num_clips)
    #     ori_map_mask = x.new_zeros(batch_size, 1, ori_num_clips, ori_num_clips)
    #     for scale_idx, scale_layers in enumerate(self.layers):
    #         for i, layer in enumerate(scale_layers):
    #             stride = stride * layer.stride
    #             x = layer(x)
    #             ori_s_idxs = list(range(0, ori_num_clips - acum_layers - i * stride, stride))
    #             ori_e_idxs = [s_idx + acum_layers + i * stride for s_idx in ori_s_idxs]
    #             ori_map_h[:, :, ori_s_idxs, ori_e_idxs] = x
    #             # print(ori_s_idxs)
    #             # print(ori_e_idxs)
    #             # print('=====================')
    #         acum_layers += stride * (len(scale_layers) + 1)
    #     props_h = ori_map_h[:, :, props[:, 0], props[:, 1] - 1]
    #     ori_map_mask[:, :, props[:, 0], props[:, 1] - 1] = 1
    #     # print(props[:, 0], props[:, 1] - 1)
    #     # exit(0)
    #     return props_h.transpose(1, 2), ori_map_h, ori_map_mask
