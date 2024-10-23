import torch
import torch.nn as nn
import torch.nn.functional as F
class MultiscaleReconBlock(nn.Module):
    def __init__(self, in_channels, pro_channels):
        super(MultiscaleReconBlock, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, pro_channels, kernel_size=1)

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, pro_channels, kernel_size=1),
            nn.Conv2d(pro_channels, pro_channels, kernel_size=3, padding=1)
        )


        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, pro_channels, kernel_size=1),
            nn.Conv2d(pro_channels, pro_channels, kernel_size=5, padding=2)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pro_channels, kernel_size=1)
        )

        self.output_channels = 4 * pro_channels
        self.in_channels = in_channels
        self.output_layer = nn.Conv2d(self.output_channels, self.in_channels, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = torch.cat([branch1, branch2, branch3, branch4], 1)
        outputs = self.output_layer(outputs)
        outputs = outputs + x
        return outputs


class MultiScaleReconNet(nn.Module):
    def __init__(self, _config):
        super(MultiScaleReconNet, self).__init__()
        self.in_out_channels = _config.n_features
        self.n = _config.n_features
        self.pro_channels = _config.pro_features
        self.w = _config.window_size
        self.sw = _config.period_length
        self.b = _config.batch_size
        self.p = _config.padding_size
        self.h = int((self.w+self.p) / self.sw)
        self.k = _config.k_times
        self.pow = _config.pow
        self.multi_scale_conv_x1 = MultiscaleReconBlock(self.in_out_channels, self.pro_channels)
        self.multi_scale_conv_x2 = MultiscaleReconBlock(self.in_out_channels, self.pro_channels)

    def forward(self, x):
        b = x.shape[0]
        # batch may not be complete
        # 1D to 2D
        if self.p == 0:
            x = torch.reshape(x, (b, self.sw, self.h, self.n))
        else:
            x = F.pad(x, (0, 0, 0, self.p, 0, 0))
            x = torch.reshape(x, (b, self.sw, self.h, self.n))
        # swap feature dimension to conv channels
        x = x.permute(0, 3, 1, 2)
        # 重构
        reconstruct = self.multi_scale_conv_x1(x)
        reconstruct = self.multi_scale_conv_x2(reconstruct)
        # swap feature dimension to conv channels
        reconstruct = reconstruct.permute(0, 1, 2, 3)
        # 2D to 1D
        if self.p == 0:
            reconstruct = torch.reshape(reconstruct, (b, self.w, self.n))
        else:
            # 去除padding对应的位置
            reconstruct = torch.reshape(reconstruct, (b, self.h * self.sw, self.n))
            reconstruct = reconstruct[:, :self.w, :]
        return reconstruct
