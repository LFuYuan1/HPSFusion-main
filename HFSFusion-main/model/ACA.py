import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveCoordAtt(nn.Module):
    def __init__(self, in_channels, reduction=16, alpha=0.9):
        super(AdaptiveCoordAtt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = max(8, in_channels // reduction)
        self.alpha=alpha

        # Shared MLP: Conv1x1 -> BN -> ReLU
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_h = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_w = nn.Conv2d(self.mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        b, c, h, w = x.size()

        # Adaptive pooling along H and W
        x_h = F.adaptive_avg_pool2d(x, (h, 1))
        x_w = F.adaptive_avg_pool2d(x, (1, w))
        x_w = x_w.permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.shared_conv(y)

        x_h_out, x_w_out = torch.split(y, [h, w], dim=2)
        x_w_out = x_w_out.permute(0, 1, 3, 2)   # 调整回原来维度

        # 分别映射到通道数
        a_h = self.conv_h(x_h_out*self.alpha).sigmoid()
        a_w = self.conv_w(x_w_out*self.alpha).sigmoid()

        # 融合：逐通道加权
        out = x * (a_h + a_w)
        return out
