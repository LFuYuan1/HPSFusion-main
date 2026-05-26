import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CLC(nn.Module):
    """
    CLCk: Conv k×k -> LeakyReLU -> Conv k×k
    """
    def __init__(self, in_ch, out_ch=None, k=3, negative_slope=0.1):
        super().__init__()
        if out_ch is None:
            out_ch = in_ch
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
        )

    def forward(self, x):
        return self.net(x)

def _choose_gn_groups(C: int) -> int:
    # 选择能被C整除的最大不超过32的组数（至少1）
    for g in [32, 16, 8, 4, 2, 1]:
        if C % g == 0:
            return g
    return 1

class DNRU(nn.Module):
    """
    DNRU: 深度卷积 3×3 + GroupNorm + ReLU + 上采样
    """
    def __init__(self, channels, up_scale=1):
        super().__init__()
        self.dwconv = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.gn = nn.GroupNorm(_choose_gn_groups(channels), channels)
        self.relu = nn.ReLU(inplace=True)
        self.up_scale = up_scale

    def forward(self, x):
        x = self.dwconv(x)
        x = self.gn(x)
        x = self.relu(x)
        if self.up_scale and self.up_scale != 1:
            x = F.interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
        return x

# 通道向量 <-> 2D 网格 的 reshape 工具（为了做 2D FFT，将 C 维通道描述铺成接近方形的网格）
def vector_to_grid(x_vec):
    """
    x_vec: (B, C, 1, 1) -> (B, 1, Hc, Wc), 同时返回 (Hc, Wc, C, pad)
    """
    B, C, _, _ = x_vec.shape
    Hc = int(math.floor(math.sqrt(C)))
    Wc = int(math.ceil(C / Hc))
    pad = Hc * Wc - C
    if pad > 0:
        x_vec = F.pad(x_vec.view(B, C), (0, pad))
        C_ = C + pad
    else:
        x_vec = x_vec.view(B, C)
        C_ = C
    grid = x_vec.view(B, 1, Hc, Wc)
    return grid, (Hc, Wc, C, pad)


def grid_to_vector(grid, meta):
    """
    grid: (B, 1, Hc, Wc) -> (B, C, 1, 1)
    """
    Hc, Wc, C, pad = meta
    B = grid.size(0)
    vec = grid.view(B, Hc * Wc)
    if pad > 0:
        vec = vec[:, :C]
    return vec.view(B, C, 1, 1)

class FourierResidualChannelAttention(nn.Module):
    def __init__(self, channels, negative_slope=0.1, up_scale=1):
        super().__init__()
        self.channels = channels
        self.clc3 = CLC(channels, channels, k=3, negative_slope=negative_slope)
        self.clc1_amp = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )
        self.clc1_pha = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
        )
        self.dnru = DNRU(channels, up_scale=up_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        assert C == self.channels, "channels mismatch"
        feat = self.clc3(x)  # (B, C, H, W)
        chan_desc = F.adaptive_avg_pool2d(feat, 1)
        grid, meta = vector_to_grid(chan_desc)
        spec = torch.fft.fft2(grid)
        amp = torch.abs(spec)
        pha = torch.angle(spec)

        # 对振幅/相位做 CLC1 调制
        amp = amp * self.clc1_amp(amp)
        pha = pha * self.clc1_pha(pha)
        spec_new = torch.polar(amp, pha)
        grid_ifft = torch.fft.ifft2(spec_new).real
        weight_vec = grid_to_vector(grid_ifft, meta)
        weight = torch.sigmoid(weight_vec)
        y = feat * weight
        out = y + x

        # DNRU
        out = self.dnru(out)
        return out
