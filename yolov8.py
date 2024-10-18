# YOLOv8 model

import torch
import torch.nn as nn
import math

class Conv(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride, bn_act=True):
    super().__init__()

    padding = (kernel_size - 1) // 2

    self.conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=not bn_act
    )
    self.bn = nn.BatchNorm2d(out_channels, 0.001, 0.03)
    self.silu = torch.nn.SiLU(inplace=True)
    self.use_bn_act = bn_act

  def forward(self, x):
    if self.use_bn_act:
      return self.silu(self.bn(self.conv(x)))
    else:
      return self.conv(x)


class Bottleneck(nn.Module):
  def __init__(self, in_channels, cat=True):
    super().__init__()

    self.cat = cat
    self.res = torch.nn.Sequential(
        Conv(in_channels, in_channels, 3, 1),
        Conv(in_channels, in_channels, 3, 1)
    )

  def forward(self, x):
    if self.cat:
      return self.res(x) + x
    else:
      return self.res(x)


class C2f(nn.Module):
  def __init__(self, in_channels, out_channels, n=1, cat=True):
    super().__init__()

    self.split_1 = Conv(in_channels, out_channels//2, 1, 1)
    self.split_2 = Conv(in_channels, out_channels//2, 1, 1)
    self.conv = Conv((n+2)*out_channels//2, out_channels, 1, 1)
    self.res_n = torch.nn.ModuleList([Bottleneck(out_channels//2, cat=cat) for _ in range(n)])

  def forward(self, x):
    split_1 = self.split_1(x)
    split_2 = self.split_2(x)

    y = [split_1, split_2]
    y.extend(b(y[-1]) for b in self.res_n)
    return self.conv(torch.cat(y, dim=1))


class SPPF(nn.Module):
  def __init__(self, in_channels, k=5):
    super().__init__()

    self.conv_1 = Conv(in_channels, in_channels//2, 1, 1)
    self.conv_2 = Conv(in_channels*2, in_channels, 1, 1)
    self.max_pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

  def forward(self, x):
    x = self.conv_1(x)
    m1 = self.max_pool(x)
    m2 = self.max_pool(m1)
    m3 = self.max_pool(m2)
    y = torch.cat([x, m1, m2, m3], dim=1)
    return self.conv_2(y)


class DarkNet(nn.Module):
  def __init__(self, w, d):
    super().__init__()

    self.P3 = torch.nn.Sequential(
        Conv(w[0], w[1], 3, 2),
        Conv(w[1], w[2], 3, 2),
        C2f(w[2], w[2], d[0]),
        Conv(w[2], w[3], 3, 2),
        C2f(w[3], w[3], d[1])
    )
    self.P4 = torch.nn.Sequential(
        Conv(w[3], w[4], 3, 2),
        C2f(w[4], w[4], d[2])
    )
    self.P5 = torch.nn.Sequential(
        Conv(w[4], w[5], 3, 2),
        C2f(w[5], w[5], d[0]),
        SPPF(w[5], k=5)
    )

  def forward(self, x):
    P3 = self.P3(x)
    P4 = self.P4(P3)
    P5 = self.P5(P4)
    return P3, P4, P5


class FPN(nn.Module):
  def __init__(self, w, d):
    super().__init__()

    self.up = torch.nn.Upsample(None, 2)
    self.h1 = C2f(w[4]+w[5], w[4], d[0], False)
    self.h2 = C2f(w[3]+w[4], w[3], d[0], False)
    self.h3 = Conv(w[3], w[3], 3, 2)
    self.h4 = C2f(w[4]+w[3], w[4], d[0], False)
    self.h5 = Conv(w[4], w[4], 3, 2)
    self.h6 = C2f(w[5]+w[4], w[5], d[0], False)

  def forward(self, P5, P4, P3):
    N5 = P5
    N4 = self.h1(torch.cat([self.up(P5), P4], dim=1))
    N3 = self.h2(torch.cat([self.up(N4), P3], dim=1))
    C3 = N3
    C4 = self.h4(torch.cat([self.h3(C3), N4], dim=1))
    C5 = self.h6(torch.cat([self.h5(C4), N5], dim=1))
    return C3, C4, C5


class Detect(nn.Module):
  def __init__(self, in_channels, num_class=36):
    super().__init__()

    self.nc = num_class
    self.conv = Conv(in_channels, in_channels*2, 3, 1)
    self.detect = Conv(in_channels*2, 3*(num_class+5), 1, 1, bn_act=False)

  def forward(self, x):
    return(
        self.detect(self.conv(x)) # x = [batch_num, 3*(num_classes + 5), N, N
        .reshape(x.shape[0], 3, self.nc+5, x.shape[2], x.shape[3])
        .permute(0, 1, 3, 4, 2)   # output = [B x 3 x N x N x 5+num_classes]
    )

class YOLOv8(nn.Module):
  def __init__(self, w, d, num_class=36):
    super().__init__()

    self.darknet = DarkNet(w, d)
    self.fpn = FPN(w, d)
    self.sbbox = Detect(w[3], num_class=num_class)
    self.mbbox = Detect(w[4], num_class=num_class)
    self.lbbox = Detect(w[5], num_class=num_class)

  def forward(self, x):
    P3, P4, P5 = self.darknet(x)
    C3, C4, C5 = self.fpn(P5, P4, P3)
    # C3 = 80x80
    # C4 = 40x40
    # C5 = 20x20
    return [self.lbbox(C5), self.mbbox(C4), self.sbbox(C3)]


def yolo_v8_n(num_classes = 36):
    depth = [1, 2, 2]
    width = [3, 16, 32, 64, 128, 256]
    return YOLOv8(width, depth, num_classes)


def yolo_v8_s(num_classes = 36):
    depth = [1, 2, 2]
    width = [3, 32, 64, 128, 256, 512]
    return YOLOv8(width, depth, num_classes)


def yolo_v8_m(num_classes = 36):
    depth = [2, 4, 4]
    width = [3, 48, 96, 192, 384, 576]
    return YOLOv8(width, depth, num_classes)


def yolo_v8_l(num_classes = 36):
    depth = [3, 6, 6]
    width = [3, 64, 128, 256, 512, 512]
    return YOLOv8(width, depth, num_classes)


def yolo_v8_x(num_classes = 36):
    depth = [3, 6, 6]
    width = [3, 80, 160, 320, 640, 640]
    return YOLOv8(width, depth, num_classes)