"""
YOLOv8 Implementation from scratch in PyTorch.

Modifications par rapport à la version initiale :
  - Import de `torch` ajouté
  - Classe `SPPF` définie (elle est utilisée par `Backbone` mais n'était pas définie)
  - `MyYolo` calcule et assigne automatiquement les strides de la tête de détection
    (sinon, `self.stride` vaut [0, 0, 0] et casse la fonction de loss)
  - `Head.forward` retourne en mode eval un tuple (inference_tensor, raw_outputs)
    afin de pouvoir calculer la loss en validation sans second forward
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Blocs de base
# ---------------------------------------------------------------------------

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.03)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super().__init__()
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut and in_channels == out_channels

    def forward(self, x):
        x_in = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.shortcut:
            x = x + x_in
        return x


class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_bottlenecks, shortcut=True):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.num_bottlenecks = num_bottlenecks

        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.ModuleList([
            Bottleneck(self.mid_channels, self.mid_channels, shortcut=shortcut)
            for _ in range(num_bottlenecks)
        ])
        self.conv2 = Conv((num_bottlenecks + 2) * out_channels // 2, out_channels,
                          kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x[:, :x.shape[1] // 2, :, :], x[:, x.shape[1] // 2:, :, :]
        outputs = [x1, x2]
        for i in range(self.num_bottlenecks):
            x1 = self.m[i](x1)
            outputs.insert(0, x1)
        outputs = torch.cat(outputs, dim=1)
        return self.conv2(outputs)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1, padding=0)
        self.cv2 = Conv(hidden_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


# ---------------------------------------------------------------------------
# Scaling YOLOv8
# ---------------------------------------------------------------------------

def yolo_params(version):
    if version == 'n':
        return 1 / 3, 1 / 4, 2.0
    elif version == 's':
        return 1 / 3, 1 / 2, 2.0
    elif version == 'm':
        return 2 / 3, 3 / 4, 1.5
    elif version == 'l':
        return 1.0, 1.0, 1.0
    elif version == 'x':
        return 1.0, 1.25, 1.0
    raise ValueError(f"Unknown YOLO version: {version}")


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class Backbone(nn.Module):
    def __init__(self, version, in_channels=3, shortcut=True):
        super().__init__()
        d, w, r = yolo_params(version)

        self.conv_0 = Conv(in_channels, int(64 * w), kernel_size=3, stride=2, padding=1)
        self.conv_1 = Conv(int(64 * w), int(128 * w), kernel_size=3, stride=2, padding=1)
        self.conv_3 = Conv(int(128 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.conv_5 = Conv(int(256 * w), int(512 * w), kernel_size=3, stride=2, padding=1)
        self.conv_7 = Conv(int(512 * w), int(512 * w * r), kernel_size=3, stride=2, padding=1)

        self.c2f_2 = C2f(int(128 * w), int(128 * w), num_bottlenecks=int(3 * d), shortcut=True)
        self.c2f_4 = C2f(int(256 * w), int(256 * w), num_bottlenecks=int(6 * d), shortcut=True)
        self.c2f_6 = C2f(int(512 * w), int(512 * w), num_bottlenecks=int(6 * d), shortcut=True)
        self.c2f_8 = C2f(int(512 * w * r), int(512 * w * r), num_bottlenecks=int(3 * d), shortcut=True)

        self.sppf = SPPF(int(512 * w * r), int(512 * w * r))

    def forward(self, x):
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.c2f_2(x)
        x = self.conv_3(x)
        out1 = self.c2f_4(x)
        x = self.conv_5(out1)
        out2 = self.c2f_6(x)
        x = self.conv_7(out2)
        x = self.c2f_8(x)
        out3 = self.sppf(x)
        return out1, out2, out3


# ---------------------------------------------------------------------------
# Neck (PAN-FPN)
# ---------------------------------------------------------------------------

class Neck(nn.Module):
    def __init__(self, version):
        super().__init__()
        d, w, r = yolo_params(version)

        self.up = Upsample()
        self.c2f_1 = C2f(int(512 * w * (1 + r)), int(512 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_2 = C2f(int(768 * w), int(256 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_3 = C2f(int(768 * w), int(512 * w), num_bottlenecks=int(3 * d), shortcut=False)
        self.c2f_4 = C2f(int(512 * w * (1 + r)), int(512 * w * r), num_bottlenecks=int(3 * d), shortcut=False)

        self.cv_1 = Conv(int(256 * w), int(256 * w), kernel_size=3, stride=2, padding=1)
        self.cv_2 = Conv(int(512 * w), int(512 * w), kernel_size=3, stride=2, padding=1)

    def forward(self, x_res_1, x_res_2, x):
        res_1 = x
        x = self.up(x)
        x = torch.cat([x, x_res_2], dim=1)
        res_2 = self.c2f_1(x)

        x = self.up(res_2)
        x = torch.cat([x, x_res_1], dim=1)
        out_1 = self.c2f_2(x)

        x = self.cv_1(out_1)
        x = torch.cat([x, res_2], dim=1)
        out_2 = self.c2f_3(x)

        x = self.cv_2(out_2)
        x = torch.cat([x, res_1], dim=1)
        out_3 = self.c2f_4(x)

        return out_1, out_2, out_3


# ---------------------------------------------------------------------------
# Head
# ---------------------------------------------------------------------------

class DFL(nn.Module):
    """Distribution Focal Loss integration layer."""
    def __init__(self, ch=16):
        super().__init__()
        self.ch = ch
        self.conv = nn.Conv2d(in_channels=ch, out_channels=1, kernel_size=1, bias=False).requires_grad_(False)
        x = torch.arange(ch, dtype=torch.float).view(1, ch, 1, 1)
        self.conv.weight.data[:] = torch.nn.Parameter(x)

    def forward(self, x):
        b, c, a = x.shape
        x = x.view(b, 4, self.ch, a).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        return x.view(b, 4, a)


class Head(nn.Module):
    def __init__(self, version, ch=16, num_classes=80):
        super().__init__()
        self.ch = ch
        self.coordinates = self.ch * 4
        self.nc = num_classes
        self.no = self.coordinates + self.nc
        self.stride = torch.zeros(3)  # Sera calibré par MyYolo après instantiation

        d, w, r = yolo_params(version=version)

        self.box = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w * r), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)),
        ])

        self.cls = nn.ModuleList([
            nn.Sequential(
                Conv(int(256 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
            nn.Sequential(
                Conv(int(512 * w * r), self.nc, kernel_size=3, stride=1, padding=1),
                Conv(self.nc, self.nc, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.nc, self.nc, kernel_size=1, stride=1)),
        ])

        self.dfl = DFL(ch=self.ch)

    def forward(self, x):
        # x = [out_1, out_2, out_3] (liste mutable requise)
        for i in range(len(self.box)):
            box = self.box[i](x[i])
            cls = self.cls[i](x[i])
            x[i] = torch.cat((box, cls), dim=1)

        if self.training:
            return x

        # --- Inférence ---
        anchors, strides = (i.transpose(0, 1) for i in self._make_anchors(x, self.stride))
        y = torch.cat([i.view(x[0].shape[0], self.no, -1) for i in x], dim=2)
        box, cls = y.split(split_size=(4 * self.ch, self.nc), dim=1)

        a, b = self.dfl(box).chunk(2, 1)
        a = anchors.unsqueeze(0) - a
        b = anchors.unsqueeze(0) + b
        box = torch.cat(tensors=((a + b) / 2, b - a), dim=1)

        inference_out = torch.cat(tensors=(box * strides, cls.sigmoid()), dim=1)
        # On retourne aussi les tenseurs bruts pour pouvoir calculer la loss en validation
        return inference_out, x

    @staticmethod
    def _make_anchors(x, strides, offset=0.5):
        assert x is not None
        anchor_tensor, stride_tensor = [], []
        dtype, device = x[0].dtype, x[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = x[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), float(stride), dtype=dtype, device=device))
        return torch.cat(anchor_tensor), torch.cat(stride_tensor)


# ---------------------------------------------------------------------------
# Modèle complet
# ---------------------------------------------------------------------------

class MyYolo(nn.Module):
    def __init__(self, version='n', num_classes=80, input_size=640):
        super().__init__()
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version, num_classes=num_classes)

        # Calibrage des strides de la tête via un forward factice
        self._initialize_strides(input_size=input_size)

    def _initialize_strides(self, input_size):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, input_size, input_size)
            out1, out2, out3 = self.backbone(x)
            n1, n2, n3 = self.neck(out1, out2, out3)
            strides = torch.tensor([
                input_size / n1.shape[-1],
                input_size / n2.shape[-1],
                input_size / n3.shape[-1],
            ], dtype=torch.float32)
        self.head.stride = strides
        if was_training:
            self.train()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x[0], x[1], x[2])
        return self.head(list(x))


if __name__ == "__main__":
    # Sanity check
    for v in ['n', 's']:
        m = MyYolo(version=v, num_classes=80)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"YOLOv8-{v}: {n_params:.4f} M parameters, strides={m.head.stride.tolist()}")
        m.train()
        outputs = m(torch.zeros(2, 3, 640, 640))
        print(f"  training outputs: {[o.shape for o in outputs]}")
        m.eval()
        inf, raw = m(torch.zeros(2, 3, 640, 640))
        print(f"  eval inference: {inf.shape}, raw: {[o.shape for o in raw]}")
