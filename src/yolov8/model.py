"""Full YOLOv8 model: backbone + neck + head."""

import torch
import torch.nn as nn

from yolov8.modules import Backbone, Neck, Head, yolo_params  # noqa: F401


class YOLO(nn.Module):
    """YOLOv8 detection model.

    The constructor also calibrates the head strides with a dummy forward
    pass, then sets the detection biases (which depend on the strides).
    """

    def __init__(self, version='n', num_classes=80, input_size=640):
        super().__init__()
        self.version = version
        self.num_classes = num_classes
        self.backbone = Backbone(version=version)
        self.neck = Neck(version=version)
        self.head = Head(version=version, num_classes=num_classes)

        self._initialize_strides(input_size=input_size)
        self.head.initialize_biases()

    def _initialize_strides(self, input_size):
        """Run a dummy forward pass to measure the stride of each scale."""
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
    # Quick sanity check of shapes and parameter counts.
    for v in ['n', 's']:
        m = YOLO(version=v, num_classes=80)
        n_params = sum(p.numel() for p in m.parameters()) / 1e6
        print(f"YOLOv8-{v}: {n_params:.4f} M parameters, "
              f"strides={m.head.stride.tolist()}")
        m.train()
        outputs = m(torch.zeros(2, 3, 640, 640))
        print(f"  training outputs: {[o.shape for o in outputs]}")
        m.eval()
        inf, raw = m(torch.zeros(2, 3, 640, 640))
        print(f"  eval inference: {inf.shape}, raw: {[o.shape for o in raw]}")
