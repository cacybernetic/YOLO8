"""Optimizer construction with YOLOv8 style parameter groups."""

import torch
from loguru import logger


def build_param_groups(model, weight_decay):
    """Split the parameters in three groups (YOLOv8 convention):
      0. weights with dim >= 2      -> weight decay
      1. 1D weights (BatchNorm...)  -> no decay
      2. biases                     -> no decay, tagged `is_bias` so the
                                       scheduler can give them their own
                                       warmup learning rate.
    """
    p_bias, p_nodecay, p_decay = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith('.bias'):
            p_bias.append(param)
        elif param.ndim <= 1:
            p_nodecay.append(param)
        else:
            p_decay.append(param)
    return [
        {'params': p_decay, 'weight_decay': weight_decay,
         'is_bias': False},
        {'params': p_nodecay, 'weight_decay': 0.0, 'is_bias': False},
        {'params': p_bias, 'weight_decay': 0.0, 'is_bias': True},
    ]


def build_optimizer(model, name='sgd', lr=0.01, momentum=0.937,
                    weight_decay=0.0005):
    """Build the optimizer over the three parameter groups.

    Supported names: 'sgd' (nesterov), 'adam', 'adamw'.
    """
    groups = build_param_groups(model, weight_decay)
    name = name.lower()
    if name == 'sgd':
        return torch.optim.SGD(groups, lr=lr, momentum=momentum,
                               nesterov=True)
    if name == 'adam':
        return torch.optim.Adam(groups, lr=lr)
    if name == 'adamw':
        return torch.optim.AdamW(groups, lr=lr)
    raise ValueError(f"Unknown optimizer: {name}")


def freeze_feature_layers(model):
    """Freeze the backbone and the neck for head-only fine tuning.

    When the target dataset is small or close to the source domain,
    freezing the feature extractor:
      - cuts the number of trained parameters (less overfitting),
      - keeps the generic pretrained features,
      - speeds up training (no backward pass in backbone + neck).

    BatchNorm modules inside the frozen blocks are also switched to
    eval mode: without that, their running statistics would still be
    updated at every forward pass.
    """
    n_frozen_params = 0
    for name, param in model.named_parameters():
        if name.startswith('backbone.') or name.startswith('neck.'):
            param.requires_grad = False
            n_frozen_params += param.numel()

    n_bn_frozen = apply_batchnorm_freeze(model)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Backbone + neck frozen: "
                f"{n_frozen_params / 1e6:.3f}M params frozen, "
                f"{n_bn_frozen} BatchNorm layers set to eval()")
    logger.info(f"Trainable params: {trainable / 1e6:.3f}M / "
                f"{total / 1e6:.3f}M ({100 * trainable / total:.1f}%)")
    return n_frozen_params


def apply_batchnorm_freeze(model):
    """Set backbone/neck BatchNorm layers to eval. Returns the count.

    Must be called again after every model.train(), because train()
    wakes up every submodule.
    """
    bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d,
                torch.nn.BatchNorm3d)
    count = 0
    for mod_name, module in model.named_modules():
        if not (mod_name.startswith('backbone.')
                or mod_name.startswith('neck.')):
            continue
        if isinstance(module, bn_types):
            module.eval()
            count += 1
    return count
