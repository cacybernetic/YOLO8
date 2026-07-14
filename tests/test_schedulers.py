"""LR scheduler tests: warmup shape, cosine decay, bias group."""

import torch

from yolov8.training import build_scheduler


def _optimizer_with_groups():
    w = torch.nn.Parameter(torch.zeros(2, 2))
    b = torch.nn.Parameter(torch.zeros(2))
    return torch.optim.SGD([
        {'params': [w], 'is_bias': False},
        {'params': [b], 'is_bias': True},
    ], lr=0.01, momentum=0.9)


def test_warmup_ramps_weights_up_and_biases_down():
    opt = _optimizer_with_groups()
    sched = build_scheduler('cosine', max_lr=0.01, min_lr=0.0001,
                            warmup_epochs=1, total_epochs=10,
                            num_steps=200, warmup_bias_lr=0.1)
    sched.step(0, opt)
    first_w = opt.param_groups[0]['lr']
    first_b = opt.param_groups[1]['lr']
    assert first_w < 0.001          # weights start near zero
    assert first_b > 0.05           # biases start near warmup_bias_lr

    sched.step(sched.warmup_steps - 1, opt)
    assert abs(opt.param_groups[0]['lr'] - 0.01) < 1e-9
    assert abs(opt.param_groups[1]['lr'] - 0.01) < 1e-9


def test_warmup_momentum_ramp():
    opt = _optimizer_with_groups()
    sched = build_scheduler('cosine', max_lr=0.01, min_lr=0.0001,
                            warmup_epochs=1, total_epochs=10,
                            num_steps=200, momentum=0.937,
                            warmup_momentum=0.8)
    sched.step(0, opt)
    assert opt.param_groups[0]['momentum'] < 0.85
    sched.step(sched.warmup_steps - 1, opt)
    assert abs(opt.param_groups[0]['momentum'] - 0.937) < 1e-9


def test_cosine_reaches_min_lr():
    opt = _optimizer_with_groups()
    sched = build_scheduler('cosine', max_lr=0.01, min_lr=0.0001,
                            warmup_epochs=1, total_epochs=10,
                            num_steps=200)
    last_step = sched.warmup_steps + sched.decay_steps
    sched.step(last_step, opt)
    assert abs(opt.param_groups[0]['lr'] - 0.0001) < 1e-9


def test_linear_reaches_min_lr():
    opt = _optimizer_with_groups()
    sched = build_scheduler('linear', max_lr=0.01, min_lr=0.0001,
                            warmup_epochs=1, total_epochs=10,
                            num_steps=200)
    last_step = sched.warmup_steps + sched.decay_steps
    sched.step(last_step, opt)
    assert abs(opt.param_groups[0]['lr'] - 0.0001) < 1e-9


def test_tiny_dataset_does_not_crash():
    # Warmup must never eat the whole step budget.
    opt = _optimizer_with_groups()
    sched = build_scheduler('cosine', max_lr=0.01, min_lr=0.0001,
                            warmup_epochs=3, total_epochs=2,
                            num_steps=3)
    for step in range(6):
        sched.step(step, opt)
    assert sched.decay_steps >= 1
