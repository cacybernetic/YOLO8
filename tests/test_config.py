"""Tests of the YAML config loader (scalar type coercion)."""

import pytest

from yolov8.config import TrainConfig, _from_dict


def test_yaml_11_scientific_notation_string_is_coerced_to_float():
    # PyYAML parses `1e-4` (no decimal point) as the string "1e-4".
    cfg = _from_dict(TrainConfig, {
        'optimization': {'max_lr': '1e-4', 'min_lr': '1e-6'},
    })
    assert isinstance(cfg.optimization.max_lr, float)
    assert cfg.optimization.max_lr == pytest.approx(1e-4)
    assert isinstance(cfg.optimization.min_lr, float)
    assert cfg.optimization.min_lr == pytest.approx(1e-6)


def test_scalar_coercion_int_bool_and_optional():
    cfg = _from_dict(TrainConfig, {
        'seed': '4',
        'optimization': {'epochs': 50.0, 'amp': 'false'},
        'dataset': {'max_train_samples': '100'},
    })
    assert cfg.seed == 4
    assert cfg.optimization.epochs == 50
    assert cfg.optimization.amp is False
    assert cfg.dataset.max_train_samples == 100


def test_invalid_scalar_raises_with_config_path():
    with pytest.raises(ValueError, match='optimization.max_lr'):
        _from_dict(TrainConfig, {'optimization': {'max_lr': 'abc'}})
    with pytest.raises(ValueError, match='optimization.epochs'):
        _from_dict(TrainConfig, {'optimization': {'epochs': 1.5}})
