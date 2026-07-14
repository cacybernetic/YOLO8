"""Run folder naming and resume selection tests."""

from yolov8.training import prepare_run_dir, list_run_dirs


def _touch_checkpoint(run_dir):
    ckpt_dir = run_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (ckpt_dir / 'checkpoint_e0001c0000.pth').write_bytes(b'x')


def test_first_run_has_no_number(tmp_path):
    run_dir, resumed = prepare_run_dir(tmp_path, 'demo', 'train')
    assert run_dir.name == 'train'
    assert not resumed
    for sub in ('weights', 'checkpoints', 'plotes', 'logs'):
        assert (run_dir / sub).is_dir()


def test_second_run_is_train2(tmp_path):
    prepare_run_dir(tmp_path, 'demo', 'train')
    run_dir, _ = prepare_run_dir(tmp_path, 'demo', 'train')
    assert run_dir.name == 'train2'
    run_dir, _ = prepare_run_dir(tmp_path, 'demo', 'train')
    assert run_dir.name == 'train3'


def test_resume_picks_highest_numbered_with_checkpoint(tmp_path):
    d1, _ = prepare_run_dir(tmp_path, 'demo', 'train')
    d2, _ = prepare_run_dir(tmp_path, 'demo', 'train')
    _touch_checkpoint(d1)
    _touch_checkpoint(d2)
    run_dir, resumed = prepare_run_dir(tmp_path, 'demo', 'train',
                                       resume=True)
    assert resumed
    assert run_dir == d2  # highest number wins


def test_resume_without_checkpoint_creates_new(tmp_path):
    prepare_run_dir(tmp_path, 'demo', 'train')
    run_dir, resumed = prepare_run_dir(tmp_path, 'demo', 'train',
                                       resume=True)
    assert not resumed
    assert run_dir.name == 'train2'


def test_eval_runs_are_independent(tmp_path):
    run_dir, _ = prepare_run_dir(tmp_path, 'demo', 'eval')
    assert run_dir.name == 'eval'
    run_dir, _ = prepare_run_dir(tmp_path, 'demo', 'eval')
    assert run_dir.name == 'eval2'
    assert len(list_run_dirs(tmp_path / 'demo', 'eval')) == 2
