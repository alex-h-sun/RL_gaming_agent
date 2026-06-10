"""Capture preprocessing: mock screen frames -> correct shape/dtype."""

import numpy as np
import pytest

from src.capture.preprocess import is_black_frame, to_observation


def test_to_observation_from_bgra():
    raw = np.random.randint(0, 256, size=(600, 300, 4), dtype=np.uint8)
    obs = to_observation(raw, size=84)
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8


def test_to_observation_from_rgb():
    raw = np.random.randint(0, 256, size=(1080, 500, 3), dtype=np.uint8)
    obs = to_observation(raw, size=84)
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8


def test_to_observation_rejects_bad_shapes():
    with pytest.raises(ValueError):
        to_observation(np.zeros((84, 84), dtype=np.uint8))
    with pytest.raises(ValueError):
        to_observation(np.zeros((84, 84, 2), dtype=np.uint8))


def test_black_frame_detection():
    assert is_black_frame(np.zeros((100, 100, 4), dtype=np.uint8))
    bright = np.full((100, 100, 4), 128, dtype=np.uint8)
    assert not is_black_frame(bright)


def test_mac_capture_pipeline_with_mocked_grab(monkeypatch):
    """MacCapture should produce 84x84x3 from a mocked window grab."""
    from src.capture import mac_capture

    rect = mac_capture.WindowRect(left=0, top=0, width=300, height=650)
    fake_raw = np.random.randint(10, 256, size=(650, 300, 4), dtype=np.uint8)

    monkeypatch.setattr(mac_capture, "find_window", lambda title: rect)
    monkeypatch.setattr(mac_capture, "grab_raw", lambda r: fake_raw)

    capture = mac_capture.MacCapture()
    obs = capture.capture()
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8
    assert capture.capture_raw().shape == (650, 300, 4)
