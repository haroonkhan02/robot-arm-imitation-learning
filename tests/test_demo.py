import numpy as np
import pytest

pytest.importorskip("torch")

from robot_arm_il.demo import label, side_by_side  # noqa: E402


def test_side_by_side_holds_the_shorter_clip_and_the_ending():
    a = [np.full((4, 4, 3), i, np.uint8) for i in range(3)]
    b = [np.full((4, 4, 3), 10 + i, np.uint8) for i in range(5)]
    out = side_by_side(a, b, step=2, hold=2)
    assert len(out) == 5 and out[0].shape == (4, 8, 3)
    assert out[-1][0, 0, 0] == 2 and out[-1][0, -1, 0] == 14


def test_label_keeps_the_shape_and_draws_on_the_frame():
    frame = np.full((40, 120, 3), 200, np.uint8)
    out = label(frame, "ACT")
    assert out.shape == frame.shape and (out != frame).any()
