from crpod.dataset.side import infer_video_side
from crpod.types import Side


def test_y_above_midpoint_is_friendly():
    assert infer_video_side(481, 960) is Side.FRIENDLY


def test_y_below_midpoint_is_enemy():
    assert infer_video_side(479, 960) is Side.ENEMY


def test_y_at_exact_midpoint_is_enemy():
    # Boundary: y == frame_height/2 fails the strict `>` check, so it
    # classifies as enemy. This biases ambiguous near-river plays toward
    # the enemy side, which matches the dataset's framing convention.
    assert infer_video_side(480, 960) is Side.ENEMY


def test_y_zero_is_enemy():
    assert infer_video_side(0, 960) is Side.ENEMY


def test_y_at_frame_height_is_friendly():
    assert infer_video_side(960, 960) is Side.FRIENDLY


def test_full_hd_portrait_split():
    assert infer_video_side(961, 1920) is Side.FRIENDLY
    assert infer_video_side(959, 1920) is Side.ENEMY
