"""V4L2 ioctl helpers (D-39): request numbers must equal the Linux UAPI values."""

from hilreview.robot.v4l2_controls import VIDIOC_G_CTRL, VIDIOC_QUERYCTRL, VIDIOC_S_CTRL, normalize_name


def test_ioctl_numbers_match_linux_uapi():
    assert VIDIOC_QUERYCTRL == 0xC0445624
    assert VIDIOC_G_CTRL == 0xC008561B
    assert VIDIOC_S_CTRL == 0xC008561C


def test_normalize_name_like_v4l2_ctl():
    assert normalize_name("Exposure Time, Absolute") == "exposure_time_absolute"
    assert normalize_name("White Balance Temperature, Auto") == "white_balance_temperature_auto"
    assert normalize_name("Auto Exposure") == "auto_exposure"
