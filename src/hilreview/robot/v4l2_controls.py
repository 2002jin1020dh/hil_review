"""Minimal V4L2 camera-control access via ioctl (Linux only, no v4l2-ctl / sudo needed).

Used to switch the experiment cameras from automatic to FIXED exposure / white balance, so the image of the same
scene does not change with what is in view (D-39). Values are reset by the driver when a camera is re-plugged or the
laptop reboots, so they are re-applied at the start of every session from a saved JSON file.
"""

from __future__ import annotations

import ctypes
import fcntl
import os
from dataclasses import asdict, dataclass

_IOC_WRITE, _IOC_READ = 1, 2
CTRL_FLAG_DISABLED = 0x0001
CTRL_FLAG_INACTIVE = 0x0010
CTRL_FLAG_NEXT_CTRL = 0x80000000
CTRL_TYPE_MENU = 3
CTRL_CLASS_MASK = 0x0FFF0000


class _QueryCtrl(ctypes.Structure):  # struct v4l2_queryctrl
    _fields_ = [("id", ctypes.c_uint32), ("type", ctypes.c_uint32), ("name", ctypes.c_char * 32),
                ("minimum", ctypes.c_int32), ("maximum", ctypes.c_int32), ("step", ctypes.c_int32),
                ("default_value", ctypes.c_int32), ("flags", ctypes.c_uint32), ("reserved", ctypes.c_uint32 * 2)]


class _Control(ctypes.Structure):  # struct v4l2_control
    _fields_ = [("id", ctypes.c_uint32), ("value", ctypes.c_int32)]


def _iowr(nr: int, size: int) -> int:
    return ((_IOC_READ | _IOC_WRITE) << 30) | (size << 16) | (ord("V") << 8) | nr


VIDIOC_QUERYCTRL = _iowr(36, ctypes.sizeof(_QueryCtrl))
VIDIOC_G_CTRL = _iowr(27, ctypes.sizeof(_Control))
VIDIOC_S_CTRL = _iowr(28, ctypes.sizeof(_Control))

# names as the kernel reports them, normalized like v4l2-ctl ("Exposure Time, Absolute" -> exposure_time_absolute)
AUTO_EXPOSURE = "auto_exposure"  # menu: 1 = manual, 3 = aperture priority (auto)
EXPOSURE_ABS = "exposure_time_absolute"
EXPOSURE_DYN_FPS = "exposure_dynamic_framerate"  # if on, auto may drop the frame rate
WB_AUTO = "white_balance_automatic"
WB_TEMP = "white_balance_temperature"
POWER_LINE = "power_line_frequency"  # menu: 0 off, 1 = 50 Hz, 2 = 60 Hz (Korea: 60 Hz mains)


def normalize_name(raw: str) -> str:
    out = []
    for ch in raw.lower():
        out.append(ch if ch.isalnum() else "_")
    s = "".join(out)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")


@dataclass(frozen=True)
class Ctrl:
    id: int
    name: str
    type: int
    minimum: int
    maximum: int
    step: int
    default: int
    flags: int
    value: int | None

    @property
    def inactive(self) -> bool:
        return bool(self.flags & CTRL_FLAG_INACTIVE)

    def to_dict(self) -> dict:
        return asdict(self)


class Camera:
    def __init__(self, path: str):
        self.path = path
        self.fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)

    def close(self) -> None:
        os.close(self.fd)

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get(self, cid: int) -> int:
        c = _Control(cid, 0)
        fcntl.ioctl(self.fd, VIDIOC_G_CTRL, c)
        return int(c.value)

    def set(self, cid: int, value: int) -> None:
        fcntl.ioctl(self.fd, VIDIOC_S_CTRL, _Control(cid, int(value)))

    def controls(self) -> dict[str, Ctrl]:
        out: dict[str, Ctrl] = {}
        q = _QueryCtrl()
        q.id = CTRL_FLAG_NEXT_CTRL
        while True:
            try:
                fcntl.ioctl(self.fd, VIDIOC_QUERYCTRL, q)
            except OSError:
                break
            is_class = (q.id & 0xFFFF) == 1 and q.type == 6  # V4L2_CTRL_TYPE_CTRL_CLASS
            if not (q.flags & CTRL_FLAG_DISABLED) and not is_class:
                try:
                    val = self.get(q.id)
                except OSError:
                    val = None
                name = normalize_name(q.name.decode(errors="replace"))
                out[name] = Ctrl(q.id, name, q.type, q.minimum, q.maximum, q.step, q.default_value, q.flags, val)
            q.id = q.id | CTRL_FLAG_NEXT_CTRL
        return out

    def set_by_name(self, values: dict[str, int]) -> dict[str, int]:
        """Set controls in the given order (put auto switches first); returns the read-back values."""
        ctrls = self.controls()
        for name, v in values.items():
            if name not in ctrls:
                raise KeyError(f"{self.path}: no control {name!r}; available {sorted(ctrls)}")
            c = ctrls[name]
            if not c.minimum <= int(v) <= c.maximum:
                raise ValueError(f"{self.path}: {name}={v} outside [{c.minimum}, {c.maximum}]")
            self.set(c.id, int(v))
        after = self.controls()
        return {n: after[n].value for n in values}
