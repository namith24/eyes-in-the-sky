"""Drone link: one live-frame source for any drone the laptop can see.

The ground-station UI should not care *how* the drone's video reaches the laptop.
``DroneLink`` hides that behind a background reader thread that always holds the
newest frame (old frames are dropped, so the picture never lags behind the drone),
reconnects on its own when the signal drops, and reports link health.

Supported sources (``parse_source`` accepts any of these strings):

=====================  ===================================================  =============
Source string          What it is                                           Kind
=====================  ===================================================  =============
``http://...``         ESP32-CAM / IP-camera MJPEG stream                   ``http``
``rtsp://`` ``rtmp://``  Drone or controller video stream (RTSP/RTMP/UDP/SRT) ``stream``
``udp://`` ``srt://``
``0``, ``1`` ...       USB webcam or HDMI capture card (e.g. drone RC HDMI)  ``device``
``window:<title>``     The drone's *own app* running on this laptop:         ``window``
                       its window is captured, so whatever the vendor
                       software shows appears in our UI
``screen`` ``screen:2``  A whole monitor                                     ``screen``
``tello``              DJI/Ryze Tello (sends ``command`` + ``streamon``)     ``tello``
anything else          Path to a recorded flight video                       ``file``
=====================  ===================================================  =============

Example::

    link = DroneLink("http://192.168.4.1:81/stream").start()
    frame = link.latest()          # Frame(image, index, timestamp) or None
    print(link.stats())            # status, fps, resolution, reconnects, ...
    link.stop()
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, NamedTuple, Optional, Union
from urllib.parse import urlparse

import cv2
import numpy as np

# Low-latency FFmpeg defaults for network streams (only if the user has not set their own).
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "fflags;nobuffer|flags;low_delay")

STREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "udp", "tcp", "srt")
HTTP_SCHEMES = ("http", "https")
DEFAULT_PORTS = {"http": 80, "https": 443, "rtsp": 554, "rtsps": 322, "rtmp": 1935}
TELLO_IP, TELLO_CMD_PORT, TELLO_VIDEO = "192.168.10.1", 8889, "udp://0.0.0.0:11111"


# ---------------------------------------------------------------------------- sources
@dataclass(frozen=True)
class SourceSpec:
    """A parsed video source."""

    kind: str                      # http | stream | device | window | screen | tello | file
    target: Union[str, int]
    label: str

    @property
    def is_live(self) -> bool:
        return self.kind != "file"


def parse_source(text: Union[str, int]) -> SourceSpec:
    """Turn a user-typed source string into a ``SourceSpec``.

    Raises:
        ValueError: if the string is empty.
    """
    t = str(text).strip()
    if not t:
        raise ValueError("No video source given")
    low = t.lower()
    if t.isdigit():
        return SourceSpec("device", int(t), f"Camera / capture card #{t}")
    if low.startswith("window:"):
        title = t.split(":", 1)[1].strip()
        if not title:
            raise ValueError("window: needs part of the window title, e.g. window:DJI")
        return SourceSpec("window", title, f"Drone app window '{title}'")
    if low == "screen" or low.startswith("screen:"):
        n = t.split(":", 1)[1].strip() if ":" in t else "1"
        return SourceSpec("screen", int(n) if n.isdigit() else 1, f"Screen {n}")
    if low == "tello":
        return SourceSpec("tello", TELLO_VIDEO, "Tello drone")
    scheme = urlparse(t).scheme.lower()
    if scheme in HTTP_SCHEMES:
        return SourceSpec("http", t, f"HTTP stream {urlparse(t).netloc}")
    if scheme in STREAM_SCHEMES:
        return SourceSpec("stream", t, f"{scheme.upper()} stream {urlparse(t).netloc}")
    return SourceSpec("file", t, f"Recorded video {Path(t).name}")


def tcp_reachable(host: str, port: int, timeout: float = 1.0) -> bool:
    """True if something accepts a TCP connection at host:port within ``timeout``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _check_reachable(url: str, timeout: float) -> None:
    p = urlparse(url)
    scheme = p.scheme.lower()
    if scheme not in DEFAULT_PORTS or not p.hostname:
        return  # UDP/SRT: nothing to probe, let FFmpeg try
    port = p.port or DEFAULT_PORTS[scheme]
    if not tcp_reachable(p.hostname, port, timeout):
        raise ConnectionError(f"Nothing is answering at {p.hostname}:{port}. "
                              "Is the laptop on the drone's WiFi and the drone powered on?")


# ---------------------------------------------------------------------------- windows (Win32)
def _user32():
    import ctypes
    return ctypes.windll.user32


def _make_dpi_aware() -> None:
    """Make window coordinates match real pixels on scaled (125%/150%) displays."""
    if sys.platform != "win32":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


SHELL_WINDOWS = {"Program Manager", "Windows Input Experience", "Windows Shell Experience Host"}


def _is_cloaked(hwnd) -> bool:
    """Suspended UWP apps report 'visible' but are cloaked (not actually on screen)."""
    import ctypes
    from ctypes import wintypes
    cloaked = wintypes.DWORD(0)
    try:
        ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
    except Exception:
        return False
    return bool(cloaked.value)


def list_windows(min_size: int = 160) -> List[str]:
    """Titles of visible, non-minimised top-level windows (Windows only; [] elsewhere)."""
    if sys.platform != "win32":
        return []
    import ctypes
    from ctypes import wintypes

    _make_dpi_aware()
    user32 = _user32()
    titles: List[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and not user32.IsIconic(hwnd) and not _is_cloaked(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                r = wintypes.RECT()
                user32.GetClientRect(hwnd, ctypes.byref(r))
                if (r.right >= min_size and r.bottom >= min_size and buf.value not in titles
                        and buf.value not in SHELL_WINDOWS):
                    titles.append(buf.value)
        return True

    user32.EnumWindows(_cb, 0)
    return titles


def _find_window(title_part: str):
    """Handle of the first visible window whose title contains ``title_part`` (case-insensitive)."""
    import ctypes
    from ctypes import wintypes

    user32 = _user32()
    want, found = title_part.lower(), []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _cb(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd) and not _is_cloaked(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if want in buf.value.lower():
                    found.append(hwnd)
                    return False
        return True

    user32.EnumWindows(_cb, 0)
    return found[0] if found else None


def _client_rect(hwnd):
    """Screen rectangle (left, top, width, height) of a window's content area, or None."""
    import ctypes
    from ctypes import wintypes

    user32 = _user32()
    if not user32.IsWindow(hwnd) or user32.IsIconic(hwnd):
        return None
    r = wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(r))
    pt = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    if r.right <= 0 or r.bottom <= 0:
        return None
    return pt.x, pt.y, r.right, r.bottom


# ---------------------------------------------------------------------------- readers
class _Reader:
    paced = False      # True for files: play back at the video's own speed

    def read(self):    # -> (ok, frame)
        raise NotImplementedError

    def close(self):
        pass


class _CaptureReader(_Reader):
    """OpenCV/FFmpeg capture: HTTP MJPEG, RTSP/RTMP/UDP, capture devices and files."""

    def __init__(self, spec: SourceSpec, timeout: float):
        if spec.kind in ("http", "stream"):
            _check_reachable(str(spec.target), timeout)
        ms = int(max(timeout, 1.0) * 1000)
        params = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, ms, cv2.CAP_PROP_READ_TIMEOUT_MSEC, ms]
        if spec.kind == "device":
            api = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            self.cap = cv2.VideoCapture(int(spec.target), api)
        elif spec.kind == "file":
            if not Path(str(spec.target)).exists():
                raise FileNotFoundError(f"Video file not found: {spec.target}")
            self.cap = cv2.VideoCapture(str(spec.target))
        else:
            self.cap = cv2.VideoCapture(str(spec.target), cv2.CAP_FFMPEG, params)
        if not self.cap.isOpened():
            raise ConnectionError(f"Could not open {spec.label}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.paced = spec.kind == "file"
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.source_fps = fps if 1 <= fps <= 240 else 25.0

    def read(self):
        return self.cap.read()

    def close(self):
        self.cap.release()


class _TelloReader(_CaptureReader):
    """Tello: wake the SDK and turn the video on, then read its UDP H.264 stream."""

    def __init__(self, spec: SourceSpec, timeout: float):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.settimeout(timeout)
        for cmd in (b"command", b"streamon"):
            self.sock.sendto(cmd, (TELLO_IP, TELLO_CMD_PORT))
            try:
                self.sock.recvfrom(64)
            except OSError:
                pass
        self._last_ping = time.time()
        super().__init__(SourceSpec("stream", spec.target, spec.label), timeout)

    def read(self):
        if time.time() - self._last_ping > 8:  # keep the SDK session alive
            self.sock.sendto(b"command", (TELLO_IP, TELLO_CMD_PORT))
            self._last_ping = time.time()
        return super().read()

    def close(self):
        super().close()
        self.sock.close()


class _ScreenReader(_Reader):
    """Grab a monitor, or the content area of the drone vendor's app window."""

    def __init__(self, spec: SourceSpec, timeout: float, max_fps: float = 30.0):
        try:
            import mss
        except ImportError as e:
            raise RuntimeError("Window/screen capture needs the 'mss' package: pip install mss") from e
        _make_dpi_aware()
        self.sct = mss.mss()
        self.spec, self.hwnd, self._min_dt, self._t = spec, None, 1.0 / max_fps, 0.0
        if spec.kind == "window":
            if sys.platform != "win32":
                raise RuntimeError("Window capture is only supported on Windows; use 'screen' instead")
            self.hwnd = _find_window(str(spec.target))
            if not self.hwnd:
                raise ConnectionError(f"No open window with '{spec.target}' in its title. "
                                      "Open the drone's app first (not minimised).")
        else:
            mons = self.sct.monitors
            idx = int(spec.target)
            if not 1 <= idx < len(mons):
                raise ValueError(f"Screen {idx} not found (this PC has {len(mons) - 1})")
            self.monitor = mons[idx]

    def read(self):
        wait = self._min_dt - (time.time() - self._t)
        if wait > 0:
            time.sleep(wait)
        self._t = time.time()
        if self.hwnd is not None:
            rect = _client_rect(self.hwnd)
            if rect is None:  # closed or minimised
                return False, None
            left, top, w, h = rect
            region = {"left": left, "top": top, "width": w, "height": h}
        else:
            region = self.monitor
        img = np.asarray(self.sct.grab(region))
        return True, cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def close(self):
        self.sct.close()


def open_reader(spec: SourceSpec, timeout: float = 3.0) -> _Reader:
    if spec.kind in ("window", "screen"):
        return _ScreenReader(spec, timeout)
    if spec.kind == "tello":
        return _TelloReader(spec, timeout)
    return _CaptureReader(spec, timeout)


# ---------------------------------------------------------------------------- link
CAMERA_BUSY = ("The camera is sending only black frames. It is probably being used by another program "
               "(the drone's app.py, Camera, Zoom or Teams) or its lens/privacy shutter is covered. "
               "Close the other program; the video comes back by itself.")


def is_blank(img: np.ndarray, level: float = 4.0) -> bool:
    """True for an (almost) fully black frame: what a busy or covered webcam delivers."""
    return float(img[::8, ::8].mean()) < level


class Frame(NamedTuple):
    image: np.ndarray
    index: int        # increases by 1 for every frame received
    timestamp: float  # time.time() when it arrived


@dataclass
class LinkStats:
    status: str = "idle"        # idle | connecting | live | reconnecting | ended | failed | stopped
    source: str = ""
    kind: str = ""
    resolution: str = "-"
    fps_in: float = 0.0
    frames: int = 0
    reconnects: int = 0
    last_error: str = ""
    connected_since: Optional[float] = None
    age_ms: Optional[float] = field(default=None)


class DroneLink:
    """Background reader that always holds the drone's newest frame.

    Args:
        source: any string accepted by ``parse_source``.
        connect_timeout: seconds to wait when opening the source.
        stall_timeout: seconds without a frame before a live link is reopened.
        max_reconnects: give up after this many reconnects in a row (None = keep trying).
    """

    def __init__(self, source: Union[str, int], connect_timeout: float = 3.0,
                 stall_timeout: float = 4.0, max_reconnects: Optional[int] = None):
        self.spec = parse_source(source)
        self.connect_timeout, self.stall_timeout = connect_timeout, stall_timeout
        self.max_reconnects = max_reconnects
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._frame: Optional[Frame] = None
        self._stats = LinkStats(source=str(self.spec.target), kind=self.spec.kind)
        self._thread: Optional[threading.Thread] = None

    # -- public API
    def start(self) -> "DroneLink":
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="DroneLink", daemon=True)
            self._thread.start()
        return self

    def stop(self, join: float = 2.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=join)
        if self._stats.status not in ("ended", "failed"):
            self._set(status="stopped")

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._frame

    def stats(self) -> LinkStats:
        with self._lock:
            s = LinkStats(**self._stats.__dict__)
            if self._frame is not None:
                s.age_ms = (time.time() - self._frame.timestamp) * 1000
            return s

    @property
    def finished(self) -> bool:
        return self._stats.status in ("ended", "failed", "stopped")

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    # -- internals
    def _set(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self._stats, k, v)

    def _run(self) -> None:
        failures = 0
        while not self._stop.is_set():
            self._set(status="connecting" if self._stats.frames == 0 else "reconnecting")
            got_frames = False
            try:
                reader = open_reader(self.spec, self.connect_timeout)
            except Exception as e:  # unreachable, missing window, bad file ...
                self._set(last_error=str(e))
                if not self.spec.is_live:
                    self._set(status="failed")
                    return
            else:
                try:
                    reason, got_frames = self._pump(reader)
                finally:
                    reader.close()
                if reason == "stopped":
                    return
                if reason == "ended":
                    self._set(status="ended")
                    return
            failures = 0 if got_frames else failures + 1
            if self.max_reconnects is not None and failures > self.max_reconnects:
                self._set(status="failed")
                return
            with self._lock:
                self._stats.reconnects += 1
            self._stop.wait(min(0.5 * (failures + 1), 3.0))  # gentle back-off

    def _pump(self, reader: _Reader):
        """Read frames until stopped, stalled or (for files) finished."""
        last_ok, got, blank = time.time(), False, False
        frame_dt = 1.0 / getattr(reader, "source_fps", 25.0)
        while not self._stop.is_set():
            t0 = time.time()
            ok, img = reader.read()
            now = time.time()
            if ok and img is not None and img.size and self.spec.kind == "device" and is_blank(img):
                # Windows hands a busy or covered webcam to a second program as black frames at ~1 fps.
                # Drop them (the AI would "see" people in the noise) and report why after the stall timeout.
                ok, blank = False, True
            if ok and img is not None and img.size:
                if not got:
                    got = True
                    self._set(status="live", last_error="", connected_since=now)
                with self._lock:
                    idx = self._frame.index + 1 if self._frame else 1
                    if self._frame is not None:
                        dt = max(now - self._frame.timestamp, 1e-3)
                        self._stats.fps_in = 0.9 * self._stats.fps_in + 0.1 / dt if self._stats.fps_in else 1 / dt
                    self._frame = Frame(img, idx, now)
                    self._stats.frames = idx
                    self._stats.resolution = f"{img.shape[1]}x{img.shape[0]}"
                last_ok = now
                if reader.paced:
                    self._stop.wait(max(frame_dt - (time.time() - t0), 0))
                continue
            if not self.spec.is_live:
                return "ended", got
            if now - last_ok > self.stall_timeout:
                self._set(last_error=CAMERA_BUSY if blank else
                          f"Signal lost: no video for {self.stall_timeout:.0f}s")
                return "stalled", got
            time.sleep(0.02)
        return "stopped", got


# ---------------------------------------------------------------------------- discovery
# Well-known endpoints of drones that act as their own WiFi hotspot.
KNOWN_ENDPOINTS = [
    ("ESP32 drone (port 80, as in the drone's own app)", "http://192.168.4.1/stream"),
    ("ESP32-CAM (CameraWebServer / our sketch, port 81)", "http://192.168.4.1:81/stream"),
    ("WiFi FPV drone (RTSP, 192.168.1.1)", "rtsp://192.168.1.1:7070/webcam"),
    ("WiFi FPV drone (RTSP, 192.168.0.1)", "rtsp://192.168.0.1:7070/webcam"),
    ("WiFi drone (RTSP, port 554)", "rtsp://192.168.1.1:554/live"),
    ("WiFi drone (MJPEG, 192.168.1.1:8080)", "http://192.168.1.1:8080/?action=stream"),
    ("Local RTMP server (DJI / controller live-stream)", "rtmp://127.0.0.1:1935/live/drone"),
]


def local_ipv4() -> List[str]:
    """This laptop's IPv4 addresses (best effort)."""
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def is_video_stream(url: str, timeout: float = 1.5) -> bool:
    """True if an HTTP URL answers with a video stream (MJPEG multipart or a JPEG/video body).

    The ESP32 serves its camera on one path (e.g. /stream); its web page lives on "/". Only the
    response headers are read and the connection is closed straight away, so the drone is freed
    for the real viewer.
    """
    import http.client
    p = urlparse(url)
    conn_cls = http.client.HTTPSConnection if p.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(p.hostname, p.port or DEFAULT_PORTS.get(p.scheme, 80), timeout=timeout)
    try:
        conn.request("GET", (p.path or "/") + (f"?{p.query}" if p.query else ""))
        ctype = (conn.getresponse().getheader("Content-Type") or "").lower()
        return any(k in ctype for k in ("multipart", "image/", "video/"))
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


def on_drone_wifi() -> Optional[str]:
    """If this laptop is on a typical drone hotspot subnet, return a short description of it."""
    for ip in local_ipv4():
        if ip.startswith("192.168.4."):
            return f"ESP32 hotspot (laptop IP {ip})"
        if ip.startswith("192.168.10."):
            return f"Tello hotspot (laptop IP {ip})"
    return None


def discover(timeout: float = 0.8) -> List[dict]:
    """Probe well-known drone endpoints in parallel.

    HTTP endpoints only count as found when they actually stream video; RTSP/RTMP when the port
    answers. Returns a list of ``{"name", "source", "found"}`` dicts, found ones first.
    """
    def probe(item):
        name, url = item
        p = urlparse(url)
        port = p.port or DEFAULT_PORTS.get(p.scheme, 80)
        found = tcp_reachable(p.hostname, port, timeout)
        if found and p.scheme in HTTP_SCHEMES:
            found = is_video_stream(url, timeout=max(timeout, 1.5))
        return {"name": name, "source": url, "found": found}

    with ThreadPoolExecutor(max_workers=len(KNOWN_ENDPOINTS)) as pool:
        results = list(pool.map(probe, KNOWN_ENDPOINTS))
    if any(ip.startswith("192.168.10.") for ip in local_ipv4()):
        results.append({"name": "Tello drone (on its WiFi)", "source": "tello", "found": True})
    return sorted(results, key=lambda r: not r["found"])
