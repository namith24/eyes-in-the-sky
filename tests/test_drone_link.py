"""Tests for the drone link (live video source layer used by the ground-station UI)."""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from uav_survivor_detection.inference.drone_link import (  # noqa: E402
    DroneLink,
    discover,
    list_windows,
    parse_source,
)


@pytest.mark.parametrize("text,kind,target", [
    ("0", "device", 0),
    (" 2 ", "device", 2),
    ("http://192.168.4.1:81/stream", "http", "http://192.168.4.1:81/stream"),
    ("rtsp://192.168.1.1:7070/webcam", "stream", "rtsp://192.168.1.1:7070/webcam"),
    ("rtmp://127.0.0.1/live/drone", "stream", "rtmp://127.0.0.1/live/drone"),
    ("udp://0.0.0.0:11111", "stream", "udp://0.0.0.0:11111"),
    ("window:DJI Assistant", "window", "DJI Assistant"),
    ("screen", "screen", 1),
    ("screen:2", "screen", 2),
    ("TELLO", "tello", "udp://0.0.0.0:11111"),
    ("flights/day1.mp4", "file", "flights/day1.mp4"),
])
def test_parse_source(text, kind, target):
    spec = parse_source(text)
    assert (spec.kind, spec.target) == (kind, target)
    assert spec.is_live == (kind != "file")


@pytest.mark.parametrize("bad", ["", "   ", "window:"])
def test_parse_source_rejects_empty(bad):
    with pytest.raises(ValueError):
        parse_source(bad)


def _wait(link, until, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not until(link):
        time.sleep(0.05)


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "flight.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 50, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    return path


def test_file_replay_delivers_frames_then_ends(clip):
    link = DroneLink(str(clip)).start()
    _wait(link, lambda lk: lk.finished)
    s = link.stats()
    assert s.status == "ended"
    assert s.resolution == "64x48"
    assert s.frames >= 15
    frame = link.latest()
    assert frame is not None and frame.image.shape == (48, 64, 3) and frame.index == s.frames
    link.stop()


def test_missing_file_fails_cleanly():
    link = DroneLink("does/not/exist.mp4").start()
    _wait(link, lambda lk: lk.finished)
    assert link.stats().status == "failed"
    assert "not found" in link.stats().last_error


def test_unreachable_drone_keeps_retrying_with_helpful_error():
    # 192.0.2.0/24 is reserved for documentation: guaranteed not to answer.
    link = DroneLink("http://192.0.2.1:81/stream", connect_timeout=0.3).start()
    _wait(link, lambda lk: lk.stats().reconnects >= 1, timeout=5)
    s = link.stats()
    link.stop()
    assert s.status in ("connecting", "reconnecting")
    assert "drone's WiFi" in s.last_error
    assert link.latest() is None


def test_discover_returns_known_endpoints():
    results = discover(timeout=0.2)
    assert any(r["source"].startswith("http://192.168.4.1") for r in results)
    assert all({"name", "source", "found"} <= set(r) for r in results)


@pytest.mark.skipif(sys.platform != "win32", reason="window capture is Windows-only")
def test_list_windows_returns_titles():
    titles = list_windows()
    assert isinstance(titles, list) and all(isinstance(t, str) and t for t in titles)
    assert "Program Manager" not in titles


# ---------------------------------------------------------------- fake ESP32 (scripts/fake_esp32.py)
@pytest.fixture
def fake_esp32(clip):
    import importlib.util
    import threading
    from http.server import ThreadingHTTPServer

    spec = importlib.util.spec_from_file_location(
        "fake_esp32", Path(__file__).resolve().parent.parent / "scripts" / "fake_esp32.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.make_handler(mod.Frames(clip, fps=25, width=640)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


def test_is_video_stream_tells_stream_from_web_page(fake_esp32):
    from uav_survivor_detection.inference.drone_link import is_video_stream
    time.sleep(0.5)
    assert is_video_stream(fake_esp32 + "/stream")
    assert not is_video_stream(fake_esp32 + "/")
    assert not is_video_stream("http://192.0.2.1/stream", timeout=0.3)


def test_drone_link_reads_esp32_style_mjpeg(fake_esp32):
    link = DroneLink(fake_esp32 + "/stream").start()
    _wait(link, lambda lk: lk.stats().frames >= 5)
    s = link.stats()
    link.stop()
    assert s.status == "live" and s.resolution == "64x48" and s.frames >= 5


def test_blank_frames_from_a_busy_webcam_are_detected():
    from uav_survivor_detection.inference.drone_link import is_blank
    assert is_blank(np.zeros((480, 640, 3), np.uint8))                      # busy / covered camera
    assert is_blank(np.random.randint(0, 6, (480, 640, 3), np.uint8))       # black + sensor noise
    assert not is_blank(np.full((480, 640, 3), 40, np.uint8))               # dim but real room
