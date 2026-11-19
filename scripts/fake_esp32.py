"""Pretend to be the ESP32 drone camera: serve a video as an MJPEG stream, like the drone does.

Use it to rehearse the whole drone -> laptop -> UI path with no drone and no internet:

    python scripts/fake_esp32.py                      # http://127.0.0.1:8081/stream
    python scripts/fake_esp32.py --port 8081 --video demo_output/test_clip.avi

Then in the UI: Live Drone -> "Drone WiFi stream" -> address http://127.0.0.1:8081/stream -> Start watching.
The stream format matches the ESP32 firmware (multipart/x-mixed-replace, one JPEG per part).
"""
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY = "123456789000000000000987654321"


class Frames:
    """Loops a video file and keeps the newest JPEG, like the camera's frame buffer."""

    def __init__(self, video: Path, fps: float, width: int):
        self.video, self.fps, self.width = video, fps, width
        self.jpeg, self.lock = None, threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            cap = cv2.VideoCapture(str(self.video))
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame.shape[1] > self.width:
                    frame = cv2.resize(frame, (self.width, int(frame.shape[0] * self.width / frame.shape[1])))
                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                with self.lock:
                    self.jpeg = buf.tobytes()
                time.sleep(1 / self.fps)
            cap.release()

    def latest(self):
        with self.lock:
            return self.jpeg


def make_handler(frames: Frames):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?")[0] != "/stream":
                body = b"<h1>Fake ESP32</h1><p>Video is at <a href='/stream'>/stream</a></p>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
            self.end_headers()
            try:
                while True:
                    jpg = frames.latest()
                    if jpg is None:
                        time.sleep(0.05)
                        continue
                    self.wfile.write(f"\r\n--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                                     f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg)
                    time.sleep(1 / frames.fps)
            except (ConnectionError, OSError):
                pass  # viewer disconnected

        def log_message(self, *args):
            pass

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--video", default=str(ROOT / "demo_output" / "test_clip.avi"))
    ap.add_argument("--fps", type=float, default=12)
    ap.add_argument("--width", type=int, default=800, help="max frame width (the ESP32 sends 640-800 px)")
    args = ap.parse_args()
    frames = Frames(Path(args.video), args.fps, args.width)
    server = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(frames))
    print(f"Fake ESP32 streaming {Path(args.video).name} at http://127.0.0.1:{args.port}/stream  (Ctrl+C to stop)")
    server.serve_forever()


if __name__ == "__main__":
    main()
