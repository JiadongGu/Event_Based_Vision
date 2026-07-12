#!/usr/bin/env python3
"""
video_to_events.py  --  FILE 1: the sensor model.

Naive frame-based event-generation model (Eq. 2): each pixel tracks its
log-brightness and emits an event whenever that change crosses a threshold C.

HOW TO RUN
    python video_to_events.py --video my_clip.mp4 --out events.txt --C 0.15
    python video_to_events.py --video my_clip.mp4 --out events.txt --resize 240x180

Output: events.txt, one event per line as:  t x y p
(t seconds; p is 1 for ON/brighter, 0 for OFF/darker). Matches UZH format.
"""

import argparse
import cv2
import numpy as np


# =====================================================================
# 1. FRAME SOURCE  (raw frames now; interpolation hook present but inert)
# =====================================================================
def frame_source(video_path, interpolation_factor=1, resize=None):
    """
    Generator. Yields (timestamp_seconds, gray_frame_float) per frame.
    gray_frame_float: 2D numpy array (H x W), normalized to [0, 1].
    resize: optional (width, height); None = native size.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        # ---- preprocess this raw frame ----
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if resize is not None:
            gray = cv2.resize(gray, resize)
        gray = gray.astype(np.float32) / 255.0
        timestamp = frame_idx / fps

        # ---- interpolation hook (inert for now) ----
        if interpolation_factor > 1:
            raise NotImplementedError("interpolation not implemented yet")

        # ---- emit the real frame, then advance ----
        yield (timestamp, gray)
        frame_idx += 1

    cap.release()


# =====================================================================
# 2. EVENT GENERATOR  (the actual sensor model -- Eq. 2)
# =====================================================================
class EventCamera:
    """Holds per-pixel log-brightness memory; turns frames into events."""

    ON = 1
    OFF = 0

    def __init__(self, height, width, C=0.15, log_eps=1e-3):
        self.height = height
        self.width = width
        self.C = C
        self.log_eps = log_eps
        self.last_log = None

    def _to_log(self, gray):
        """Map a [0,1] grayscale frame to log intensity."""
        return np.log(gray + self.log_eps)

    def init_reference(self, gray):
        """Seed the reference from the first frame. Emits no events."""
        self.last_log = self._to_log(gray)

    def generate(self, gray, t):
        """
        Compare this frame to the stored reference, emit events, update reference.
        Returns a list of (x, y, t, polarity) tuples (may be empty).
        """
        log_frame = self._to_log(gray)
        diff = log_frame - self.last_log

        on_mask = diff >= self.C
        off_mask = diff <= -self.C

        on_ys, on_xs = np.where(on_mask)
        off_ys, off_xs = np.where(off_mask)

        events = []
        for x, y in zip(on_xs, on_ys):
            events.append((int(x), int(y), t, self.ON))
        for x, y in zip(off_xs, off_ys):
            events.append((int(x), int(y), t, self.OFF))

        # reset reference at fired pixels ONLY
        fired = on_mask | off_mask
        self.last_log[fired] = log_frame[fired]

        # --- ENHANCEMENTS (leave for later) ---
        # multi-event per pixel: k = floor(|diff|/C); reset with remainder
        # timestamp spreading across the inter-frame interval

        return events


# =====================================================================
# 3. WRITER  (dataset-compatible events.txt)
# =====================================================================
def open_event_writer(path):
    """Open the output file for writing and return the handle."""
    return open(path, "w")


def write_events(fh, events):
    """Append events, one per line: 't x y p'  (time first!)."""
    for (x, y, t, p) in events:
        fh.write(f"{t:.9f} {x} {y} {p}\n")


# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================
def _parse_resize(s):
    """Parse '240x180' -> (240, 180); None stays None."""
    if s is None:
        return None
    w, h = s.lower().split("x")
    return (int(w), int(h))


def main():
    parser = argparse.ArgumentParser(description="Convert an MP4 to a DVS event stream.")
    parser.add_argument("--video", required=True, help="input .mp4")
    parser.add_argument("--out", default="events.txt", help="output events file")
    parser.add_argument("--C", type=float, default=0.15, help="contrast threshold (log units)")
    parser.add_argument("--resize", default=None, help="e.g. 240x180; default native")
    args = parser.parse_args()

    resize = _parse_resize(args.resize)

    writer = open_event_writer(args.out)
    cam = None
    total_events = 0
    frames_processed = 0
    t = 0.0

    for i, (t, gray) in enumerate(frame_source(args.video, resize=resize)):
        if i == 0:
            H, W = gray.shape
            cam = EventCamera(H, W, C=args.C)
            cam.init_reference(gray)
            frames_processed += 1
            continue

        events = cam.generate(gray, t)
        write_events(writer, events)
        total_events += len(events)
        frames_processed += 1

    writer.close()

    eps = total_events / t if (frames_processed > 1 and t > 0) else 0.0
    print(f"Frames processed : {frames_processed}")
    print(f"Total events     : {total_events}")
    print(f"Events / second  : {eps:,.0f}")
    print(f"Written to       : {args.out}")


if __name__ == "__main__":
    main()