def frame_source(video_path, interpolation_factor=1, resize=None):
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
     
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if resize is not None:
            gray = cv2.resize(gray, resize)
        gray = gray.astype(np.float32) / 255.0
        timestamp = frame_idx / fps

        if interpolation_factor > 1:
            raise NotImplementedError("interpolation not implemented yet")

        yield (timestamp, gray)
        frame_idx += 1

    cap.release()


class EventCamera:

    ON = 1
    OFF = 0

    def __init__(self, height, width, C=0.15, log_eps=1e-3):
        self.height = height
        self.width = width
        self.C = C
        self.log_eps = log_eps
        self.last_log = None

    def _to_log(self, gray):
        return np.log(gray + self.log_eps)

    def init_reference(self, gray):
        self.last_log = self._to_log(gray)

    def generate(self, gray, t):
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

        fired = on_mask | off_mask
        self.last_log[fired] = log_frame[fired]

        return events

def open_event_writer(path):
    return open(path, "w")


def write_events(fh, events):
    for (x, y, t, p) in events:
        fh.write(f"{t:.9f} {x} {y} {p}\n")

def _parse_resize(s):
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