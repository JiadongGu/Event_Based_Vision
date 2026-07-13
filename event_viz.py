import argparse
import numpy as np
import cv2
 
 

BG_COLOR = (40, 40, 40)    
ON_COLOR = (0, 255, 0)    
OFF_COLOR = (0, 0, 255)    
HUD_COLOR = (230, 230, 230) 
 
 
def load_events(path):
    print(f"Loading {path} ...")
    data = np.loadtxt(path)
    if data.ndim == 1:         
        data = data[None, :]
    if data.size == 0:
        raise ValueError("No events found in file.")
 
    t = data[:, 0].astype(np.float64)
    x = data[:, 1].astype(np.int32)
    y = data[:, 2].astype(np.int32)
    p = data[:, 3].astype(np.int32)
 
    order = np.argsort(t, kind="stable")
    t, x, y, p = t[order], x[order], y[order], p[order]
 
    width = int(x.max()) + 1
    height = int(y.max()) + 1
 
    print(f"  {len(t):,} events | {width}x{height} | "
          f"{t[0]:.3f}s -> {t[-1]:.3f}s")
    return t, x, y, p, width, height
 
 
def render_window(x, y, p, lo, hi, width, height):
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = BG_COLOR
 
    xs = x[lo:hi]
    ys = y[lo:hi]
    ps = p[lo:hi]
 
    on = ps == 1
    off = ~on
    frame[ys[on], xs[on]] = ON_COLOR
    frame[ys[off], xs[off]] = OFF_COLOR
    return frame
 
 
def draw_hud(display, t_now, n_events, speed, paused):
    status = "PAUSED" if paused else "PLAY"
    text = f"t={t_now:6.3f}s  events={n_events:5d}  speed={speed:.2f}x  [{status}]"
    cv2.putText(display, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, HUD_COLOR, 1, cv2.LINE_AA)
    cv2.putText(display, "space=pause  +/-=speed  r=restart  q=quit",
                (8, display.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, HUD_COLOR, 1, cv2.LINE_AA)
 
 
def main():
    parser = argparse.ArgumentParser(description="Live event-camera visualization feed.")
    parser.add_argument("--events", default="events.txt", help="input events file")
    parser.add_argument("--window", type=float, default=20.0,
                        help="accumulation window per frame, in milliseconds")
    parser.add_argument("--scale", type=int, default=3,
                        help="integer display upscale factor")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback speed multiplier (1.0 = real time)")
    args = parser.parse_args()
 
    t, x, y, p, width, height = load_events(args.events)
 
    window_s = args.window / 1000.0    
    t_min, t_max = float(t[0]), float(t[-1])
 
    win_name = "Event Camera Feed"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, width * args.scale, height * args.scale)
 
    t_now = t_min
    speed = args.speed
    paused = False
    display = None  
 
    while True:
        if not paused:
            lo = int(np.searchsorted(t, t_now, side="left"))
            hi = int(np.searchsorted(t, t_now + window_s, side="left"))
 
            frame = render_window(x, y, p, lo, hi, width, height)
 
            display = cv2.resize(frame, (width * args.scale, height * args.scale),
                                 interpolation=cv2.INTER_NEAREST)
            draw_hud(display, t_now, hi - lo, speed, paused)
 
            t_now += window_s
            if t_now >= t_max:
                t_now = t_min
        else:
            if display is not None:
                lo = int(np.searchsorted(t, t_now, side="left"))
                hi = int(np.searchsorted(t, t_now + window_s, side="left"))
                frame = render_window(x, y, p, lo, hi, width, height)
                display = cv2.resize(frame, (width * args.scale, height * args.scale),
                                     interpolation=cv2.INTER_NEAREST)
                draw_hud(display, t_now, hi - lo, speed, paused)
 
        if display is not None:
            cv2.imshow(win_name, display)
        delay = max(1, int(window_s * 1000 / max(speed, 1e-6)))
        key = cv2.waitKey(delay) & 0xFF
 
        if key == ord("q") or key == 27:        
            break
        elif key == ord(" "):                   
            paused = not paused
        elif key in (ord("+"), ord("=")):      
            speed = min(speed * 1.5, 64.0)
        elif key == ord("-"):                  
            speed = max(speed / 1.5, 0.05)
        elif key == ord("r"):                 
            t_now = t_min
 
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
 
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
 