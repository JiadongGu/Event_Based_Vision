# Event_Based_Vision
Turns any MP4 into a DVS-style event stream: a list of (x, y, t, polarity).
This is a *naive*, frame-based approximation of the event-generation model
(Eq. 2 from the Gallego et al. survey): each pixel tracks its log-brightness,
and emits an event whenever that brightness changes past a threshold C since
its last event.

python video_to_events.py --video Big_Buck_Bunny_360_10s_30MB.mp4 --out events.txt --resize 240x180