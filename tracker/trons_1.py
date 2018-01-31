import json
from VisualTracking import VisTracker

v = {}
with open('base_files/default_vals', 'r') as f:
    v = json.loads(f.read())

#print(v)

tracker = VisTracker()

#tracker.run_full() # for full enclosed process

while tracker.cap.isOpened():
    tracker.process_loop() # each iteration, process_loop returns a list of all detected circles in the current image
    if not tracker.run:
        break
tracker.end_process()