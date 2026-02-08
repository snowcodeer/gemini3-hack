import cv2
print(f"OpenCV Version: {cv2.__version__}")
try:
    tracker = cv2.TrackerKCF_create()
    print("KCF Tracker available")
except AttributeError:
    print("KCF Tracker NOT available")

try:
    tracker = cv2.TrackerCSRT_create()
    print("CSRT Tracker available")
except AttributeError:
    # Try legacy or new API
    try:
        tracker = cv2.legacy.TrackerCSRT_create()
        print("Legacy CSRT Tracker available")
    except AttributeError:
        print("CSRT Tracker NOT available")
