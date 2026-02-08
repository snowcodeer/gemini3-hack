import cv2
import numpy as np
import os

with open("codec_report.txt", "w") as f:
    f.write("Codec Report:\n")

def test_codec(codec, ext=".mp4"):
    filename = f"test_{codec}{ext}"
    print(f"Testing codec: {codec} -> {filename}")
    
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(filename, fourcc, 30.0, (640, 480))
    
    if not writer.isOpened():
        print(f"  [FAIL] Failed to open writer for {codec}")
        with open("codec_report.txt", "a") as f:
            f.write(f"{codec}: FAIL (Open Error)\n")
        if os.path.exists(filename): os.remove(filename)
        return False
        
    # Write 30 frames
    for _ in range(30):
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        writer.write(frame)
        
    writer.release()
    
    if not os.path.exists(filename) or os.path.getsize(filename) == 0:
         print(f"  [FAIL] File empty or not created for {codec}")
         return False

    # Verify playback
    cap = cv2.VideoCapture(filename)
    if not cap.isOpened():
        print(f"  [FAIL] OpenCV could not read back {filename}")
        return False
        
    ret, _ = cap.read()
    with open("codec_report.txt", "a") as f:
        if not ret:
            print(f"  [FAIL] Could not read first frame of {filename}")
            f.write(f"{codec}: FAIL (Read Error)\n")
            return False
            
        print(f"  [PASS] Codec {codec} seems working!")
        f.write(f"{codec}: PASS\n")
        
    cap.release()
    # Cleanup
    try:
        os.remove(filename)
    except:
        pass
    return True

print("Checking available VideoWriter codecs for MP4...")
codecs_to_test = ['mp4v', 'avc1', 'h264', 'hevc']

working = []
for c in codecs_to_test:
    if test_codec(c):
        working.append(c)

print("\nWorking codecs:", working)
