import torch
import os
try:
    import sam2
    print("SAM2 package found")
    from sam2.build_sam import build_sam2_video_predictor
    print("SAM2 Predictor builder found")
except ImportError as e:
    print(f"SAM2 NOT found: {e}")

print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
