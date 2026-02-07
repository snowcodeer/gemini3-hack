import numpy as np
import torch
from typing import Optional, List, Tuple, Generator
import os

# Assuming segment_anything_2 is installed
# If not, this will need to be adjusted or mocked for feasibility
try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    build_sam2_video_predictor = None

class SAM2Tracker:
    """
    Wrapper for SAM 2.1 to track objects in video.
    """
    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        if build_sam2_video_predictor is not None:
            self.predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=self.device)
        else:
            print("Warning: SAM2 library not found. Running in mock mode.")
            self.predictor = None
        
        self.inference_state = None

    def initialize_video(self, video_path: str):
        """Initialize tracker with a video."""
        if self.predictor is not None:
            # SAM2 video predictor often takes a directory of frames or a video path
            self.inference_state = self.predictor.init_state(video_path=video_path)
        else:
            print("Mock: Initializing video tracking.")

    def add_prompt(self, frame_idx: int, obj_id: int, points: np.ndarray, labels: np.ndarray):
        """
        Add a prompt (point/box) to start tracking.
        points: (N, 2) labels: (N,) 1 for positive, 0 for negative
        """
        if self.predictor is not None:
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points(
                inference_state=self.inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points,
                labels=labels,
            )
            return out_obj_ids, out_mask_logits
        else:
            print(f"Mock: Adding prompt at frame {frame_idx} for object {obj_id}")
            return [obj_id], None

    def propagate(self) -> Generator[Tuple[int, List[int], np.ndarray], None, None]:
        """
        Propagate tracking throughout the video.
        Yields (frame_idx, obj_ids, masks)
        """
        if self.predictor is not None:
            for frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                # Process mask logits to binary masks
                masks = (out_mask_logits > 0.0).cpu().numpy() # (num_objs, 1, H, W)
                yield frame_idx, out_obj_ids, masks
        else:
            print("Mock: Propagating tracking. Returning dummy masks.")
            # Dummy generator
            for i in range(10): # dummy 10 frames
                yield i, [0], np.zeros((1, 1, 480, 640))

    def reset(self):
        if self.predictor is not None and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        self.inference_state = None
