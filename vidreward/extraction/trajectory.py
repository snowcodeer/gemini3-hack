from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

@dataclass
class HandTrajectory:
    """
    Data structure for hand landmarks across video frames.
    Landmarks are expected in MediaPipe format: 21 points with (x, y, z).
    """
    # shape: (num_frames, 21, 3)
    landmarks: np.ndarray
    # shape: (num_frames,)
    confidences: np.ndarray
    # Optional metadata per frame
    metadata: List[Dict] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.landmarks, np.ndarray):
            self.landmarks = np.array(self.landmarks)
        if not isinstance(self.confidences, np.ndarray):
            self.confidences = np.array(self.confidences)

@dataclass
class ObjectTrajectory:
    """
    Data structure for object tracking across video frames.
    """
    # shape: (num_frames, 2) normalized centroid (x, y)
    centroids: np.ndarray
    # shape: (num_frames, H, W) binary masks (stored efficiently if possible)
    masks: Optional[np.ndarray] = None
    # shape: (num_frames,)
    confidences: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if not isinstance(self.centroids, np.ndarray):
            self.centroids = np.array(self.centroids)
        if self.masks is not None and not isinstance(self.masks, np.ndarray):
            self.masks = np.array(self.masks)

@dataclass
class VideoTrajectory:
    """
    Full extracted data from a single video clip.
    """
    video_path: str
    hand: Optional[HandTrajectory] = None
    object: Optional[ObjectTrajectory] = None
    fps: float = 0.0
    num_frames: int = 0
