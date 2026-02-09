"""
Detect the release position from a demonstration video.

Extracts where the object ends up after being released, to use as the target
position for RL training.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import cv2


@dataclass
class ReleaseInfo:
    """Information about where and when the object is released."""
    # Release frame
    release_frame: int

    # Object position at release (normalized 0-1 in video space)
    release_pos_video: np.ndarray  # (x, y) normalized

    # Object position at release (3D sim coordinates)
    release_pos_sim: np.ndarray  # (x, y, z) in meters

    # Object start position (for reference)
    start_pos_video: np.ndarray
    start_pos_sim: np.ndarray

    # Hand position at release
    hand_pos_at_release: np.ndarray  # (x, y, z) normalized

    # Grasp frame (for reference)
    grasp_frame: int


def detect_release_position(
    video_path: str,
    mirror: bool = False,
    table_height: float = 0.0,
    release_height: float = 0.15,
) -> ReleaseInfo:
    """
    Detect where the object is released in a demonstration video.

    Args:
        video_path: Path to demonstration video
        mirror: Whether to mirror video (left hand -> right hand)
        table_height: Height of table in sim (z=0 typically)
        release_height: Expected release height above table

    Returns:
        ReleaseInfo with positions in both video and sim coordinates
    """
    from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
    from vidreward.utils.video_io import VideoReader
    from vidreward.extraction.vision import detect_rubiks_cube_classical

    # Load video
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())
    num_frames = len(frames)

    if mirror:
        frames = [cv2.flip(f, 1) for f in frames]

    # Track object throughout video
    obj_positions = []  # (frame, x, y) normalized
    for i, frame in enumerate(frames):
        result = detect_rubiks_cube_classical(frame)
        if result is not None:
            x, y, w, h = result
            fh, fw = frame.shape[:2]
            cx = (x + w/2) / fw
            cy = (y + h/2) / fh
            obj_positions.append((i, cx, cy))

    # Extract hand trajectory
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)
    tracker.close()

    # Find key positions
    if not obj_positions:
        raise ValueError("Could not detect object in video")

    # Start position (first detection)
    start_frame, start_x, start_y = obj_positions[0]
    start_pos_video = np.array([start_x, start_y])

    # Release position detection
    # Method: Find where object velocity drops to near zero in the late part of video
    release_frame, release_x, release_y = _find_release_frame(obj_positions, num_frames)
    release_pos_video = np.array([release_x, release_y])

    # Grasp frame detection (where fingertips converge)
    grasp_frame = _find_grasp_frame(hand_traj.landmarks)

    # Hand position at release
    if release_frame < len(hand_traj.landmarks):
        wrist = hand_traj.landmarks[release_frame, 0, :]
        hand_pos_at_release = wrist.copy()
    else:
        hand_pos_at_release = hand_traj.landmarks[-1, 0, :].copy()

    # Convert to sim coordinates
    start_pos_sim = video_to_sim_coords(start_pos_video, z=table_height + 0.035)
    release_pos_sim = video_to_sim_coords(release_pos_video, z=release_height)

    return ReleaseInfo(
        release_frame=release_frame,
        release_pos_video=release_pos_video,
        release_pos_sim=release_pos_sim,
        start_pos_video=start_pos_video,
        start_pos_sim=start_pos_sim,
        hand_pos_at_release=hand_pos_at_release,
        grasp_frame=grasp_frame,
    )


def _find_release_frame(
    obj_positions: List[Tuple[int, float, float]],
    num_frames: int
) -> Tuple[int, float, float]:
    """
    Find the frame where object is released (velocity drops, position stabilizes).
    """
    if len(obj_positions) < 10:
        # Not enough data, use last position
        return obj_positions[-1]

    # Compute velocities
    positions = np.array([(x, y) for _, x, y in obj_positions])
    frames = np.array([f for f, _, _ in obj_positions])

    velocities = np.diff(positions, axis=0)
    speeds = np.linalg.norm(velocities, axis=1)

    # Smooth speeds
    window = min(5, len(speeds) // 4)
    if window > 1:
        kernel = np.ones(window) / window
        speeds_smooth = np.convolve(speeds, kernel, mode='valid')
    else:
        speeds_smooth = speeds

    # Find where speed drops in the latter half of video
    mid_point = len(speeds_smooth) // 2
    late_speeds = speeds_smooth[mid_point:]

    if len(late_speeds) > 0:
        # Find first frame where speed is below threshold (object stopped)
        threshold = np.mean(late_speeds) * 0.3
        stopped_indices = np.where(late_speeds < threshold)[0]

        if len(stopped_indices) > 0:
            # First stop in late video
            release_idx = mid_point + stopped_indices[0] + window
        else:
            # Use last position
            release_idx = len(obj_positions) - 1
    else:
        release_idx = len(obj_positions) - 1

    release_idx = min(release_idx, len(obj_positions) - 1)
    return obj_positions[release_idx]


def _find_grasp_frame(landmarks: np.ndarray) -> int:
    """
    Find the grasp frame using fingertip convergence.
    """
    THUMB_TIP = 4
    INDEX_TIP = 8
    MIDDLE_TIP = 12

    num_frames = len(landmarks)
    convergence = []

    for i in range(num_frames):
        thumb = landmarks[i, THUMB_TIP, :2]
        index = landmarks[i, INDEX_TIP, :2]
        middle = landmarks[i, MIDDLE_TIP, :2]

        dist = (np.linalg.norm(thumb - index) + np.linalg.norm(thumb - middle)) / 2
        convergence.append(dist)

    # Smooth and find minimum
    convergence = np.array(convergence)
    window = min(5, num_frames // 10)
    if window > 1:
        kernel = np.ones(window) / window
        convergence_smooth = np.convolve(convergence, kernel, mode='valid')
        grasp_frame = np.argmin(convergence_smooth) + window // 2
    else:
        grasp_frame = np.argmin(convergence)

    return int(grasp_frame)


def video_to_sim_coords(
    pos_video: np.ndarray,
    z: float = 0.15,
    workspace_scale: float = 0.4,
    workspace_center: Tuple[float, float] = (0.0, 0.1),
) -> np.ndarray:
    """
    Convert normalized video coordinates to simulation coordinates.

    Video coordinate system:
        - x: 0 (left) to 1 (right)
        - y: 0 (top) to 1 (bottom)

    Sim coordinate system (Adroit):
        - x: left/right (negative = left of hand, positive = right)
        - y: forward/backward (negative = toward hand, positive = away)
        - z: up/down (positive = up)

    Args:
        pos_video: (x, y) normalized position in video
        z: Height in simulation
        workspace_scale: Scale factor for mapping video to sim workspace
        workspace_center: Center of workspace in sim (x, y)

    Returns:
        (x, y, z) position in simulation coordinates
    """
    # Map video x (0-1) to sim x
    # Video left (0) -> sim negative x
    # Video right (1) -> sim positive x
    sim_x = (pos_video[0] - 0.5) * workspace_scale + workspace_center[0]

    # Map video y (0-1, top to bottom) to sim y
    # Video top (0) -> sim farther (positive y)
    # Video bottom (1) -> sim closer (negative y)
    sim_y = (0.5 - pos_video[1]) * workspace_scale + workspace_center[1]

    return np.array([sim_x, sim_y, z])


def sim_to_video_coords(
    pos_sim: np.ndarray,
    workspace_scale: float = 0.4,
    workspace_center: Tuple[float, float] = (0.0, 0.1),
) -> np.ndarray:
    """
    Convert simulation coordinates to normalized video coordinates.
    Inverse of video_to_sim_coords.
    """
    video_x = (pos_sim[0] - workspace_center[0]) / workspace_scale + 0.5
    video_y = 0.5 - (pos_sim[1] - workspace_center[1]) / workspace_scale

    return np.array([video_x, video_y])


if __name__ == "__main__":
    # Test with pick-3.mp4
    import sys

    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/pick-3.mp4"

    print(f"Detecting release position from: {video_path}")
    print("=" * 50)

    info = detect_release_position(video_path)

    print(f"\nGrasp frame: {info.grasp_frame}")
    print(f"Release frame: {info.release_frame}")

    print(f"\nObject Start Position:")
    print(f"  Video (normalized): x={info.start_pos_video[0]:.3f}, y={info.start_pos_video[1]:.3f}")
    print(f"  Sim (meters):       x={info.start_pos_sim[0]:.3f}, y={info.start_pos_sim[1]:.3f}, z={info.start_pos_sim[2]:.3f}")

    print(f"\nObject Release Position (TARGET):")
    print(f"  Video (normalized): x={info.release_pos_video[0]:.3f}, y={info.release_pos_video[1]:.3f}")
    print(f"  Sim (meters):       x={info.release_pos_sim[0]:.3f}, y={info.release_pos_sim[1]:.3f}, z={info.release_pos_sim[2]:.3f}")

    print(f"\nHand at release:")
    print(f"  x={info.hand_pos_at_release[0]:.3f}, y={info.hand_pos_at_release[1]:.3f}, z={info.hand_pos_at_release[2]:.3f}")
