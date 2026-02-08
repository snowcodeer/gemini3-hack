import gymnasium as gym
import gymnasium_robotics
import numpy as np
import cv2
import argparse
import os
import mujoco

from vidreward.extraction.mediapipe_tracker import MediaPipeTracker
from vidreward.retargeting.landmarks_to_angles import AdroitRetargeter
from vidreward.extraction.vision import detect_rubiks_cube_classical
from vidreward.utils.video_io import VideoReader, VideoWriter
from vidreward.extraction.trajectory import ObjectTrajectory
from vidreward.phases.phase_detector import PhaseDetector

def replay_m1(video_path: str, output_path: str = "adroit_replay.mp4"):
    print(f"Processing video: {video_path}")

    # Mirror video for left hand -> right hand conversion
    # Set to False for right-handed videos
    FLIP_VIDEO = False

    # 1. Load video frames first
    reader = VideoReader(video_path)
    frames = list(reader.read_frames())

    if FLIP_VIDEO:
        frames = [cv2.flip(f, 1) for f in frames]  # 1 = horizontal flip
        print("Video mirrored horizontally (left hand -> right hand)")

    # 2. Extract Hand Trajectory from (possibly mirrored) frames
    tracker = MediaPipeTracker()
    hand_traj = tracker.process_frames(frames)  # Track on mirrored frames
    tracker.close()

    h_vid, w_vid = frames[0].shape[:2]
    num_frames = len(frames)
    
    obj_pos_sim = np.zeros((num_frames, 3))
    obj_pos_vid_norm = np.zeros((num_frames, 2))  # Object position in normalized video space [0,1]
    
    # Basic tracking
    bbox = None
    for i in range(min(50, num_frames)):
        bbox = detect_rubiks_cube_classical(frames[i])
        if bbox: break
        
    if bbox is None:
        print("Warning: Object not detected clearly. Using center.")
        curr_box = (w_vid//2 - 40, h_vid//2 - 40, 80, 80)
    else:
        curr_box = bbox

    # Sim Workspace approx: X[-0.3, 0.3], Y[-0.3, 0.3], Z[0.035] (table)
    # Video Coords: X[0, w], Y[0, h]
    # Simple mapping: Center video -> Center table
    
    sim_tracker = cv2.TrackerCSRT_create()
    sim_tracker.init(frames[0], curr_box)
    
    for i in range(num_frames):
        success, box = sim_tracker.update(frames[i])
        if success:
            curr_box = box
        else:
             # Try re-detect
             det = detect_rubiks_cube_classical(frames[i])
             if det:
                 curr_box = det
                 sim_tracker = cv2.TrackerCSRT_create()
                 sim_tracker.init(frames[i], curr_box)
        
        bx, by, bw, bh = curr_box
        cx, cy = bx + bw/2, by + bh/2

        # Store normalized video position [0, 1] for delta computation
        obj_pos_vid_norm[i] = [cx / w_vid, cy / h_vid]

        # Normalize to [-0.5, 0.5] for sim mapping
        nx = (cx / w_vid) - 0.5
        ny = (cy / h_vid) - 0.5

        # Map to Sim coordinates
        SCALE = 0.8
        obj_pos_sim[i] = [nx * SCALE, -ny * SCALE + 0.1, 0.035]  # Z fixed on table

    # Detect GRASP phase to calibrate object position
    print("Detecting grasp phase...")
    obj_traj = ObjectTrajectory(centroids=obj_pos_vid_norm)
    phase_detector = PhaseDetector()
    phases = phase_detector.detect_phases(hand_traj, obj_traj)

    # Print all detected phases
    for phase in phases:
        print(f"  Phase: {phase.label} [{phase.start_frame}-{phase.end_frame}]")

    # Find first GRASP frame
    grasp_frame = 0
    for phase in phases:
        if phase.label == "GRASP":
            grasp_frame = phase.start_frame
            break

    print(f"First grasp at frame: {grasp_frame}")

    # Use MIDDLE FINGER MCP (landmark 9) for grasp positioning - closer to actual contact point
    GRASP_LANDMARK = 9  # Middle finger MCP
    grasp_hand_vid = hand_traj.landmarks[grasp_frame, GRASP_LANDMARK].copy()
    print(f"Hand position at grasp (middle finger MCP): {grasp_hand_vid}")

    # Use GRASP frame as reference for hand tracking
    init_hand_vid = grasp_hand_vid.copy()
    print(f"Using grasp frame as hand reference: {init_hand_vid}")

    # 3. Retarget Hand
    print("Retargeting hand...")
    retargeter = AdroitRetargeter()
    joint_traj = retargeter.retarget_sequence(hand_traj.landmarks)
    
    # 4. Simulation Loop
    env = gym.make("AdroitHandRelocate-v1", render_mode="rgb_array")
    env.reset()
    
    # Get qpos indices
    model = env.unwrapped.model
    # Assumed structure: [30 hand, 7 object, 7 target]
    # Check joint names
    jnames = [model.joint(i).name for i in range(model.njnt)]
    # 'hand' joints usually start at 0. 'OBJTx' etc for object.
    
    # Calibration & Debug
    print("-" * 30)
    print("Calibration Data:")
    print(f"Default Reset QPos [0:3]: {env.unwrapped.data.qpos[0:3]}")
    
    # Calibration & Debug
    print("-" * 30)
    print("Calibration Data:")
    
    # Analyze computed ranges
    min_pos = joint_traj[:, 0:3].min(axis=0)
    max_pos = joint_traj[:, 0:3].max(axis=0)
    print(f"Computed Hand Pos Range: Min={min_pos}, Max={max_pos}")
    
    min_obj = obj_pos_sim.min(axis=0)
    max_obj = obj_pos_sim.max(axis=0)
    print(f"Computed Obj Pos Range: Min={min_obj}, Max={max_obj}")
    print("-" * 30)
    
    # Force a known good state for first frame to check camera
    # Reset puts hand above table
    env.reset()

    # Get model and data references
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Get ACTUAL object position from body xpos (not qpos which is joint-relative)
    obj_body_id = model.body("Object").id
    default_obj_pos = data.xpos[obj_body_id].copy()
    print(f"Default object pos (from xpos): {default_obj_pos}")

    # Get S_grasp site position - this is the hand's grasp point
    grasp_site_id = model.site("S_grasp").id
    grasp_site_pos = data.site_xpos[grasp_site_id].copy()
    print(f"S_grasp site pos: {grasp_site_pos}")

    palm_body_id = model.body("palm").id

    # Debug: Print hand geometry info
    print("\n--- HAND GEOMETRY DEBUG ---")

    # Print body names and positions
    print("Bodies:")
    for i in range(model.nbody):
        name = model.body(i).name
        pos = data.xpos[i]
        print(f"  {i}: {name} -> pos: {pos}")

    # Print site names if available
    print("\nSites:")
    for i in range(model.nsite):
        name = model.site(i).name
        pos = data.site_xpos[i]
        print(f"  {i}: {name} -> pos: {pos}")
    print("--- END DEBUG ---\n")

    # Object stays at default position
    obj_init_pos = default_obj_pos.copy()
    print(f"Object stays at: {obj_init_pos}")

    # Compute fingertip offset with grasp finger pose
    # Set root to origin, apply grasp frame finger angles, measure fingertip position
    data.qpos[:] = 0
    data.qpos[3:30] = joint_traj[grasp_frame, 3:]  # Grasp finger pose
    mujoco.mj_forward(model, data)
    mftip_id = model.site("S_mftip").id
    mftip_at_origin = data.site_xpos[mftip_id].copy()
    print(f"Fingertip when root at origin: {mftip_at_origin}")
    fingertip_offset = mftip_at_origin  # Offset from root to fingertip

    # Reset and recapture object position (reset changes it)
    env.reset()
    mujoco.mj_forward(model, data)
    obj_init_pos = data.xpos[obj_body_id].copy()
    print(f"Object after reset: {obj_init_pos}")

    # CALIBRATE: Find qpos that positions palm to grasp the object
    # Offset palm so fingers wrap around cube (not palm center at cube center)
    GRASP_OFFSET = np.array([0.0, -0.06, 0.05])  # Back and up from object center
    target_pos = obj_init_pos + GRASP_OFFSET
    print(f"Grasp target (with offset): {target_pos}")
    qpos_guess = np.array([0.0, 0.0, 0.0])

    for iteration in range(10):  # Iterative refinement
        data.qpos[0:3] = qpos_guess
        data.qpos[3:30] = joint_traj[grasp_frame, 3:]  # Use grasp finger pose
        mujoco.mj_forward(model, data)

        actual_palm = data.xpos[palm_body_id].copy()
        error = target_pos - actual_palm

        # Update qpos based on error (using known axis mapping)
        # palm_x = -qpos[0], palm_y = qpos[2] + offset, palm_z = qpos[1] + offset
        qpos_guess[0] -= error[0]  # X: qpos[0] moves palm in -X
        qpos_guess[2] += error[1]  # Y: qpos[2] moves palm in +Y
        qpos_guess[1] += error[2]  # Z: qpos[1] moves palm in +Z

        if np.linalg.norm(error) < 0.001:  # 1mm tolerance
            break

    grasp_qpos = qpos_guess.copy()
    print(f"Calibrated grasp qpos: {grasp_qpos}")
    print(f"Palm error after calibration: {error} (norm: {np.linalg.norm(error):.6f})")

    frame = env.render()
    if frame is None or np.mean(frame) < 1:
        print("CRITICAL: Default render is black! Camera issue?")
        cv2.imwrite("debug_black_frame.png", np.zeros((100,100)))
    else:
        cv2.imwrite("debug_reset_frame.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print("Default render OK. Saved to debug_reset_frame.png")

    print(f"Rendering side-by-side replay to {output_path}...")
    height, width = 480, 480

    # Resize video frames to match sim
    vid_frames_resized = [cv2.resize(f, (width, height)) for f in frames]

    GRIP_GAIN = 1.5  # Amplify finger closing for better grasping

    # Fix NaN values in joint trajectory (from divide-by-zero in retargeting)
    joint_traj = np.nan_to_num(joint_traj, nan=0.0, posinf=1.0, neginf=-1.0)
    # Clamp finger angles to reasonable range
    joint_traj[:, 3:] = np.clip(joint_traj[:, 3:], -1.5, 1.5)

    with VideoWriter(output_path, 30.0, width * 2, height) as writer:
        for i in range(num_frames):
            # Compute hand movement from grasp position (in video space)
            hand_pos = hand_traj.landmarks[i, GRASP_LANDMARK]
            hand_delta_x = hand_pos[0] - init_hand_vid[0]
            hand_delta_y = hand_pos[1] - init_hand_vid[1]
            hand_delta_z = hand_pos[2] - init_hand_vid[2]

            # Scale video movement to sim space
            SCALE_X = 0.5  # Video X -> Sim X
            SCALE_Y = 0.3  # Video Y -> Sim Z (height)
            SCALE_Z = 0.3  # Video depth -> Sim Y (forward/back)

            # Apply delta to grasp_qpos (using same axis mapping as calibration)
            sim_qpos = grasp_qpos.copy()
            sim_qpos[0] -= hand_delta_x * SCALE_X  # X movement
            sim_qpos[2] -= hand_delta_y * SCALE_Y  # Y in video -> height in sim
            sim_qpos[1] += hand_delta_z * SCALE_Z  # Depth in video -> forward in sim

            sim_qpos[1] = np.clip(sim_qpos[1], -0.5, 0.5)
            sim_qpos[2] = np.clip(sim_qpos[2], -0.5, 0.8)

            joint_traj[i, 0:3] = sim_qpos

            # Apply GRIP_GAIN to finger angles
            joint_traj[i, 3:] *= GRIP_GAIN

            # Hybrid mode: kinematic before grasp, physics after
            if i < grasp_frame:
                # Kinematic mode - just compute positions, no physics
                env.unwrapped.data.qpos[0:30] = joint_traj[i]
                mujoco.mj_forward(env.unwrapped.model, env.unwrapped.data)
            else:
                # Physics mode - use smooth position control
                # Blend towards target position instead of snapping
                target_qpos = joint_traj[i]
                current_qpos = env.unwrapped.data.qpos[0:30].copy()

                # Smooth interpolation (avoid sudden jumps)
                alpha = 0.3  # Blend factor
                blended_qpos = current_qpos + alpha * (target_qpos - current_qpos)
                env.unwrapped.data.qpos[0:30] = blended_qpos
                env.unwrapped.data.qvel[0:30] = 0

                for _ in range(2):
                    mujoco.mj_step(env.unwrapped.model, env.unwrapped.data)

            # Debug at grasp frame - see where things ACTUALLY end up
            if i == grasp_frame:
                actual_obj = data.xpos[obj_body_id]
                actual_palm = data.xpos[palm_body_id]
                print(f"\n--- GRASP FRAME {i} - ACTUAL POSITIONS ---")
                print(f"Object:    {actual_obj}")
                print(f"Palm:      {actual_palm}")
                print(f"Gap (palm-obj): {actual_palm - actual_obj}")
                print(f"--- END ---\n")

            # Render Sim
            sim_frame = env.render()
            sim_frame = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)

            # Get Video Frame
            vid_frame = vid_frames_resized[i].copy()

            # --- Draw MediaPipe landmarks on video ---
            landmarks = hand_traj.landmarks[i]
            h_frame, w_frame = vid_frame.shape[:2]

            # MediaPipe hand connections
            HAND_CONNECTIONS = [
                (0,1),(1,2),(2,3),(3,4),  # Thumb
                (0,5),(5,6),(6,7),(7,8),  # Index
                (0,9),(9,10),(10,11),(11,12),  # Middle
                (0,13),(13,14),(14,15),(15,16),  # Ring
                (0,17),(17,18),(18,19),(19,20),  # Pinky
                (5,9),(9,13),(13,17)  # Palm
            ]

            # Draw connections
            for conn in HAND_CONNECTIONS:
                pt1 = (int(landmarks[conn[0], 0] * w_frame), int(landmarks[conn[0], 1] * h_frame))
                pt2 = (int(landmarks[conn[1], 0] * w_frame), int(landmarks[conn[1], 1] * h_frame))
                cv2.line(vid_frame, pt1, pt2, (0, 255, 0), 2)

            # Draw landmark points
            for j, lm in enumerate(landmarks):
                x, y = int(lm[0] * w_frame), int(lm[1] * h_frame)
                color = (0, 0, 255) if j == GRASP_LANDMARK else (0, 255, 255)  # Red for tracking point
                cv2.circle(vid_frame, (x, y), 4, color, -1)

            # --- Draw object tracking box ---
            obj_x, obj_y = obj_pos_vid_norm[i]
            obj_cx, obj_cy = int(obj_x * w_frame), int(obj_y * h_frame)
            cv2.rectangle(vid_frame, (obj_cx - 30, obj_cy - 30), (obj_cx + 30, obj_cy + 30), (255, 0, 255), 2)
            cv2.circle(vid_frame, (obj_cx, obj_cy), 5, (255, 0, 255), -1)

            # --- Draw current phase ---
            current_phase = "UNKNOWN"
            for phase in phases:
                if phase.start_frame <= i <= phase.end_frame:
                    current_phase = phase.label
                    break

            # Phase color coding
            phase_colors = {
                "IDLE": (128, 128, 128),
                "APPROACH": (0, 255, 255),
                "GRASP": (0, 255, 0),
                "TRANSPORT": (255, 165, 0),
                "RELEASE": (255, 0, 0),
                "RETREAT": (128, 0, 128),
                "UNKNOWN": (255, 255, 255)
            }
            phase_color = phase_colors.get(current_phase, (255, 255, 255))

            # Draw phase label with background
            cv2.rectangle(vid_frame, (5, 5), (150, 40), (0, 0, 0), -1)
            cv2.putText(vid_frame, current_phase, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

            # Frame number on video
            cv2.putText(vid_frame, f"Frame: {i}", (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Overlay Info on Sim
            cv2.putText(sim_frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(sim_frame, current_phase, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, phase_color, 2)

            # Concatenate
            combined = np.hstack((vid_frame, sim_frame))
            
            writer.write_frame(combined)
            
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/pick-rubiks-cube.mp4")
    parser.add_argument("--out", default="adroit_replay_sbs.mp4") # Side-by-side
    args = parser.parse_args()
    
    replay_m1(args.video, args.out)
