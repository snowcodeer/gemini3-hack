import numpy as np
from typing import Dict

def unit_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'."""
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class AdroitRetargeter:
    """
    Retargets MediaPipe hand landmarks to Adroit/ShadowHand joint angles.
    """
    def __init__(self):
        # Joint limits for Adroit (representative values, should be verified with MJCF)
        self.joint_limits = {
            "FFJ3": (-0.349, 0.349), # Index MCP abduction
            "FFJ2": (0, 1.57),      # Index MCP flexion
            "FFJ1": (0, 1.57),      # Index PIP flexion
            "FFJ0": (0, 1.57),      # Index DIP flexion
            # ... and so on for other fingers
        }

    def compute_angles(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Compute Adroit-compatible joint angles from 21 MediaPipe landmarks.
        landmarks: (21, 3)
        """
        # MediaPipe Indices
        INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
        MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
        
        angles = {}
        
        # 1. Index Finger
        v_mcp_pip = landmarks[INDEX_PIP] - landmarks[INDEX_MCP]
        v_pip_dip = landmarks[INDEX_DIP] - landmarks[INDEX_PIP]
        v_dip_tip = landmarks[INDEX_TIP] - landmarks[INDEX_DIP]
        
        # PIP Flexion
        angles["FFJ1"] = angle_between(v_mcp_pip, v_pip_dip)
        # DIP Flexion
        angles["FFJ0"] = angle_between(v_pip_dip, v_dip_tip)
        
        # MCP Flexion requires palm reference (approx using wrist and knuckles)
        wrist = landmarks[0]
        mcp_mid = landmarks[9] # Middle MCP
        mcp_index = landmarks[5]
        
        v_palm = mcp_mid - wrist
        angles["FFJ2"] = angle_between(v_palm, v_mcp_pip)
        
        # Abduction (Angle between Index MCP and Middle MCP)
        v_mcp_index_mid = mcp_index - mcp_mid
        # This is a simplification; actual abduction is in the palm plane
        
        # TODO: Implement full 24 DoF mapping
        # Thumb, Middle, Ring, Pinky
        
        return angles

    def retarget_sequence(self, hand_traj: np.ndarray) -> np.ndarray:
        """
        Retarget a sequence of landmarks to joint angle vectors.
        hand_traj: (num_frames, 21, 3)
        Returns: (num_frames, 28) including arm
        """
        num_frames = hand_traj.shape[0]
        joint_vectors = np.zeros((num_frames, 28))
        
        for i in range(num_frames):
            angles = self.compute_angles(hand_traj[i])
            # Map angles dict to fixed vector indices for Adroit
            # Placeholder mapping
            joint_vectors[i, 4] = angles.get("FFJ2", 0) # Index MCP
            joint_vectors[i, 5] = angles.get("FFJ1", 0) # Index PIP
            joint_vectors[i, 6] = angles.get("FFJ0", 0) # Index DIP
            
        return joint_vectors
