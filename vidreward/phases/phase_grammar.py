from typing import List, Dict
from .phase_detector import PhaseSegment

class PhaseGrammar:
    """
    Defines allowed transitions and validates phase sequences.
    """
    # Standard manipulation sequence
    REQUIRED_SEQUENCE = [
        "APPROACH",
        "GRASP",
        "TRANSPORT",
        "RELEASE"
    ]
    
    # Allowed transitions map
    TRANSITIONS = {
        "IDLE": ["APPROACH"],
        "APPROACH": ["GRASP", "IDLE"],
        "GRASP": ["TRANSPORT", "RELEASE"],
        "TRANSPORT": ["RELEASE"],
        "RELEASE": ["RETREAT", "IDLE"],
        "RETREAT": ["IDLE", "APPROACH"]
    }

    @classmethod
    def validate_sequence(cls, phases: List[PhaseSegment]) -> Dict:
        """
        Check if the phase sequence makes physical sense.
        """
        labels = [p.label for p in phases]
        unique_labels = []
        if labels:
            unique_labels.append(labels[0])
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    unique_labels.append(labels[i])
        
        found_required = [label for label in cls.REQUIRED_SEQUENCE if label in unique_labels]
        is_complete = len(found_required) == len(cls.REQUIRED_SEQUENCE)
        
        # Check for illegal jumps
        violations = []
        for i in range(len(unique_labels) - 1):
            curr, nxt = unique_labels[i], unique_labels[i+1]
            if nxt not in cls.TRANSITIONS.get(curr, []):
                violations.append(f"Illegal transition: {curr} -> {nxt}")
        
        return {
            "is_complete": is_complete,
            "missing_phases": set(cls.REQUIRED_SEQUENCE) - set(found_required),
            "violations": violations,
            "sequence": unique_labels
        }
