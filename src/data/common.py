from dataclasses import dataclass
import numpy as np

@dataclass
class FrameData:
    frame_id: int
    image: np.ndarray # (H, W, 3), uint8
    depth: np.ndarray # (H, W), float32, meters
    pose_gt: np.ndarray # (4, 4), float32, SE(3) transformation matrix
    K: np.ndarray # (3, 3), float32, camera intrinsic matrix