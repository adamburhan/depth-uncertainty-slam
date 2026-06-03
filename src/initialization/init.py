import numpy as np

def initialize_state(dataset, landmark_xyz):
    return {
        "poses": {
            f.frame_id: f.pose_gt.astype(np.float64) for f in dataset.frames
        },
        "landmarks": {
            lm_id: xyz.astype(np.float64) for lm_id, xyz in landmark_xyz.items()
        },
        "intrinsics": dataset.K.astype(np.float64),
    }