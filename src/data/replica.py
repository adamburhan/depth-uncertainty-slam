import numpy as np
import cv2
from pathlib import Path
from .common import FrameData

import json

def load_K(data_path: Path) -> tuple[np.ndarray, dict]:
    with open(data_path / "cam_params.json") as f:
        cam = json.load(f)["camera"]
    K = np.array([
        [cam["fx"],        0, cam["cx"]],
        [       0, cam["fy"], cam["cy"]],
        [       0,         0,         1]
    ], dtype=np.float32)
    return K, cam

def load_frame(seq_path: Path, frame_id: int, depth_scale: float) -> tuple[FrameData, np.ndarray]:
    img = cv2.imread(str(seq_path / "results" / f"frame{frame_id:06d}.jpg"))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(str(seq_path / "results" / f"depth{frame_id:06d}.png"), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / depth_scale
    return img, depth

def load_gt_poses(seq_path: Path) -> np.ndarray:
    gt_poses = np.loadtxt(seq_path / "traj.txt")
    gt_poses = gt_poses.reshape(-1, 4, 4).astype(np.float32)
    return gt_poses

def load_seq(seq_path: Path, K: np.ndarray, depth_scale: float) -> list[FrameData]:
    if not seq_path.is_dir():
        raise ValueError(f"Sequence path {seq_path} is not a directory")
    
    gt_poses = load_gt_poses(seq_path)

    frame_ids = sorted([int(p.stem[5:]) for p in (seq_path / "results").glob("frame*.jpg")])
    frames = []
    for frame_id in frame_ids:
        img, depth = load_frame(seq_path, frame_id, depth_scale)
        frame_data = FrameData(
            frame_id=frame_id,
            image=img,
            depth=depth,
            pose_gt=gt_poses[frame_id],
            K=K
        )
        frames.append(frame_data)
    return frames


if __name__ == "__main__":
    data_path = Path("/home/adam/scratch/datasets/replica/Replica")
    K, cam = load_K(data_path)
    seq_path = data_path / "office0" 
    frames = load_seq(seq_path, K, depth_scale=cam["scale"])
    print(f"Loaded {len(frames)} frames")
    
    f = frames[0]
    
    # image
    print(f"image shape: {f.image.shape}, dtype: {f.image.dtype}")
    print(f"image range: [{f.image.min()}, {f.image.max()}]")
    
    # depth
    print(f"depth shape: {f.depth.shape}, dtype: {f.depth.dtype}")
    print(f"depth range: [{f.depth.min():.3f}, {f.depth.max():.3f}]")
    print(f"depth invalid (0): {(f.depth == 0).mean()*100:.1f}%")
    
    # pose sanity
    print(f"pose[0]:\n{f.pose_gt}")
    print(f"pose det: {np.linalg.det(f.pose_gt[:3,:3]):.6f}")  # should be ~1.0
    
    # visual check
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(f.image)
    axes[1].imshow(f.depth, cmap='plasma')
    plt.savefig("diagnostic.png")