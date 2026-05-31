"""
Synthetic scene generator for testing bimodal-depth bundle adjustment.

Builds a two-surface scene -- a foreground patch in front of a background
plane -- viewed by several cameras with known poses. Each observation is a
(u, v, Z) measurement: pixel location plus ground-truth camera-frame depth.

Keypoints on the foreground-patch boundary are "ambiguous": near the depth
discontinuity, localization noise can make the depth read land on the wrong
surface. For those observations the generator records BOTH candidate depths
(foreground and background along that ray) and, with probability
`missample_rate`, replaces the read depth with the background depth -- the
wrong-surface sample the bimodal model is meant to reject. Ground-truth surface
labels and the noiseless true-surface depth are kept for scoring/initialization.

Pure NumPy. Poses are 4x4 world_from_cam matrices; convert to your optimizer's
convention downstream. Occlusion is NOT modeled (surfaces are transparent point
clouds) -- adequate for testing the depth factor; revisit if needed.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Observation:
    cam_id: int
    landmark_id: int
    u: float            # pixel x (with localization noise)
    v: float            # pixel y (with localization noise)
    z_read: float       # measured depth: true surface, unless missampled (+ noise)
    z_true: float       # noiseless camera-frame depth of the TRUE surface
    z_fg: float         # foreground candidate depth along this ray
    z_bg: float         # background candidate depth along this ray (nan if none)
    is_ambiguous: bool
    true_surface: str   # 'fg' or 'bg'
    was_missampled: bool


@dataclass
class Landmark:
    id: int
    xyz: np.ndarray     # true 3D world position
    surface: str        # 'fg' or 'bg'
    is_ambiguous: bool


@dataclass
class Scene:
    K: np.ndarray                 # 3x3 intrinsics
    image_size: tuple             # (width, height)
    poses: list                   # 4x4 world_from_cam per camera
    landmarks: list               # list[Landmark]
    observations: list            # list[Observation]
    z_fg_plane: float             # GT foreground plane (world Z)
    z_bg_plane: float             # GT background plane (world Z)
    params: dict = field(default_factory=dict)


# ---------- geometry helpers ----------

def _lookat(eye, target, up=(0.0, -1.0, 0.0)):
    """4x4 world_from_cam. Camera +Z -> target, +Y -> image-down."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    up = np.asarray(up, float)
    z = target - eye; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    T[:3, 3] = eye
    return T


def _cam_from_world(pose):
    R = pose[:3, :3]; t = pose[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _project(K, pose, Xw):
    """Return (u, v, depth). u,v are None if the point is behind the camera."""
    cfw = _cam_from_world(pose)
    Xc = cfw[:3, :3] @ Xw + cfw[:3, 3]
    depth = float(Xc[2])
    if depth <= 1e-6:
        return None, None, depth
    uvw = K @ Xc
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2]), depth


def _ray_plane_depth(K, pose, u, v, z_plane_world):
    """Back-project pixel (u,v), intersect world plane Z=z_plane_world,
    return the camera-frame depth of that intersection (or None)."""
    d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    R = pose[:3, :3]; eye = pose[:3, 3]
    d_world = R @ d_cam
    if abs(d_world[2]) < 1e-9:
        return None
    s = (z_plane_world - eye[2]) / d_world[2]
    if s <= 0:
        return None
    pt = eye + s * d_world
    _, _, depth = _project(K, pose, pt)
    return depth


# ---------- generator ----------

def generate_scene(
    seed=0,
    n_cameras=1,
    baseline=0.8,            # total horizontal spread of cameras (m)
    z_fg_plane=3.0,
    z_bg_plane=6.0,
    fg_half_extent=0.8,      # foreground patch half-size in world XY (m)
    bg_half_extent=2.5,      # background plane half-size in world XY (m)
    n_fg=40,
    n_bg=80,
    edge_margin=0.12,        # outer fraction of fg patch flagged ambiguous
    image_size=(640, 480),
    focal=500.0,
    pixel_noise_std=0.5,     # keypoint localization noise (px)
    depth_noise_std=0.02,    # within-mode depth noise (m); keeps BA non-trivial
    missample_rate=0.3,      # P(ambiguous obs reads the wrong surface)
):
    rng = np.random.default_rng(seed)
    W, H = image_size
    K = np.array([[focal, 0, W / 2.0],
                  [0, focal, H / 2.0],
                  [0, 0, 1.0]])

    # cameras: horizontal arc at z=0 looking toward the scene
    xs = np.linspace(-baseline / 2, baseline / 2, n_cameras)
    target = (0.0, 0.0, (z_fg_plane + z_bg_plane) / 2)
    poses = [_lookat((x, 0.0, 0.0), target) for x in xs]

    # landmarks
    landmarks = []
    lid = 0
    thr = (1.0 - edge_margin) * fg_half_extent
    for _ in range(n_fg):
        x = rng.uniform(-fg_half_extent, fg_half_extent)
        y = rng.uniform(-fg_half_extent, fg_half_extent)
        near_edge = (abs(x) > thr) or (abs(y) > thr)
        landmarks.append(Landmark(lid, np.array([x, y, z_fg_plane]), 'fg', near_edge))
        lid += 1
    for _ in range(n_bg):
        x = rng.uniform(-bg_half_extent, bg_half_extent)
        y = rng.uniform(-bg_half_extent, bg_half_extent)
        landmarks.append(Landmark(lid, np.array([x, y, z_bg_plane]), 'bg', False))
        lid += 1

    # observations
    obs = []
    for cid, pose in enumerate(poses):
        for lm in landmarks:
            u, v, depth = _project(K, pose, lm.xyz)
            if u is None or not (0 <= u < W and 0 <= v < H):
                continue
            un = u + rng.normal(0, pixel_noise_std)
            vn = v + rng.normal(0, pixel_noise_std)
            if not (0 <= un < W and 0 <= vn < H):
                continue

            if lm.is_ambiguous:
                z_fg = depth                                   # true fg-surface depth
                z_bg = _ray_plane_depth(K, pose, un, vn, z_bg_plane)
                if z_bg is None:
                    z_bg = np.nan
                missampled = (not np.isnan(z_bg)) and (rng.random() < missample_rate)
                z_surface = z_bg if missampled else z_fg
                z_read = z_surface + rng.normal(0, depth_noise_std)
                obs.append(Observation(cid, lm.id, un, vn, z_read, depth,
                                       z_fg, z_bg, True, 'fg', missampled))
            else:
                z_read = depth + rng.normal(0, depth_noise_std)
                obs.append(Observation(cid, lm.id, un, vn, z_read, depth,
                                       depth, np.nan, False, lm.surface, False))

    return Scene(K=K, image_size=image_size, poses=poses, landmarks=landmarks,
                 observations=obs, z_fg_plane=z_fg_plane, z_bg_plane=z_bg_plane,
                 params=dict(seed=seed, n_cameras=n_cameras, baseline=baseline,
                             missample_rate=missample_rate,
                             pixel_noise_std=pixel_noise_std,
                             depth_noise_std=depth_noise_std,
                             edge_margin=edge_margin))


if __name__ == "__main__":
    scene = generate_scene()
    n_amb = sum(o.is_ambiguous for o in scene.observations)
    n_mis = sum(o.was_missampled for o in scene.observations)
    n_amb_lm = sum(lm.is_ambiguous for lm in scene.landmarks)
    n_fg = sum(lm.surface == 'fg' for lm in scene.landmarks)
    n_bg = sum(lm.surface == 'bg' for lm in scene.landmarks)
    print(f"cameras:      {len(scene.poses)}")
    print(f"landmarks:    {len(scene.landmarks)}  ({n_fg} fg / {n_bg} bg, "
          f"{n_amb_lm} ambiguous)")
    print(f"observations: {len(scene.observations)}")
    print(f"  ambiguous:  {n_amb}")
    print(f"  missampled: {n_mis}  ({100 * n_mis / max(n_amb, 1):.0f}% of ambiguous)")
    print("\nsample ambiguous obs (cam, lm, z_read, z_fg, z_bg, missampled):")
    shown = 0
    for o in scene.observations:
        if o.is_ambiguous and shown < 6:
            print(f"  cam{o.cam_id} lm{o.landmark_id:<3d} "
                  f"z_read={o.z_read:.3f}  z_fg={o.z_fg:.3f}  "
                  f"z_bg={o.z_bg:.3f}  missampled={o.was_missampled}")
            shown += 1