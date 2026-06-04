import pycolmap
import os, shutil
from pathlib import Path
from collections import defaultdict
import sqlite3
import numpy as np
from dataclasses import dataclass
from .depth_hypotheses import analyze_depth_patch

@dataclass
class Observation:
    frame_id: int
    landmark_id: int
    uv: np.ndarray        # (2,) pixel coordinates
    depth: float          # depth at keypoint location
    ambiguous: bool       # near depth discontinuity
    depth_alt: float | None = None  # second depth hypothesis if ambiguous
    ambiguity_score: float = 0.0

def pair_id_to_image_ids(pair_id):
    """From colmap documentation: https://colmap.github.io/database.html"""
    img_id2 = pair_id % 2147483647
    img_id1 = (pair_id - img_id2) // 2147483647
    return img_id1, img_id2

def build_img_id_to_frame_id(database_path) -> dict[int, int]:
    conn = sqlite3.connect(str(database_path))
    rows = conn.execute("SELECT image_id, name FROM images").fetchall()
    conn.close()
    return {img_id: int(name[5:11]) for img_id, name in rows}

def run_colmap_frontend(database_path, image_path, cam):
    if database_path.exists():
        database_path.unlink()
    reader_options = pycolmap.ImageReaderOptions()
    reader_options.camera_model = "PINHOLE"
    reader_options.camera_params = f"{cam['fx']},{cam['fy']},{cam['cx']},{cam['cy']}"
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(
        database_path,
        str(image_path),
        camera_mode=pycolmap.CameraMode.SINGLE,
        reader_options=reader_options,
    )
    verification_options = pycolmap.TwoViewGeometryOptions()
    verification_options.compute_relative_pose = True
    pycolmap.match_exhaustive(database_path, verification_options=verification_options)


def seed_reconstruction(cam, frames, name_to_img_id):
    rec = pycolmap.Reconstruction()
    camera = pycolmap.Camera.create_from_model_name(1, "PINHOLE", cam["fx"], cam["w"], cam["h"])
    camera.params = [cam["fx"], cam["fy"], cam["cx"], cam["cy"]]
    camera.camera_id = 1
    rec.add_camera_with_trivial_rig(camera)
    for f in frames:
        img_name = f"frame{f.frame_id:06d}.jpg"
        img_id = name_to_img_id[img_name]
        
        # Replica pose_gt is cam-to-world, need world-to-cam
        R_wc = f.pose_gt[:3, :3].astype(np.float64)
        t_wc = f.pose_gt[:3, 3].astype(np.float64)
        R_cw = R_wc.T
        t_cw = -R_wc.T @ t_wc
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R_cw), t_cw)
        
        im = pycolmap.Image()
        im.image_id = img_id
        im.name = img_name
        im.camera_id = 1
        rec.add_image_with_trivial_frame(im, cam_from_world)
    return rec

def build_observations(dataset, config):
    database_path = dataset.sequence_dir / "database.db"
    
    if config["dataset"]["name"] == "replica":
        tmp_dir = dataset.sequence_dir / "tmp_colmap"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(exist_ok=True)
        for frame in dataset.frames:
            src = dataset.image_dir / f"frame{frame.frame_id:06d}.jpg"
            dst = tmp_dir / f"frame{frame.frame_id:06d}.jpg"
            if not dst.exists():
                os.symlink(src, dst)
        image_path = tmp_dir
    else:
        image_path = dataset.image_dir

    run_colmap_frontend(database_path, image_path, dataset.cam)

    with sqlite3.connect(str(database_path)) as conn:
        name_to_img_id = {
            name: image_id
            for image_id, name in conn.execute("SELECT image_id, name FROM images")
        }

    rec = seed_reconstruction(dataset.cam, dataset.frames, name_to_img_id)  # frames carry image_id + GT pose
    out = dataset.sequence_dir / "tri_model"; out.mkdir(exist_ok=True)
    rec = pycolmap.triangulate_points(rec, str(database_path), str(image_path), str(out))

    img_id_to_frame_id = build_img_id_to_frame_id(database_path)
    frame_id_to_frame = {f.frame_id: f for f in dataset.frames}

    observations = []
    landmark_xyz = {}                       
    for lm_id, p3d in rec.points3D.items():
        recs = []
        seen = set()
        for e in p3d.track.elements:
            if e.image_id in seen:
                continue
            frame = frame_id_to_frame[img_id_to_frame_id[e.image_id]]
            uv = rec.images[e.image_id].points2D[e.point2D_idx].xy
            assert frame.depth.shape == (dataset.cam["h"], dataset.cam["w"])
            u = np.clip(int(np.floor(uv[0])), 0, frame.depth.shape[1] - 1)
            v = np.clip(int(np.floor(uv[1])), 0, frame.depth.shape[0] - 1)
            ambiguous, d, d_alt, score = analyze_depth_patch(frame.depth, u, v, config["depth_hypotheses"])
            if d == 0.0:
                continue
            seen.add(e.image_id) # keep one obs per (landmark, image)
            recs.append(
                Observation(
                    frame.frame_id, 
                    lm_id, 
                    uv, 
                    d, 
                    ambiguous=ambiguous, 
                    depth_alt=d_alt,
                    ambiguity_score=score
                )
            )
        if len(recs) >= 2:                  # guard: track must survive depth filtering with >=2 obs
            observations.extend(recs)
            landmark_xyz[lm_id] = p3d.xyz
    print(f"Landmarks: {len(landmark_xyz)}, observations: {len(observations)}")

    return observations, landmark_xyz