import pycolmap
import os
from pathlib import Path
from collections import defaultdict
import sqlite3
import numpy as np

@dataclass
class Observation:
    frame_id: int
    landmark_id: int
    uv: np.ndarray        # (2,) pixel coordinates
    depth: float          # depth at keypoint location
    ambiguous: bool       # near depth discontinuity
    depth_alt: float | None = None  # second depth hypothesis if ambiguous


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

def run_colmap_frontend(database_path, image_path):
    if database_path.exists():
        database_path.unlink()
    pycolmap.set_random_seed(0)
    pycolmap.extract_features(database_path, str(image_path))
    pycolmap.match_exhaustive(database_path)

def load_keypoints(database_path) -> dict[int, np.ndarray]:
    """Returns {image_id: (N, 2) array of (u, v) coordinates}"""
    conn = sqlite3.connect(str(database_path))
    rows = conn.execute(
        "SELECT image_id, rows, cols, data FROM keypoints"
    ).fetchall()
    conn.close()
    keypoints = {}
    for image_id, n_kps, cols, data in rows:
        kps = np.frombuffer(data, dtype=np.float32).reshape(n_kps, cols)
        keypoints[image_id] = kps[:, :2]  # only (u, v), drop scale/orientation
    return keypoints

def load_two_view_matches(database_path) -> dict[tuple[int,int], np.ndarray]:
    conn = sqlite3.connect(str(database_path))
    rows = conn.execute(
        "SELECT pair_id, rows, data FROM two_view_geometries WHERE rows > 0"
    ).fetchall()
    conn.close()
    two_view_matches = {}
    for pair_id, n_matches, data in rows:
        img_id1, img_id2 = pair_id_to_image_ids(pair_id)
        matches = np.frombuffer(data, dtype=np.uint32).reshape(n_matches, 2)
        two_view_matches[(img_id1, img_id2)] = matches
    return two_view_matches

def build_tracks(two_view_matches) -> dict[int, list]:
    """
    Returns:
        tracks = {
            landmark_id: [(img_id, kp_idx), ...],
            ...
        }
    Invalid tracks (multiple observations from same image) are filtered out.
    """
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for (img1, img2), matches in two_view_matches.items():
        for kp1, kp2 in matches:
            union((img1, int(kp1)), (img2, int(kp2)))

    component_to_id = {}
    tracks = defaultdict(list)
    for node in parent:
        root = find(node)
        if root not in component_to_id:
            component_to_id[root] = len(component_to_id)
        tracks[component_to_id[root]].append(node)

    # filter invalid tracks: multiple keypoints from same image
    valid_tracks = {}
    n_invalid = 0
    for landmark_id, obs in tracks.items():
        image_ids = [img for img, kp in obs]
        if len(image_ids) == len(set(image_ids)):
            valid_tracks[landmark_id] = obs
        else:
            n_invalid += 1
    print(f"Tracks: {len(valid_tracks)} valid, {n_invalid} invalid (filtered)")

    return valid_tracks

def build_observations(dataset, config):
    database_path = dataset.sequence_dir / "database.db"
    
    if config["dataset"]["name"] == "replica":
        tmp_dir = dataset.sequence_dir / "tmp_colmap"
        tmp_dir.mkdir(exist_ok=True)
        for frame in dataset.frames:
            src = dataset.image_dir / f"frame{frame.frame_id:06d}.jpg"
            dst = tmp_dir / f"frame{frame.frame_id:06d}.jpg"
            if not dst.exists():
                os.symlink(src, dst)
        image_path = tmp_dir
    else:
        image_path = dataset.image_dir

    run_colmap_frontend(database_path, image_path)

    keypoints = load_keypoints(database_path)
    two_view_matches = load_two_view_matches(database_path)
    img_id_to_frame_id = build_img_id_to_frame_id(database_path)
    tracks = build_tracks(two_view_matches)

    frame_id_to_frame = {f.frame_id: f for f in dataset.frames}

    observations = []
    for landmark_id, obs in tracks.items():
        for img_id, kp_idx in obs:
            frame_id = img_id_to_frame_id[img_id]
            frame = frame_id_to_frame[frame_id]
            uv = keypoints[img_id][kp_idx]
            u, v = int(round(uv[0])), int(round(uv[1]))
            depth = float(frame.depth[v, u])

            if depth == 0.0:
                continue  # invalid depth

            # TODO: implement depth discontinuity heuristic
            ambiguous = False
            depth_alt = None

            observations.append(Observation(
                frame_id=frame_id,
                landmark_id=landmark_id,
                uv=uv,
                depth=depth,
                ambiguous=ambiguous,
                depth_alt=depth_alt,
            ))

    print(f"Total observations: {len(observations)}")
    return observations