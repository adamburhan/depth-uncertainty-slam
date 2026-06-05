import numpy as np
import trimesh
import open3d as o3d
from gtsam import symbol_shorthand
L = symbol_shorthand.L

def dist_to_mesh(points, mesh):
    _, dist, _ = trimesh.proximity.closest_point(mesh, points)
    return np.asarray(dist)

def summary(d, thresholds=(0.05, 0.10)):
    out = {"n": int(d.size), "median": float(np.median(d)),
           "p95": float(np.percentile(d, 95)), "p99": float(np.percentile(d, 99)),
           "max": float(d.max())}
    for t in thresholds:
        out[f"floater_rate@{int(t*100)}cm"] = float((d > t).mean())
    return out

def evaluate(result, observations, state0, dataset):
    lm_ids    = list(state0["landmarks"].keys())
    pts_final = np.stack([np.asarray(result.atPoint3(L(i))) for i in lm_ids])
    pts_init  = np.stack([np.asarray(state0["landmarks"][i]) for i in lm_ids])

    mesh = trimesh.load(dataset.mesh_path, process=False)   # GT surface, same world frame as pinned poses

    amb_lms = {o.landmark_id for o in observations if o.ambiguous}  # per-obs flag -> per-landmark
    mask = np.array([i in amb_lms for i in lm_ids])

    metrics = {}
    for tag, pts in (("init", pts_init), ("final", pts_final)):
        d = dist_to_mesh(pts, mesh)
        for sub, m in (("all", np.ones(len(d), bool)), ("amb", mask), ("clean", ~mask)):
            if m.any():
                metrics[f"{tag}/{sub}"] = summary(d[m])
    return metrics