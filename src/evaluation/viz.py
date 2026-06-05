import numpy as np
import open3d as o3d

COLORS = {
    "surface":       [0.55, 0.55, 0.55],
    "clean_floater": [1.00, 0.55, 0.00],
    "amb_floater":   [0.90, 0.05, 0.05],
}

def floater_cloud(pts, dist, amb_mask, tau=0.05):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(np.asarray(pts, float))
    colors = np.tile(COLORS["surface"], (len(pts), 1))
    fl = dist > tau
    colors[fl & ~amb_mask] = COLORS["clean_floater"]
    colors[fl &  amb_mask] = COLORS["amb_floater"]
    pc.colors = o3d.utility.Vector3dVector(colors)
    return pc
