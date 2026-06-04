import numpy as np

def largest_gap(depth, u, v, cfg, scale=1.0):
    r = cfg["patch_radius"]
    h, w = depth.shape
    patch = depth[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
    d_center = float(depth[v, u])
    valid = patch[patch > 0.0]
    if d_center <= 0.0 or valid.size < cfg["min_valid"]:
        return False, d_center, None, 0.0

    logs = np.sort(np.log(valid))
    gaps = np.diff(logs)
    if gaps.size == 0:
        return False, d_center, None, 0.0
    
    k = int(np.argmax(gaps))
    max_gap = float(gaps[k])
    if max_gap < cfg["gap_thresh"]:
        return False, d_center, None, 0.0

    split = 0.5 * (logs[k] + logs[k + 1])
    near = valid[np.log(valid) <= split]
    far = valid[np.log(valid) > split]
    d_near, d_far = float(np.median(near)), float(np.median(far))
    d_alt = d_far if abs(np.log(d_center) - np.log(d_near)) <= abs(np.log(d_center) - np.log(d_far)) else d_near
    
    frac_far = far.size / valid.size
    balance = 1.0 - abs(0.5 - frac_far) * 2.0
    score = max_gap * balance
    if score >= cfg["ambiguity_thresh"]:
        return True, d_center, d_alt, score
    return False, d_center, None, score


def percentile(depth, u, v, cfg, scale=1.0):
    pass

METHODS = {
    "largest_gap": largest_gap,
    "percentile": percentile,
}

def analyze_depth_patch(depth, u, v, cfg, scale=1.0):
    return METHODS[cfg["depth_patch_method"]](depth, u, v, cfg, scale=scale)