import gtsam
import numpy as np
from gtsam import symbol_shorthand
L = symbol_shorthand.L
X = symbol_shorthand.X

from gtsam import (
    Cal3_S2,
    DoglegOptimizer,
    LevenbergMarquardtOptimizer,
    GenericProjectionFactorCal3_S2,
    Marginals,
    NonlinearFactorGraph,
    PinholeCameraCal3_S2,
    Point3,
    Point2,
    Pose3,
    PriorFactorPoint3,
    PriorFactorPose3,
    Rot3,
    Values
)
from gtsam.utils import plot

def make_depth_factor(pose_key, lm_key, d, noise):
    """Binary camera-Z depth factor on pose and landmark"""
    
    def error_func(this, values, H):
        pose_wTc = values.atPose3(pose_key)
        point_w = values.atPoint3(lm_key)

        H_pose = np.zeros((3, 6), dtype=np.float64, order="F")
        H_point = np.zeros((3, 3), dtype=np.float64, order="F")

        point_c = pose_wTc.transformTo(point_w, H_pose, H_point)
        z_pred = float(point_c[2])

        if H is not None:
            H[0] = H_pose[2:3, :]
            H[1] = H_point[2:3, :]
        
        return np.array([z_pred - d], dtype=np.float64)
    
    return gtsam.CustomFactor(noise, [pose_key, lm_key], error_func)

def make_bimodal_depth_factor(pose_key, lm_key, d, d_alt, noise):
    """Binary camera-Z depth factor on pose and landmark, 
    max-mixture with second depth hypothesis for ambiguous measurements near depth discontinuities"""
    
    def error_func(this, values, H):
        pose_wTc = values.atPose3(pose_key)
        point_w = values.atPoint3(lm_key)

        H_pose = np.zeros((3, 6), dtype=np.float64, order="F")
        H_point = np.zeros((3, 3), dtype=np.float64, order="F")

        point_c = pose_wTc.transformTo(point_w, H_pose, H_point)
        z_pred = float(point_c[2])
        r1, r2 = z_pred - d, z_pred - d_alt
        err = r1 if abs(r1) < abs(r2) else r2

        if H is not None:
            H[0] = H_pose[2:3, :]
            H[1] = H_point[2:3, :]
        
        return np.array([err], dtype=np.float64)
    
    return gtsam.CustomFactor(noise, [pose_key, lm_key], error_func)


def optimize(state0, observations, optimizer_config):
    K = Cal3_S2(
        state0["intrinsics"][0, 0], 
        state0["intrinsics"][1, 1], 
        0.0, 
        state0["intrinsics"][0, 2], 
        state0["intrinsics"][1, 2]
    )

    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(
        2, optimizer_config["measurement_noise_sigma"]
    )
    depth_noise = gtsam.noiseModel.Isotropic.Sigma(
        1, optimizer_config["depth_noise_sigma"]
    )

    depth_model = optimizer_config["depth_model"]

    poses_dict = state0["poses"] # frame_id -> (4, 4) SE(3) matrix
    landmarks_dict = state0["landmarks"] # lm_id -> (3,) XYZ point

    graph = NonlinearFactorGraph()

    for obs in observations:
        pose_key = X(obs.frame_id)
        lm_key = L(obs.landmark_id)

        graph.add(
            GenericProjectionFactorCal3_S2(
                Point2(obs.uv), measurement_noise, pose_key, lm_key, K
            )
        )

        if depth_model == "none":
            continue
        
        if depth_model == "drop_ambiguous" and obs.ambiguous:
            continue

        if depth_model == "bimodal" and obs.ambiguous:
            assert obs.depth_alt is not None

            graph.add(
                make_bimodal_depth_factor(
                    pose_key, lm_key, obs.depth, obs.depth_alt, depth_noise
                )
            )
        else:
            graph.add(
                make_depth_factor(
                    pose_key, lm_key, obs.depth, depth_noise
                )
            )


    # assign initial values
    initial = Values()
    for frame_id, pose in poses_dict.items():
        initial.insert(X(frame_id), Pose3(pose))
    for lm_id, xyz in landmarks_dict.items():
        initial.insert(L(lm_id), Point3(xyz))

    # poses come from gt
    for frame_id, pose in poses_dict.items():
        graph.add(
            PriorFactorPose3(
                X(frame_id), Pose3(pose), gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
            )
        )

    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")
    params.setMaxIterations(optimizer_config.get("max_iterations", 100))
    params.setRelativeErrorTol(optimizer_config.get("relative_error_tol", 1e-5))

    initial_error = graph.error(initial)
    print(f"Initial error: {initial_error:.6f}")

    optimizer = LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    final_error = graph.error(result)
    print(f"Final error: {final_error:.6f}")

    return result, {"initial_error": initial_error, "final_error": final_error}