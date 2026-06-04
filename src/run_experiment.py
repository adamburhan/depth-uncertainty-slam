from .data.loader import load_dataset
from .observations.builder import build_observations
import numpy as np
from collections import defaultdict
from .initialization.init import initialize_state
from .optimization.ba import optimize
# from evaluation.metrics import evaluate

config = {
    "dataset": {
        "name": "replica",
        "path": "/home/adam/scratch/datasets/replica/Replica",
        "sequence": "office0",
        "stride": 1,
        "max_frames": 10
    },
    "optimizer": {
        # optimizer config here
        "depth_model": "bimodal",
        "measurement_noise_sigma": 1.0,
        "depth_noise_sigma": 0.1,
    },
    "depth_hypotheses": {
        "depth_patch_method": "largest_gap",
        "patch_radius": 5,
        "gap_thresh": 0.15,
        "ambiguity_thresh": 0.20,
        "min_valid": 10,
    }
}



def run_experiment(config):
    dataset = load_dataset(config) # returns Dataset

    observations, landmark_xyz = build_observations(dataset, config)


    state0 = initialize_state(dataset, landmark_xyz)

    result, metrics = optimize(state0, observations, config["optimizer"])

    # metrics = evaluate(result, config)

    # return metrics

if __name__ == "__main__":
    run_experiment(config)