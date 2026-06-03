from .data.loader import load_dataset
from .observations.builder import build_observations
import numpy as np
# from initialization.init import initialize
# from optimization.ba import optimize
# from evaluation.metrics import evaluate

config = {
    "dataset": {
        "name": "replica",
        "path": "/home/adam/scratch/datasets/replica/Replica",
        "sequence": "office0",
        "stride": 5,
        "max_frames": 100
    },
    "optimizer": {
        # optimizer config here
    }
}

def run_experiment(config):
    dataset = load_dataset(config) # returns Dataset

    observations, landmark_xyz = build_observations(dataset, config)

    # basic stats
    print(f"Observations: {len(observations)}")
    print(f"Unique landmarks: {len(set(o.landmark_id for o in observations))}")
    print(f"Unique cameras: {len(set(o.frame_id for o in observations))}")

    # depth sanity
    depths = np.array([o.depth for o in observations])
    print(f"Depth range: [{depths.min():.3f}, {depths.max():.3f}]")
    print(f"Depth mean: {depths.mean():.3f}")

    # track length distribution
    from collections import Counter
    track_lengths = Counter(o.landmark_id for o in observations)
    lengths = list(track_lengths.values())
    print(f"Track length min/mean/max: {min(lengths)}/{np.mean(lengths):.1f}/{max(lengths)}")
    print(f"Landmarks seen in 2+ images: {sum(1 for l in lengths if l >= 2)}")

    # state0 = initialize(observations, config)
    # result = optimize(state0, observations, config.optimizer)

    # metrics = evaluate(result, config)

    # return metrics

if __name__ == "__main__":
    run_experiment(config)