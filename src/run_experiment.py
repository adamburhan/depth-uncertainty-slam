from .data.loader import load_dataset
from .observations.builder import build_observations
import numpy as np
import argparse
import pickle
import yaml
from pathlib import Path
from collections import defaultdict
from .initialization.init import initialize_state
from .optimization.ba import optimize
from .evaluation.metrics import evaluate

config = {
    "dataset": {
        "name": "replica",
        "path": "/home/adam/scratch/datasets/replica/Replica",
        "sequence": "office0",
        "stride": 1,
        "max_frames": 100
    },
    "optimizer": {
        # optimizer config here
        "depth_model": "bimodal",  # "none", "drop_ambiguous", "bimodal"
        "measurement_noise_sigma": 1.0,
        "depth_noise_sigma": 0.1,
    },
    "depth_hypotheses": {
        "depth_patch_method": "largest_gap",
        "patch_radius": 5,
        "gap_thresh": 0.15,
        "ambiguity_thresh": 0.20,
        "min_valid": 10,
    },
    "output": {
        "dir": "runs/office0_100",
    },
}

def diagnose_ambiguity(observations, config):
    n = len(observations)
    if n == 0:
        print("No observations.")
        return
    scores = np.array([o.ambiguity_score for o in observations])
    flagged = sum(o.ambiguous for o in observations)
    thresh = config["depth_hypotheses"]["ambiguity_thresh"]
    print(f"Ambiguous: {flagged}/{n} ({100*flagged/n:.2f}%) [thresh={thresh}]")
    print(f"Score min/mean/max: {scores.min():.4f}/{scores.mean():.4f}/{scores.max():.4f}")
    qs = [50, 75, 90, 95, 99]
    pcts = np.percentile(scores, qs)
    print("Percentiles: " + ", ".join(f"p{q}={v:.4f}" for q, v in zip(qs, pcts)))
    n_with_alt = sum(1 for o in observations if o.depth_alt is not None)
    print(f"Have depth_alt: {n_with_alt}/{n} ({100*n_with_alt/n:.2f}%)")

def build_state(config, save_path):
    dataset = load_dataset(config)  # returns Dataset

    observations, landmark_xyz = build_observations(dataset, config)

    diagnose_ambiguity(observations, config)

    state0 = initialize_state(dataset, landmark_xyz)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump({
            "observations": observations,
            "state0": state0,
            "mesh_path": dataset.mesh_path,
            "config": config,
        }, f)
    return observations, state0, dataset.mesh_path

def run_from_frozen(frozen_path, depth_model):
    """Optimize + evaluate from a frozen front-end, varying only depth_model."""
    with open(frozen_path, "rb") as f:
        S = pickle.load(f)
    observations, state0, mesh_path = S["observations"], S["state0"], S["mesh_path"]
    opt_cfg = {**S["config"]["optimizer"], "depth_model": depth_model}

    result, opt_metrics = optimize(state0, observations, opt_cfg)

    eval_metrics = evaluate(
        result, observations, state0, mesh_path,
        out_dir=S["config"]["output"]["dir"],
        tag=depth_model,
    )
    return {**opt_metrics, **eval_metrics}

def print_metrics(metrics):
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        else:
            print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="YAML config (defaults to built-in config dict).")
    parser.add_argument("--depth-model", required=True,
                        choices=("none", "drop_ambiguous", "bimodal"))
    args = parser.parse_args()

    cfg = config if args.config is None else yaml.safe_load(open(args.config))
    frozen_path = Path(cfg["output"]["dir"]) / "frozen.pkl"

    if not frozen_path.exists():
        build_state(cfg, frozen_path)  # pins observations + state0 for ALL depth_models

    print(f"===== depth_model={args.depth_model} =====")
    metrics = run_from_frozen(frozen_path, args.depth_model)
    print_metrics(metrics)