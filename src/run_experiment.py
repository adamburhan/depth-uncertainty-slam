from .data.loader import load_dataset
from .observations.builder import build_observations
# from initialization.init import initialize
# from optimization.ba import optimize
# from evaluation.metrics import evaluate

config = {
    "dataset": {
        "name": "replica",
        "path": "/home/adam/scratch/datasets/replica/Replica",
        "sequence": "office0",
        "stride": 10,
        "max_frames": 100
    },
    "optimizer": {
        # optimizer config here
    }
}

def run_experiment(config):
    dataset = load_dataset(config) # returns Dataset

    observations = build_observations(dataset, config)

    # state0 = initialize(observations, config)
    # result = optimize(state0, observations, config.optimizer)

    # metrics = evaluate(result, config)

    # return metrics

if __name__ == "__main__":
    run_experiment(config)