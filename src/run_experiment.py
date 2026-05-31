from data.loader import load_dataset
from observations.builder import build_observations
from initialization.init import initialize
from optimization.ba import optimize
from evaluation.metrics import evaluate

def run_experiment(config):
    images, depths, poses, K = load_dataset(config)

    observations = build_observations(images, depths, config)

    state0 = initialize(observations, poses, K, config)

    result = optimize(state0, observations, config.optimizer)

    metrics = evaluate(result, config)

    return metrics