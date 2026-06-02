from pathlib import Path
from .replica import load_K, load_seq
from .common import Dataset

def load_dataset(config):
    dataset_name = config["dataset"]["name"]
    sequence = config["dataset"]["sequence"]
    data_path = Path(config["dataset"]["path"])
    dataset = None
    if dataset_name == "replica":
        K, cam = load_K(data_path)
        seq_path = data_path / sequence
        frames = load_seq(
            seq_path, 
            depth_scale=cam["scale"],
            stride=config["dataset"]["stride"],
            max_frames=config["dataset"]["max_frames"]
        )
        dataset = Dataset(
            frames=frames,
            sequence_dir=seq_path,
            image_dir=seq_path / "results",
            K=K
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return dataset