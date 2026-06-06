import torch


def load_checkpoint(path, map_location="cuda"):
    """Load checkpoints across PyTorch versions.

    PyTorch 1.x does not support the weights_only argument, while newer
    versions warn unless it is set explicitly.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

