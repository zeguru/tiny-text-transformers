
import torch

def get_device(device: str | torch.device | None = None) -> torch.device:
    """Return the requested device, or choose the best available one."""
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def move_to_device(data, device: str | torch.device):
    """Recursively move tensors in common batch containers to a device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)

    if isinstance(data, dict):
        return {
            key: move_to_device(value, device)
            for key, value in data.items()
        }

    if isinstance(data, tuple):
        return tuple(move_to_device(value, device) for value in data)

    if isinstance(data, list):
        return [move_to_device(value, device) for value in data]

    return data
