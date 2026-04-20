import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Adjust this import to your project structure
# Example:
# from src.models.market_transformer import MarketTransformer
# from src.models.config import TransformerConfig
from ..transformer import MarketTransformer,TransformerConfig


class WindowDataset(Dataset):
    """
    Dataset for pre-windowed features.

    Expected array shape:
        (N, W, d_feat + d_sent)

    where
        N = number of samples
        W = window length
    """

    def __init__(self, features: np.ndarray):
        if features.ndim != 3:
            raise ValueError(
                f"Expected features with shape (N, W, D), got {features.shape}"
            )
        self.features = torch.tensor(features, dtype=torch.float32)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]


def load_features(path: str) -> np.ndarray:
    """
    Load windowed dataset from .npy or .pt.

    Supported:
      - .npy: numpy array of shape (N, W, D)
      - .pt : torch tensor or dict containing 'x'
    """
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == ".npy":
        arr = np.load(path_obj)
        return np.asarray(arr, dtype=np.float32)

    if suffix == ".pt":
        obj = torch.load(path_obj, map_location="cpu")
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().astype(np.float32)
        if isinstance(obj, dict):
            if "x" in obj:
                x = obj["x"]
                if isinstance(x, torch.Tensor):
                    return x.detach().cpu().numpy().astype(np.float32)
                return np.asarray(x, dtype=np.float32)
            raise KeyError("Expected key 'x' in .pt dict dataset.")
        raise TypeError("Unsupported .pt dataset format.")

    raise ValueError(f"Unsupported dataset file type: {suffix}")


def load_config(config_path: str) -> TransformerConfig:
    """
    Load TransformerConfig from a saved .pt/.pth file or a Python module.

    Option 1:
      torch.save(config.__dict__, "config.pt")

    Option 2:
      store a dict with keys matching TransformerConfig fields
    """
    obj = torch.load(config_path, map_location="cpu")

    if isinstance(obj, TransformerConfig):
        return obj

    if isinstance(obj, dict):
        return TransformerConfig(**obj)

    raise TypeError(
        "Unsupported config format. Expected TransformerConfig or dict."
    )


def build_model(
    config_path: str,
    checkpoint_path: str,
    device: torch.device,
) -> MarketTransformer:
    config = load_config(config_path)
    model = MarketTransformer(config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Support either raw state_dict or training checkpoint dict
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warning] Missing keys when loading checkpoint: {missing}")
    if unexpected:
        print(f"[Warning] Unexpected keys when loading checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_latents(
    model: MarketTransformer,
    dataloader: DataLoader,
    device: torch.device,
    return_z_pre: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    z_all = []
    z_pre_all = [] if return_z_pre else None

    for batch_x in dataloader:
        batch_x = batch_x.to(device, non_blocking=True)

        out = model(batch_x)
        # forward() returns keys "z" and "z_pre" among others :contentReference[oaicite:1]{index=1}
        z = out["z"].detach().cpu().numpy()
        z_all.append(z)

        if return_z_pre:
            z_pre = out["z_pre"].detach().cpu().numpy()
            z_pre_all.append(z_pre)

    z_all = np.concatenate(z_all, axis=0)

    if return_z_pre:
        assert z_pre_all is not None
        z_pre_all = np.concatenate(z_pre_all, axis=0)
        return z_all, z_pre_all

    return z_all, None


def main():
    parser = argparse.ArgumentParser(
        description="Load trained MarketTransformer and extract latent vectors z."
    )
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to saved TransformerConfig (.pt/.pth containing dict or config object).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt/.pth).",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to dataset file (.npy or .pt) containing windowed inputs of shape (N, W, D).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="latents_z.npy",
        help="Path to save extracted z vectors.",
    )
    parser.add_argument(
        "--output-zpre-path",
        type=str,
        default=None,
        help="Optional path to save z_pre vectors.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use: cpu or cuda.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading features...")
    features = load_features(args.data_path)
    print(f"Feature shape: {features.shape}")

    dataset = WindowDataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("Loading model...")
    model = build_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        device=device,
    )

    return_z_pre = args.output_zpre_path is not None

    print("Running inference...")
    z, z_pre = extract_latents(
        model=model,
        dataloader=dataloader,
        device=device,
        return_z_pre=return_z_pre,
    )

    print(f"Latent z shape: {z.shape}")
    np.save(args.output_path, z)
    print(f"Saved z to: {args.output_path}")

    if return_z_pre and z_pre is not None:
        print(f"Latent z_pre shape: {z_pre.shape}")
        np.save(args.output_zpre_path, z_pre)
        print(f"Saved z_pre to: {args.output_zpre_path}")


if __name__ == "__main__":
    main()