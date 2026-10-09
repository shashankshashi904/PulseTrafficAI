"""
PulseTrafficAI - XTraffic Phase 1 Data Loader

Loads the prepared XTraffic incident-aware traffic datasets
for Phase 1 IGSTGNN implementation.

Expected dataset structure:

data/raw/XTraffic/
    Alameda/
        adj_matrix.npy
        incident_train.npy
        incident_val.npy
        incident_test.npy
        incident_stats.npz
        sensors.csv
        desc_mapping.json
        type_mapping.json
    Contra_Costa/
        ...
    Orange/
        ...

This implementation is intentionally kept separate from temp_IGSTGNN.
"""

from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class XTrafficIncidentDataset(Dataset):
    """
    PyTorch Dataset for XTraffic incident-aware samples.

    Each sample contains:
        x_data
        y_data
        incident_features
        incident_position
        incident_distances
        durations
    """

    REQUIRED_KEYS = {
        "x_data",
        "y_data",
        "incident_features",
        "incident_position",
        "incident_distances",
        "durations",
    }

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.file_path}"
            )

        self.samples = np.load(
            self.file_path,
            allow_pickle=True
        )

        if len(self.samples) == 0:
            raise ValueError(
                f"Dataset contains no samples: {self.file_path}"
            )

        # Validate first sample structure.
        first_sample = self.samples[0]

        missing_keys = self.REQUIRED_KEYS - set(first_sample.keys())

        if missing_keys:
            raise ValueError(
                f"Missing keys in dataset sample: {missing_keys}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]

        return {
            "x_data": torch.tensor(
                sample["x_data"],
                dtype=torch.float32
            ),
            "y_data": torch.tensor(
                sample["y_data"],
                dtype=torch.float32
            ),
            "incident_features": sample["incident_features"],
            "incident_position": torch.tensor(
                sample["incident_position"],
                dtype=torch.long
            ),
            "incident_distances": torch.tensor(
                sample["incident_distances"],
                dtype=torch.float32
            ),
            "durations": torch.tensor(
                sample["durations"],
                dtype=torch.float32
            ),
        }


def load_adjacency(
    dataset_dir: str | Path,
) -> np.ndarray:
    """
    Load the XTraffic adjacency matrix.
    """

    dataset_dir = Path(dataset_dir)

    adjacency_path = dataset_dir / "adj_matrix.npy"

    if not adjacency_path.exists():
        raise FileNotFoundError(
            f"Adjacency matrix not found: {adjacency_path}"
        )

    adjacency = np.load(adjacency_path)

    return adjacency


def load_statistics(
    dataset_dir: str | Path,
) -> Dict[str, np.ndarray]:
    """
    Load traffic normalization statistics.
    """

    dataset_dir = Path(dataset_dir)

    stats_path = dataset_dir / "incident_stats.npz"

    if not stats_path.exists():
        raise FileNotFoundError(
            f"Statistics file not found: {stats_path}"
        )

    stats = np.load(stats_path)

    return {
        "mean": stats["mean"],
        "std": stats["std"],
        "normalized": stats["normalized"],
    }


def load_xtraffic_dataset(
    data_root: str | Path,
    dataset_name: str,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[
    XTrafficIncidentDataset,
    XTrafficIncidentDataset,
    XTrafficIncidentDataset,
    DataLoader,
    DataLoader,
    DataLoader,
]:
    """
    Load train, validation and test datasets.

    Parameters
    ----------
    data_root:
        Root directory containing XTraffic.

    dataset_name:
        Dataset/county name:
        Alameda
        Contra_Costa
        Orange

    batch_size:
        Number of samples per training batch.

    num_workers:
        PyTorch DataLoader worker count.

    Returns
    -------
    train_dataset
    val_dataset
    test_dataset
    train_loader
    val_loader
    test_loader
    """

    data_root = Path(data_root)

    dataset_dir = data_root / dataset_name

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    train_path = dataset_dir / "incident_train.npy"
    val_path = dataset_dir / "incident_val.npy"
    test_path = dataset_dir / "incident_test.npy"

    train_dataset = XTrafficIncidentDataset(train_path)
    val_dataset = XTrafficIncidentDataset(val_path)
    test_dataset = XTrafficIncidentDataset(test_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return (
        train_dataset,
        val_dataset,
        test_dataset,
        train_loader,
        val_loader,
        test_loader,
    )


def inspect_dataset(
    data_root: str | Path,
    dataset_name: str,
) -> None:
    """
    Print basic information about an XTraffic dataset.
    """

    data_root = Path(data_root)
    dataset_dir = data_root / dataset_name

    print("=" * 70)
    print(f"XTraffic Dataset: {dataset_name}")
    print("=" * 70)

    train_dataset = XTrafficIncidentDataset(
        dataset_dir / "incident_train.npy"
    )

    val_dataset = XTrafficIncidentDataset(
        dataset_dir / "incident_val.npy"
    )

    test_dataset = XTrafficIncidentDataset(
        dataset_dir / "incident_test.npy"
    )

    adjacency = load_adjacency(dataset_dir)
    statistics = load_statistics(dataset_dir)

    sample = train_dataset[0]

    print(f"Train samples       : {len(train_dataset)}")
    print(f"Validation samples  : {len(val_dataset)}")
    print(f"Test samples        : {len(test_dataset)}")

    print()
    print(f"x_data shape        : {sample['x_data'].shape}")
    print(f"y_data shape        : {sample['y_data'].shape}")
    print(
        f"incident_position  : "
        f"{sample['incident_position'].item()}"
    )
    print(
        f"incident_distances : "
        f"{sample['incident_distances'].shape}"
    )
    print(
        f"durations shape     : "
        f"{sample['durations'].shape}"
    )

    print()
    print(f"Adjacency shape     : {adjacency.shape}")
    print(f"Statistics keys     : {list(statistics.keys())}")

    print()
    print("Incident features:")
    print(sample["incident_features"])

    print("=" * 70)


if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "XTraffic"

    inspect_dataset(
        data_root=DATA_ROOT,
        dataset_name="Alameda",
    )