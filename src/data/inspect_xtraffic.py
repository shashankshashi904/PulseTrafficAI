from pathlib import Path
import json
import numpy as np
import pandas as pd


# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# XTraffic raw-data directory
XTRAFFIC_DIR = PROJECT_ROOT / "data" / "raw" / "XTraffic"

DATASETS = [
    "Alameda",
    "Contra_Costa",
    "Orange",
]


def inspect_dataset(dataset_name):
    dataset_dir = XTRAFFIC_DIR / dataset_name

    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        return

    # ------------------------------------------------------------------
    # 1. List files
    # ------------------------------------------------------------------
    print("\n[1] Files")

    for file in sorted(dataset_dir.iterdir()):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  {file.name:<25} {size_mb:.2f} MB")

    # ------------------------------------------------------------------
    # 2. Adjacency matrix
    # ------------------------------------------------------------------
    adjacency_path = dataset_dir / "adj_matrix.npy"

    print("\n[2] Adjacency Matrix")

    if adjacency_path.exists():
        adjacency = np.load(adjacency_path)

        print(f"  Shape       : {adjacency.shape}")
        print(f"  Data type   : {adjacency.dtype}")
        print(f"  Min         : {np.min(adjacency)}")
        print(f"  Max         : {np.max(adjacency)}")
        print(f"  Non-zero    : {np.count_nonzero(adjacency)}")
    else:
        print("  [MISSING] adj_matrix.npy")

    # ------------------------------------------------------------------
    # 3. Sensors
    # ------------------------------------------------------------------
    sensors_path = dataset_dir / "sensors.csv"

    print("\n[3] Sensors")

    if sensors_path.exists():
        sensors = pd.read_csv(sensors_path)

        print(f"  Shape       : {sensors.shape}")
        print(f"  Columns     : {list(sensors.columns)}")
        print("\n  First 5 rows:")
        print(sensors.head().to_string(index=False))
    else:
        print("  [MISSING] sensors.csv")

    # ------------------------------------------------------------------
    # 4. Incident data
    # ------------------------------------------------------------------
    incident_path = dataset_dir / "incident_all.npy"

    print("\n[4] Incident Data")

    if incident_path.exists():
        incidents = np.load(
            incident_path,
            allow_pickle=True
        )

        print(f"  Shape       : {incidents.shape}")
        print(f"  Data type   : {incidents.dtype}")

        # Display a small sample without dumping the entire file
        print("\n  Sample:")
        print(incidents[:2])
    else:
        print("  [MISSING] incident_all.npy")

    # ------------------------------------------------------------------
    # 5. Incident statistics
    # ------------------------------------------------------------------
    stats_path = dataset_dir / "incident_stats.npz"

    print("\n[5] Incident Statistics")

    if stats_path.exists():
        stats = np.load(stats_path)

        print(f"  Keys        : {stats.files}")

        for key in stats.files:
            value = stats[key]

            print(
                f"  {key:<20} "
                f"shape={value.shape}, "
                f"dtype={value.dtype}"
            )
    else:
        print("  [MISSING] incident_stats.npz")

    # ------------------------------------------------------------------
    # 6. Description mapping
    # ------------------------------------------------------------------
    desc_path = dataset_dir / "desc_mapping.json"

    print("\n[6] Description Mapping")

    if desc_path.exists():
        with open(desc_path, "r", encoding="utf-8") as file:
            desc_mapping = json.load(file)

        print(f"  Number of entries: {len(desc_mapping)}")

        sample_items = list(desc_mapping.items())[:5]

        print("  Sample:")
        for key, value in sample_items:
            print(f"    {key}: {value}")
    else:
        print("  [MISSING] desc_mapping.json")

    # ------------------------------------------------------------------
    # 7. Type mapping
    # ------------------------------------------------------------------
    type_path = dataset_dir / "type_mapping.json"

    print("\n[7] Incident Type Mapping")

    if type_path.exists():
        with open(type_path, "r", encoding="utf-8") as file:
            type_mapping = json.load(file)

        print(f"  Number of entries: {len(type_mapping)}")

        sample_items = list(type_mapping.items())[:10]

        print("  Sample:")
        for key, value in sample_items:
            print(f"    {key}: {value}")
    else:
        print("  [MISSING] type_mapping.json")


def main():
    print("=" * 80)
    print("PulseTrafficAI - XTraffic Dataset Inspection")
    print("=" * 80)

    print(f"\nProject root : {PROJECT_ROOT}")
    print(f"XTraffic dir : {XTRAFFIC_DIR}")

    for dataset_name in DATASETS:
        inspect_dataset(dataset_name)

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()