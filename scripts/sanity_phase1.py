import os
import sys
import logging
from types import SimpleNamespace

import numpy as np
import torch

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------
# Project imports
# ---------------------------------------------------------
from src.models.phase1.igstgnn.dataloader import load_dataset
from src.models.phase1.igstgnn.graph_algo import normalize_adj_mx
from src.models.phase1.igstgnn.igstgnn import IGSTGNN


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATASET = "Alameda"

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw",
    "XTraffic",
    DATASET
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

NODE_NUM = 521
INPUT_DIM = 3
OUTPUT_DIM = 1
SEQ_LEN = 12
HORIZON = 12

NUM_FEAT = 1
NUM_HIDDEN = 32
NODE_HIDDEN = 12
TIME_EMB_DIM = 12

LAYER = 5
K_T = 3
K_S = 2
GAP = 3
TPD = 288

USE_PRE = True
DY_GRAPH = True
STA_GRAPH = True

ICSF_DIM = 64
LAMBDA_INCIDENT = 1.0
SIGMA_T = 1.0

USE_SENSOR_INFO = False

BATCH_SIZE = 10


# ---------------------------------------------------------
# Simple console logger
# ---------------------------------------------------------
logger = logging.getLogger(
    "pulse_traffic_sanity"
)

logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(message)s")
    )
    logger.addHandler(handler)


def main():

    print()
    print("=" * 70)
    print("PulseTrafficAI - Phase 1 IGSTGNN SANITY TEST")
    print("=" * 70)

    print(f"Dataset : {DATASET}")
    print(f"Data    : {DATA_ROOT}")
    print(f"Device  : {DEVICE}")
    print(f"Batch   : {BATCH_SIZE}")
    print()

    # -----------------------------------------------------
    # 1. Check dataset directory
    # -----------------------------------------------------
    if not os.path.isdir(DATA_ROOT):
        raise FileNotFoundError(
            f"Dataset directory not found:\n{DATA_ROOT}"
        )

    print("[1/7] Dataset directory ........ PASS")

    required_files = [
        "adj_matrix.npy",
        "incident_train.npy",
        "incident_val.npy",
        "incident_test.npy",
        "incident_stats.npz",
        "desc_mapping.json",
        "type_mapping.json",
    ]

    for filename in required_files:
        path = os.path.join(
            DATA_ROOT,
            filename
        )

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required file missing:\n{path}"
            )

    print("[2/7] Required dataset files ... PASS")

    # -----------------------------------------------------
    # 2. Load dataset using the ACTUAL dataloader.py
    # -----------------------------------------------------
    args = SimpleNamespace(
        dataset=DATASET,
        node_num=NODE_NUM,
        input_dim=INPUT_DIM,
        bs=BATCH_SIZE,
        incident=True,
        use_sensor_info=USE_SENSOR_INFO,
    )

    dataloaders, scaler = load_dataset(
        DATA_ROOT,
        args,
        logger
    )

    print("[3/7] Dataset loading .......... PASS")

    # -----------------------------------------------------
    # 3. Inspect one training batch
    # -----------------------------------------------------
    train_loader = dataloaders["train_loader"]

    batch_iterator = train_loader.get_iterator()

    batch = next(batch_iterator)

    if not isinstance(batch, dict):
        raise TypeError(
            "Expected incident training batch to be a dictionary."
        )

    x_data = torch.as_tensor(
        batch["x_data"],
        dtype=torch.float32,
        device=DEVICE
    )

    y_data = torch.as_tensor(
        batch["y_data"],
        dtype=torch.float32,
        device=DEVICE
    )

    incident_features = torch.as_tensor(
        batch["incident_features"],
        dtype=torch.float32,
        device=DEVICE
    )

    incident_position = torch.as_tensor(
        batch["incident_position"],
        dtype=torch.long,
        device=DEVICE
    )

    incident_distances = torch.as_tensor(
        batch["incident_distances"],
        dtype=torch.float32,
        device=DEVICE
    )

    print()
    print("First batch:")
    print(f"  x_data              : {tuple(x_data.shape)}")
    print(f"  y_data              : {tuple(y_data.shape)}")
    print(
        f"  incident_features   : "
        f"{tuple(incident_features.shape)}"
    )
    print(
        f"  incident_position   : "
        f"{tuple(incident_position.shape)}"
    )
    print(
        f"  incident_distances  : "
        f"{tuple(incident_distances.shape)}"
    )

    expected_x = (
        BATCH_SIZE,
        SEQ_LEN,
        NODE_NUM,
        INPUT_DIM
    )

    expected_y = (
        BATCH_SIZE,
        HORIZON,
        NODE_NUM,
        1
    )

    if tuple(x_data.shape) != expected_x:
        raise RuntimeError(
            f"Unexpected x_data shape. "
            f"Expected {expected_x}, "
            f"got {tuple(x_data.shape)}"
        )

    if tuple(y_data.shape) != expected_y:
        raise RuntimeError(
            f"Unexpected y_data shape. "
            f"Expected {expected_y}, "
            f"got {tuple(y_data.shape)}"
        )

    print("[4/7] Batch shapes ............. PASS")

    # -----------------------------------------------------
    # 4. Load and normalize adjacency matrix
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # This follows the reference IGSTGNN implementation.
    #
    # Reference flow:
    #
    #   adj_matrix.npy
    #          |
    #          v
    #   normalize_adj_mx(..., "doubletransition")
    #          |
    #          +---- forward transition matrix
    #          |
    #          +---- backward transition matrix
    #
    # The model therefore receives TWO adjacency matrices.
    # -----------------------------------------------------
    adj_path = os.path.join(
        DATA_ROOT,
        "adj_matrix.npy"
    )

    adj_numpy = np.load(
        adj_path
    )

    if adj_numpy.shape != (
        NODE_NUM,
        NODE_NUM
    ):
        raise RuntimeError(
            "Unexpected raw adjacency shape: "
            f"{adj_numpy.shape}"
        )

    # Match the reference implementation exactly:
    #
    # adj_mx = load_adj_from_numpy(adj_path)
    # adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
    #
    normalized_adjs = normalize_adj_mx(
        adj_numpy,
        "doubletransition"
    )

    if not isinstance(normalized_adjs, list):
        raise RuntimeError(
            "Expected normalized adjacency output "
            "to be a list."
        )

    if len(normalized_adjs) != 2:
        raise RuntimeError(
            "Expected doubletransition normalization "
            "to produce exactly 2 adjacency matrices, "
            f"but got {len(normalized_adjs)}."
        )

    adjs = []

    for index, adjacency in enumerate(normalized_adjs):

        adjacency = np.asarray(
            adjacency,
            dtype=np.float32
        )

        if adjacency.shape != (
            NODE_NUM,
            NODE_NUM
        ):
            raise RuntimeError(
                f"Unexpected normalized adjacency shape "
                f"for graph {index}: {adjacency.shape}"
            )

        adjacency_tensor = torch.as_tensor(
            adjacency,
            dtype=torch.float32,
            device=DEVICE
        )

        if not torch.isfinite(
            adjacency_tensor
        ).all():
            raise RuntimeError(
                f"Normalized adjacency {index} "
                "contains NaN or Inf values."
            )

        adjs.append(adjacency_tensor)

    print()
    print(
        f"Raw adjacency shape       : "
        f"{tuple(adj_numpy.shape)}"
    )

    print(
        f"Normalized adjacency count: "
        f"{len(adjs)}"
    )

    for index, adjacency_tensor in enumerate(adjs):
        print(
            f"  adjacency[{index}] shape  : "
            f"{tuple(adjacency_tensor.shape)}"
        )

    print(
        f"  adjacency[0] device      : "
        f"{adjs[0].device}"
    )

    print(
        f"  adjacency[1] device      : "
        f"{adjs[1].device}"
    )

    print(
        "[5/7] Adjacency loading and "
        "normalization ............ PASS"
    )

    # -----------------------------------------------------
    # 5. Build model
    # -----------------------------------------------------
    model_args = {
        "dataset": DATASET,
        "data_path": DATA_ROOT,

        "node_num": NODE_NUM,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,

        "num_feat": NUM_FEAT,
        "num_hidden": NUM_HIDDEN,
        "node_hidden": NODE_HIDDEN,
        "time_emb_dim": TIME_EMB_DIM,

        "layer": LAYER,
        "k_t": K_T,
        "k_s": K_S,
        "gap": GAP,
        "tpd": TPD,

        "use_pre": USE_PRE,
        "dy_graph": DY_GRAPH,
        "sta_graph": STA_GRAPH,

        "icsf_dim": ICSF_DIM,
        "lambda_incident": LAMBDA_INCIDENT,
        "sigma_t": SIGMA_T,

        "use_sensor_info": USE_SENSOR_INFO,

        "dropout": 0.1,

        # Reference-compatible adjacency list.
        "adjs": adjs,
    }

    model = IGSTGNN(
        model_args=model_args,
        adjs=adjs
    ).to(DEVICE)

    model.eval()

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print()
    print(
        f"Model parameters      : "
        f"{parameter_count:,}"
    )

    print("[6/7] Model creation ........... PASS")

    # -----------------------------------------------------
    # 6. Prepare incident data
    # -----------------------------------------------------
    incident_data = {
        "incident": incident_features,
        "position": incident_position,
        "distances": incident_distances,
    }

    # -----------------------------------------------------
    # 7. Forward pass
    # -----------------------------------------------------
    print()
    print("Running forward pass...")

    with torch.no_grad():

        prediction = model(
            x_data,
            label=y_data,
            incident_data=incident_data,
            sensor_data=None,
        )

    print(
        f"Prediction shape      : "
        f"{tuple(prediction.shape)}"
    )

    expected_prediction_shape = (
        BATCH_SIZE,
        HORIZON,
        NODE_NUM,
        1
    )

    if tuple(prediction.shape) != expected_prediction_shape:
        raise RuntimeError(
            "Unexpected prediction shape. "
            f"Expected {expected_prediction_shape}, "
            f"got {tuple(prediction.shape)}"
        )

    if not torch.isfinite(
        prediction
    ).all():
        raise RuntimeError(
            "Prediction contains NaN or Inf values."
        )

    print("[7/7] Forward pass ............. PASS")

    # -----------------------------------------------------
    # Final result
    # -----------------------------------------------------
    print()
    print("=" * 70)
    print("SANITY TEST PASSED")
    print("=" * 70)
    print()
    print("Verified:")
    print("  Dataset loading")
    print("  Incident batch creation")
    print("  Input tensor shapes")
    print("  Raw adjacency loading")
    print("  Double-transition adjacency normalization")
    print("  Forward/backward adjacency creation")
    print("  IGSTGNN model construction")
    print("  Model parameter initialization")
    print("  Incident data flow")
    print("  Forward propagation")
    print("  Prediction tensor shape")
    print("  NaN/Inf check")
    print()
    print("No training was performed.")
    print("=" * 70)


if __name__ == "__main__":
    main()