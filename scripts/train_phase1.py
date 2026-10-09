import os
import sys
import random
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


# ---------------------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------

from src.models.phase1.igstgnn.igstgnn import IGSTGNN
from src.models.phase1.igstgnn.igstgnn_engine import IGSTGNN_Engine
from src.models.phase1.igstgnn.metrics import masked_mae
from src.models.phase1.igstgnn.dataloader import load_dataset
from src.models.phase1.igstgnn.graph_algo import normalize_adj_mx


# =====================================================================
# USER CONTROLLED PARAMETERS
# =====================================================================

# ---------------------------------------------------------------------
# 1. SELECT EXPERIMENT
# ---------------------------------------------------------------------

# Choose ONLY ONE:
#
#   "baseline_with_incident"
#   "baseline_no_incident"
#
# You do NOT need to manually change incident=True/False elsewhere.

EXPERIMENT = "baseline_with_incident"


# ---------------------------------------------------------------------
# 2. TOTAL TARGET EPOCH
# ---------------------------------------------------------------------
#
# First training session:
#
#     MAX_EPOCHS = 10
#
# Second training session:
#
#     MAX_EPOCHS = 20
#
# Third training session:
#
#     MAX_EPOCHS = 30
#
# The checkpoint automatically determines the starting epoch.
#
# Example:
#
#     last checkpoint = Epoch 10
#     MAX_EPOCHS = 20
#
# Then training runs:
#
#     Epoch 11 -> Epoch 20

MAX_EPOCHS = 25


# ---------------------------------------------------------------------
# 3. RESUME TRAINING
# ---------------------------------------------------------------------
#
# True:
#     Automatically use an existing checkpoint if available.
#
# False:
#     Ignore existing checkpoints and start a new model training.
#
# Keep this TRUE for normal training.

RESUME = True


# ---------------------------------------------------------------------
# 4. VALIDATION METRIC USED FOR BEST CHECKPOINT
# ---------------------------------------------------------------------
#
# Phase-1 reproduction:
#
#     "val_loss"
#
# Supported:
#
#     "val_loss"
#     "val_mape"
#     "val_rmse"
#
# We keep val_loss for the baseline reproduction.

MONITOR_METRIC = "val_loss"


# =====================================================================
# AUTOMATIC EXPERIMENT SETTINGS
# =====================================================================

if EXPERIMENT == "baseline_with_incident":

    INCIDENT = True

elif EXPERIMENT == "baseline_no_incident":

    INCIDENT = False

else:

    raise ValueError(
        "\nInvalid EXPERIMENT.\n"
        "Use either:\n"
        "  baseline_with_incident\n"
        "or\n"
        "  baseline_no_incident"
    )


# =====================================================================
# MODEL OUTPUT DIRECTORY
# =====================================================================

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "phase1"
    / EXPERIMENT
)


# =====================================================================
# LOG DIRECTORY
# =====================================================================

LOG_ROOT = (
    PROJECT_ROOT
    / "experiments"
    / "phase1"
    / "logs"
)


# =====================================================================
# RANDOM SEED
# =====================================================================

SEED = 2025


def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    # Reproducibility settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)


# =====================================================================
# DEVICE
# =====================================================================

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)


# =====================================================================
# SIMPLE LOGGER
# =====================================================================

logger = logging.getLogger(
    "pulse_traffic_phase1_training"
)

logger.setLevel(logging.INFO)

if not logger.handlers:

    console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setFormatter(
        logging.Formatter("%(message)s")
    )

    logger.addHandler(console_handler)


# =====================================================================
# TRAINING INFORMATION
# =====================================================================

print("\n" + "=" * 70)
print("PulseTrafficAI - Phase 1 IGSTGNN Training")
print("=" * 70)

print(
    f"Experiment       : {EXPERIMENT}"
)

print(
    f"Incident         : {INCIDENT}"
)

print(
    f"Device           : {DEVICE}"
)

print(
    f"Seed             : {SEED}"
)

print(
    f"Max Epochs       : {MAX_EPOCHS}"
)

print(
    f"Resume           : {RESUME}"
)

print(
    f"Monitor Metric   : {MONITOR_METRIC}"
)

print(
    f"Model Root       : {MODEL_ROOT}"
)

print("=" * 70)


# =====================================================================
# DATASET
# =====================================================================

DATASET = "Alameda"


DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "XTraffic"
)


# =====================================================================
# DATASET / MODEL PARAMETERS
# =====================================================================

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


# =====================================================================
# TRAINING PARAMETERS
# =====================================================================

# Keep batch size = 10.
#
# Larger batch sizes previously caused GPU memory problems.

BATCH_SIZE = 10

LEARNING_RATE = 0.002

WEIGHT_DECAY = 1e-5

DROPOUT = 0.1

CLIP_GRAD_VALUE = 5

PATIENCE = 5


# =====================================================================
# SCHEDULER
# =====================================================================

SCHEDULER_MILESTONES = [
    1,
    38,
    46,
    54,
    62,
    70,
    80,
]

SCHEDULER_GAMMA = 0.5


# =====================================================================
# CURRICULUM LEARNING
# =====================================================================

CL_EPOCH = 0

WARM_EPOCH = 0


# =====================================================================
# CREATE OUTPUT DIRECTORIES
# =====================================================================

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# DATASET ARGUMENTS
# =====================================================================
#
# The actual PulseTrafficAI dataloader.py exposes:
#
#     load_dataset(data_path, args, logger)
#
# It does NOT expose a DataLoader constructor.
#
# These arguments match the working sanity_phase1.py data-loading
# configuration.

data_loader_args = SimpleNamespace(
    dataset=DATASET,
    node_num=NODE_NUM,
    input_dim=INPUT_DIM,
    bs=BATCH_SIZE,
    incident=INCIDENT,
    use_sensor_info=USE_SENSOR_INFO,
)


# =====================================================================
# LOAD DATASET
# =====================================================================

print("\nLoading XTraffic dataset...")

dataloaders, scaler = load_dataset(
    str(DATA_ROOT / DATASET),
    data_loader_args,
    logger,
)

train_loader = dataloaders["train_loader"]

val_loader = dataloaders["val_loader"]

test_loader = dataloaders["test_loader"]


print("\nDataset loaders created successfully.")

print(
    f"Train batches : {train_loader.num_batch}"
)

print(
    f"Val batches   : {val_loader.num_batch}"
)

print(
    f"Test batches  : {test_loader.num_batch}"
)


# =====================================================================
# LOAD ADJACENCY MATRIX
# =====================================================================
#
# IMPORTANT:
#
# This follows the reference IGSTGNN implementation.
#
# Reference:
#
#     adj_mx = load_adj_from_numpy(adj_path)
#     adj_mx = normalize_adj_mx(adj_mx, 'doubletransition')
#     args.adjs = [torch.tensor(i).to(device) for i in adj_mx]
#
# doubletransition creates:
#
#     adjs[0] = forward transition graph
#     adjs[1] = backward transition graph
#
# The Mask component of IGSTGNN expects these two graphs.

adj_path = (
    DATA_ROOT
    / DATASET
    / "adj_matrix.npy"
)


if not adj_path.exists():

    raise FileNotFoundError(
        f"\nAdjacency matrix not found:\n{adj_path}"
    )


adj_matrix = np.load(
    adj_path
)


if adj_matrix.shape != (
    NODE_NUM,
    NODE_NUM
):

    raise RuntimeError(
        "Unexpected adjacency shape: "
        f"{adj_matrix.shape}. "
        f"Expected ({NODE_NUM}, {NODE_NUM})."
    )


print("\nDataset information")
print("-" * 70)

print(
    f"Dataset          : {DATASET}"
)

print(
    f"Adjacency shape  : {adj_matrix.shape}"
)

print(
    f"Node count       : {NODE_NUM}"
)


# =====================================================================
# NORMALIZE ADJACENCY
# =====================================================================

print("\nNormalizing adjacency using doubletransition...")


normalized_adjs = normalize_adj_mx(
    adj_matrix,
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
        f"but received {len(normalized_adjs)}."
    )


# =====================================================================
# CONVERT NORMALIZED ADJACENCIES TO TENSORS
# =====================================================================

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


    adjs.append(
        adjacency_tensor
    )


print(
    f"Normalized adjacency count : {len(adjs)}"
)

print(
    f"Forward adjacency shape    : "
    f"{tuple(adjs[0].shape)}"
)

print(
    f"Backward adjacency shape   : "
    f"{tuple(adjs[1].shape)}"
)

print(
    f"Adjacency device            : "
    f"{adjs[0].device}"
)


# =====================================================================
# MODEL ARGUMENTS
# =====================================================================

model_args = {

    "dataset": DATASET,

    "data_path": str(
        DATA_ROOT / DATASET
    ),

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

    "incident": INCIDENT,

    "icsf_dim": ICSF_DIM,

    "lambda_incident": LAMBDA_INCIDENT,

    "sigma_t": SIGMA_T,

    "use_sensor_info": USE_SENSOR_INFO,

    "node_num": NODE_NUM,

    # Reference-compatible adjacency list.
    "adjs": adjs,

    "dropout": DROPOUT,
}


# =====================================================================
# SERIALIZABLE CHECKPOINT CONFIG
# =====================================================================
#
# Do NOT store CUDA tensors such as adjacency matrices in checkpoint
# metadata.
#
# This dictionary contains only simple values that can safely be saved.

checkpoint_model_config = {

    "dataset": DATASET,

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

    "incident": INCIDENT,

    "icsf_dim": ICSF_DIM,

    "lambda_incident": LAMBDA_INCIDENT,

    "sigma_t": SIGMA_T,

    "use_sensor_info": USE_SENSOR_INFO,

    "batch_size": BATCH_SIZE,

    "learning_rate": LEARNING_RATE,

    "weight_decay": WEIGHT_DECAY,

    "dropout": DROPOUT,

    "clip_grad_value": CLIP_GRAD_VALUE,

    "patience": PATIENCE,

    "scheduler_milestones": SCHEDULER_MILESTONES,

    "scheduler_gamma": SCHEDULER_GAMMA,

    "cl_epoch": CL_EPOCH,

    "warm_epoch": WARM_EPOCH,

    # Important for explaining the graph configuration later.
    "adjacency_type": "doubletransition",

    "adjacency_count": 2,
}


# =====================================================================
# MODEL
# =====================================================================

print("\nCreating IGSTGNN model...")


model = IGSTGNN(
    model_args=model_args,
    adjs=adjs,
)


model = model.to(
    DEVICE
)


# =====================================================================
# MODEL PARAMETER COUNT
# =====================================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)


trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    f"Total parameters     : {total_parameters:,}"
)

print(
    f"Trainable parameters : {trainable_parameters:,}"
)


# =====================================================================
# OPTIMIZER
# =====================================================================

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    eps=1e-8,
)


# =====================================================================
# LOSS
# =====================================================================

loss_fn = masked_mae


# =====================================================================
# SCHEDULER
# =====================================================================

scheduler = MultiStepLR(
    optimizer,
    milestones=SCHEDULER_MILESTONES,
    gamma=SCHEDULER_GAMMA,
)


# =====================================================================
# CURRICULUM LEARNING STEPS
# =====================================================================

cl_step = (
    CL_EPOCH
    * train_loader.num_batch
)


warm_step = (
    WARM_EPOCH
    * train_loader.num_batch
)


# =====================================================================
# ENGINE
# =====================================================================

print("\nCreating training engine...")


engine = IGSTGNN_Engine(

    model=model,

    dataloader={
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
    },

    optimizer=optimizer,

    loss_fn=loss_fn,

    scaler=scaler,

    max_epochs=MAX_EPOCHS,

    patience=PATIENCE,

    scheduler=scheduler,

    seed=SEED,

    log_dir=str(MODEL_ROOT),

    clip_grad_value=CLIP_GRAD_VALUE,

    null_val=0.0,

    cl_step=cl_step,

    warm_step=warm_step,

    device=DEVICE,

    incident=INCIDENT,

    horizon=HORIZON,

    # ---------------------------------------------------------------
    # Checkpoint settings
    # ---------------------------------------------------------------

    experiment_name=EXPERIMENT,

    model_config=checkpoint_model_config,

    resume=RESUME,

    monitor_metric=MONITOR_METRIC,
)


# =====================================================================
# START TRAINING
# =====================================================================

print("\nStarting training...")


engine.train()


# =====================================================================
# COMPLETION
# =====================================================================

print("\n" + "=" * 70)

print("TRAINING COMPLETED")

print("=" * 70)


print(
    f"Experiment : {EXPERIMENT}"
)

print(
    f"Incident   : {INCIDENT}"
)

print(
    f"Model root : {MODEL_ROOT}"
)

print("=" * 70)