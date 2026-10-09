import logging
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


# =====================================================================
# PROJECT PATH
# =====================================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# =====================================================================
# IMPORTS
# =====================================================================

from src.models.phase1.igstgnn.igstgnn import (
    IGSTGNN,
)

from src.models.phase1.igstgnn.igstgnn_engine import (
    IGSTGNN_Engine,
)

from src.models.phase1.igstgnn.metrics import (
    masked_mae,
)

from src.models.phase1.igstgnn.dataloader import (
    load_dataset,
)

from src.models.phase1.igstgnn.graph_algo import (
    normalize_adj_mx,
)


# =====================================================================
# USER CONTROLLED EXPERIMENT
# =====================================================================

# -------------------------------------------------------------
# Select which trained model to evaluate.
#
# Options:
#
#   baseline_with_incident
#   baseline_no_incident
# -------------------------------------------------------------

EXPERIMENT = "baseline_with_incident"


# =====================================================================
# AUTOMATIC INCIDENT SETTING
# =====================================================================

if (
    EXPERIMENT
    == "baseline_with_incident"
):

    INCIDENT = True

elif (
    EXPERIMENT
    == "baseline_no_incident"
):

    INCIDENT = False

else:

    raise ValueError(
        f"Unknown experiment: "
        f"{EXPERIMENT}"
    )


# =====================================================================
# DATASET / MODEL CONFIGURATION
# =====================================================================

DATASET = "Alameda"

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

LEARNING_RATE = 0.002

WEIGHT_DECAY = 1e-5

DROPOUT = 0.1

CLIP_GRAD_VALUE = 5

PATIENCE = 5

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

CL_EPOCH = 0

WARM_EPOCH = 0

SEED = 2025


# =====================================================================
# PATHS
# =====================================================================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "XTraffic"
)

DATASET_ROOT = (
    DATA_ROOT
    / DATASET
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "models"
    / "phase1"
    / EXPERIMENT
)


# =====================================================================
# RANDOM SEED
# =====================================================================

def set_seed(
    seed
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed(
            seed
        )

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# =====================================================================
# LOGGER
# =====================================================================

def create_logger():

    logger = logging.getLogger(
        f"PulseTrafficAI_Evaluation_{EXPERIMENT}"
    )

    logger.setLevel(
        logging.INFO
    )

    if not logger.handlers:

        console_handler = (
            logging.StreamHandler()
        )

        formatter = logging.Formatter(
            "%(asctime)s - "
            "%(levelname)s - "
            "%(message)s"
        )

        console_handler.setFormatter(
            formatter
        )

        logger.addHandler(
            console_handler
        )

    return logger


# =====================================================================
# MAIN
# =====================================================================

def main():

    logger = create_logger()

    set_seed(
        SEED
    )

    # -------------------------------------------------------------
    # Device
    # -------------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # -------------------------------------------------------------
    # Check paths
    # -------------------------------------------------------------

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"XTraffic root not found:\n"
            f"{DATA_ROOT}"
        )

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATASET_ROOT}"
        )

    # -------------------------------------------------------------
    # Check model files
    # -------------------------------------------------------------

    final_model_path = (
        MODEL_ROOT
        / f"final_model_s{SEED}.pt"
    )

    best_checkpoint_path = (
        MODEL_ROOT
        / f"best_checkpoint_s{SEED}.pt"
    )

    last_checkpoint_path = (
        MODEL_ROOT
        / f"last_checkpoint_s{SEED}.pt"
    )

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PulseTrafficAI - "
        "Phase 1 IGSTGNN TEST EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"Experiment    : "
        f"{EXPERIMENT}"
    )

    print(
        f"Incident      : "
        f"{INCIDENT}"
    )

    print(
        f"Dataset       : "
        f"{DATASET}"
    )

    print(
        f"Data path     : "
        f"{DATASET_ROOT}"
    )

    print(
        f"Model path    : "
        f"{MODEL_ROOT}"
    )

    print(
        f"Device        : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Nodes         : "
        f"{NODE_NUM}"
    )

    print(
        f"Input         : "
        f"{SEQ_LEN} steps"
    )

    print(
        f"Horizon       : "
        f"{HORIZON} steps"
    )

    print(
        "=" * 70
    )

    # -------------------------------------------------------------
    # Verify model availability
    # -------------------------------------------------------------

    print(
        "\nCheckpoint information"
    )

    print(
        "-" * 70
    )

    print(
        f"Best checkpoint : "
        f"{best_checkpoint_path}"
    )

    print(
        f"Last checkpoint : "
        f"{last_checkpoint_path}"
    )

    print(
        f"Final model     : "
        f"{final_model_path}"
    )

    if (
        not final_model_path.exists()
        and not best_checkpoint_path.exists()
    ):

        raise FileNotFoundError(
            "\nNo trained model found.\n"
            "Train the selected experiment first."
        )

    # -------------------------------------------------------------
    # Dataset loader arguments
    #
    # This matches train_phase1.py and sanity_phase1.py.
    # -------------------------------------------------------------

    data_loader_args = SimpleNamespace(
        dataset=DATASET,
        node_num=NODE_NUM,
        input_dim=INPUT_DIM,
        bs=BATCH_SIZE,
        incident=INCIDENT,
        use_sensor_info=USE_SENSOR_INFO,
    )

    # -------------------------------------------------------------
    # Load XTraffic data
    # -------------------------------------------------------------

    print(
        "\nLoading XTraffic data..."
    )

    dataloaders, scaler = load_dataset(
        str(DATASET_ROOT),
        data_loader_args,
        logger,
    )

    train_loader = (
        dataloaders["train_loader"]
    )

    val_loader = (
        dataloaders["val_loader"]
    )

    test_loader = (
        dataloaders["test_loader"]
    )

    print(
        "\nDataset loaded successfully."
    )

    print(
        f"Train batches      : "
        f"{train_loader.num_batch}"
    )

    print(
        f"Validation batches : "
        f"{val_loader.num_batch}"
    )

    print(
        f"Test batches       : "
        f"{test_loader.num_batch}"
    )

    # -------------------------------------------------------------
    # Adjacency
    # -------------------------------------------------------------

    adj_path = (
        DATASET_ROOT
        / "adj_matrix.npy"
    )

    if not adj_path.exists():

        raise FileNotFoundError(
            f"Adjacency matrix not found:\n"
            f"{adj_path}"
        )

    logger.info(
        f"Loading adjacency: "
        f"{adj_path}"
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
            f"Expected "
            f"({NODE_NUM}, {NODE_NUM})."
        )

    print(
        f"\nRaw adjacency shape: "
        f"{adj_matrix.shape}"
    )

    # -------------------------------------------------------------
    # IMPORTANT:
    #
    # Match the reference IGSTGNN implementation and
    # train_phase1.py.
    #
    # doubletransition produces:
    #
    #   adjs[0] = forward transition matrix
    #   adjs[1] = backward transition matrix
    # -------------------------------------------------------------

    normalized_adjs = normalize_adj_mx(
        adj_matrix,
        "doubletransition"
    )

    if not isinstance(
        normalized_adjs,
        list
    ):

        raise RuntimeError(
            "Expected normalized adjacency "
            "output to be a list."
        )

    if len(normalized_adjs) != 2:

        raise RuntimeError(
            "Expected doubletransition "
            "normalization to produce exactly "
            f"2 adjacency matrices, but got "
            f"{len(normalized_adjs)}."
        )

    adjs = []

    for index, adjacency in enumerate(
        normalized_adjs
    ):

        adjacency = np.asarray(
            adjacency,
            dtype=np.float32
        )

        if adjacency.shape != (
            NODE_NUM,
            NODE_NUM
        ):

            raise RuntimeError(
                f"Unexpected normalized adjacency "
                f"shape for graph {index}: "
                f"{adjacency.shape}"
            )

        adjacency_tensor = torch.as_tensor(
            adjacency,
            dtype=torch.float32,
            device=device
        )

        if not torch.isfinite(
            adjacency_tensor
        ).all():

            raise RuntimeError(
                f"Normalized adjacency "
                f"{index} contains NaN or Inf."
            )

        adjs.append(
            adjacency_tensor
        )

    print(
        f"Normalized adjacency count: "
        f"{len(adjs)}"
    )

    print(
        f"Forward adjacency shape   : "
        f"{tuple(adjs[0].shape)}"
    )

    print(
        f"Backward adjacency shape  : "
        f"{tuple(adjs[1].shape)}"
    )

    print(
        f"Adjacency device           : "
        f"{adjs[0].device}"
    )

    # -------------------------------------------------------------
    # Model arguments
    # -------------------------------------------------------------

    model_args = {

        "dataset": DATASET,

        "data_path": str(
            DATASET_ROOT
        ),

        "input_dim":
            INPUT_DIM,

        "output_dim":
            OUTPUT_DIM,

        "seq_len":
            SEQ_LEN,

        "horizon":
            HORIZON,

        "num_feat":
            NUM_FEAT,

        "num_hidden":
            NUM_HIDDEN,

        "node_hidden":
            NODE_HIDDEN,

        "time_emb_dim":
            TIME_EMB_DIM,

        "layer":
            LAYER,

        "k_t":
            K_T,

        "k_s":
            K_S,

        "gap":
            GAP,

        "tpd":
            TPD,

        "use_pre":
            USE_PRE,

        "dy_graph":
            DY_GRAPH,

        "sta_graph":
            STA_GRAPH,

        "incident":
            INCIDENT,

        "icsf_dim":
            ICSF_DIM,

        "lambda_incident":
            LAMBDA_INCIDENT,

        "sigma_t":
            SIGMA_T,

        "use_sensor_info":
            USE_SENSOR_INFO,

        "node_num":
            NODE_NUM,

        "adjs":
            adjs,

        "dropout":
            DROPOUT,
    }

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print(
        "\nCreating IGSTGNN model..."
    )

    model = IGSTGNN(
        model_args=model_args,
        adjs=adjs,
    )

    model = model.to(
        device
    )

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters     : "
        f"{parameter_count:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_parameters:,}"
    )

    # -------------------------------------------------------------
    # Optimizer
    #
    # Evaluation does not train.
    #
    # These objects are created because the existing
    # IGSTGNN_Engine interface expects them.
    # -------------------------------------------------------------

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=SCHEDULER_MILESTONES,
        gamma=SCHEDULER_GAMMA,
    )

    # -------------------------------------------------------------
    # Checkpoint configuration
    # -------------------------------------------------------------

    checkpoint_model_config = {

        "dataset":
            DATASET,

        "node_num":
            NODE_NUM,

        "input_dim":
            INPUT_DIM,

        "output_dim":
            OUTPUT_DIM,

        "seq_len":
            SEQ_LEN,

        "horizon":
            HORIZON,

        "num_feat":
            NUM_FEAT,

        "num_hidden":
            NUM_HIDDEN,

        "node_hidden":
            NODE_HIDDEN,

        "time_emb_dim":
            TIME_EMB_DIM,

        "layer":
            LAYER,

        "k_t":
            K_T,

        "k_s":
            K_S,

        "gap":
            GAP,

        "tpd":
            TPD,

        "use_pre":
            USE_PRE,

        "dy_graph":
            DY_GRAPH,

        "sta_graph":
            STA_GRAPH,

        "incident":
            INCIDENT,

        "icsf_dim":
            ICSF_DIM,

        "lambda_incident":
            LAMBDA_INCIDENT,

        "sigma_t":
            SIGMA_T,

        "use_sensor_info":
            USE_SENSOR_INFO,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "dropout":
            DROPOUT,

        "clip_grad_value":
            CLIP_GRAD_VALUE,

        "patience":
            PATIENCE,

        "scheduler_milestones":
            SCHEDULER_MILESTONES,

        "scheduler_gamma":
            SCHEDULER_GAMMA,

        "cl_epoch":
            CL_EPOCH,

        "warm_epoch":
            WARM_EPOCH,

        "adjacency_type":
            "doubletransition",

        "adjacency_count":
            2,
    }

    # -------------------------------------------------------------
    # Engine
    #
    # Evaluation does NOT resume training.
    # -------------------------------------------------------------

    engine = IGSTGNN_Engine(

        model=model,

        dataloader={
            "train_loader":
                train_loader,

            "val_loader":
                val_loader,

            "test_loader":
                test_loader,
        },

        optimizer=optimizer,

        loss_fn=masked_mae,

        scaler=scaler,

        max_epochs=1,

        patience=PATIENCE,

        scheduler=scheduler,

        seed=SEED,

        log_dir=str(
            MODEL_ROOT
        ),

        clip_grad_value=
            CLIP_GRAD_VALUE,

        null_val=0.0,

        cl_step=0,

        warm_step=0,

        device=device,

        incident=INCIDENT,

        horizon=HORIZON,

        experiment_name=EXPERIMENT,

        model_config=
            checkpoint_model_config,

        resume=False,

        monitor_metric="val_loss",
    )

    # -------------------------------------------------------------
    # Load best checkpoint
    # -------------------------------------------------------------

    loaded_checkpoint = False

    if best_checkpoint_path.exists():

        print(
            "\nBest checkpoint found."
        )

        print(
            "Loading best checkpoint "
            "for evaluation..."
        )

        checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        if (
            isinstance(
                checkpoint,
                dict
            )
            and
            "model_state_dict"
            in checkpoint
        ):

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

            loaded_checkpoint = True

            print(
                f"Best Epoch       : "
                f"{checkpoint.get('epoch')}"
            )

            print(
                f"Best Metric      : "
                f"{checkpoint.get('best_metric')}"
            )

            print(
                f"Validation Loss  : "
                f"{checkpoint.get('val_loss')}"
            )

            print(
                f"Validation RMSE  : "
                f"{checkpoint.get('val_rmse')}"
            )

            print(
                f"Validation MAPE  : "
                f"{checkpoint.get('val_mape')}"
            )

        else:

            raise RuntimeError(
                "Best checkpoint does not contain "
                "'model_state_dict'."
            )

    # -------------------------------------------------------------
    # Fallback to legacy final_model
    # -------------------------------------------------------------

    elif final_model_path.exists():

        print(
            "\nBest checkpoint not found."
        )

        print(
            "Loading final_model instead..."
        )

        checkpoint = torch.load(
            final_model_path,
            map_location=device,
            weights_only=False,
        )

        if isinstance(
            checkpoint,
            dict
        ):

            model.load_state_dict(
                checkpoint
            )

        else:

            raise RuntimeError(
                "Unexpected final model format."
            )

        loaded_checkpoint = True

    if not loaded_checkpoint:

        raise RuntimeError(
            "Unable to load a trained model."
        )

    # -------------------------------------------------------------
    # Test evaluation
    # -------------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Starting TEST evaluation..."
    )

    print(
        "=" * 70
    )

    model.eval()

    # -------------------------------------------------------------
    # Important:
    #
    # IGSTGNN_Engine.evaluate("test") uses the project's
    # established test-evaluation path.
    #
    # The model has already been loaded above.
    # -------------------------------------------------------------

    engine.evaluate(
        "test"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEST evaluation completed."
    )

    print(
        "=" * 70
    )


# =====================================================================
# ENTRY POINT
# =====================================================================

if __name__ == "__main__":

    main()