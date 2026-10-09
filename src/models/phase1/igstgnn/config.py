"""
PulseTrafficAI - Phase 1 IGSTGNN Configuration

Reference configuration derived from the official IGSTGNN
implementation in temp_IGSTGNN.

Do not modify the reference implementation.
"""

from dataclasses import dataclass


@dataclass
class IGSTGNNConfig:
    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    dataset: str = "Alameda"

    # ---------------------------------------------------------
    # Hardware
    # ---------------------------------------------------------
    device: str = "cuda"

    # ---------------------------------------------------------
    # Reproducibility
    # ---------------------------------------------------------
    seed: int = 2025

    # ---------------------------------------------------------
    # Data dimensions
    # ---------------------------------------------------------
    input_dim: int = 3
    output_dim: int = 1

    # Historical input sequence
    seq_len: int = 12

    # Future prediction horizon
    horizon: int = 12

    # Actual traffic features used by the model
    num_feat: int = 1

    # ---------------------------------------------------------
    # IGSTGNN architecture
    # ---------------------------------------------------------
    num_hidden: int = 32
    node_hidden: int = 12
    time_emb_dim: int = 12

    # Number of DecoupleLayers
    layer: int = 5

    # Temporal graph kernel
    k_t: int = 3

    # Spatial graph order
    k_s: int = 2

    # Forecasting gap
    gap: int = 3

    # Time periods per day
    # 24 hours × 12 five-minute intervals = 288
    tpd: int = 288

    # ---------------------------------------------------------
    # Graph configuration
    # ---------------------------------------------------------
    use_pre: bool = True
    dy_graph: bool = True
    sta_graph: bool = True

    # ---------------------------------------------------------
    # Incident configuration
    # ---------------------------------------------------------
    incident: bool = True

    # Incident Context Spatial Fusion
    icsf_dim: int = 64

    # Temporal Incident Impact Decay
    lambda_incident: float = 1.0
    sigma_t: float = 1.0

    # ---------------------------------------------------------
    # Sensor information
    # ---------------------------------------------------------
    use_sensor_info: bool = False

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 20

    learning_rate: float = 2e-3
    weight_decay: float = 1e-5

    dropout: float = 0.1
    clip_grad_value: float = 5.0

    # Curriculum learning
    cl_epoch: int = 3
    warm_epoch: int = 30

    # ---------------------------------------------------------
    # Optimizer scheduler
    # ---------------------------------------------------------
    scheduler_milestones: tuple = (
        1,
        38,
        46,
        54,
        62,
        70,
        80,
    )

    scheduler_gamma: float = 0.5


def get_config() -> IGSTGNNConfig:
    """
    Return the Phase 1 IGSTGNN configuration.
    """
    return IGSTGNNConfig()