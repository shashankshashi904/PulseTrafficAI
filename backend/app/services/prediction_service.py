from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from src.models.phase1.igstgnn.igstgnn import IGSTGNN
from src.models.phase1.igstgnn.graph_algo import normalize_adj_mx


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "XTraffic"

MODEL_ROOT = PROJECT_ROOT / "models" / "phase1"

NO_INCIDENT_CHECKPOINT = (
    MODEL_ROOT
    / "baseline_no_incident"
    / "best_checkpoint_s2025.pt"
)

WITH_INCIDENT_CHECKPOINT = (
    MODEL_ROOT
    / "baseline_with_incident"
    / "best_checkpoint_s2025.pt"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SUPPORTED_LOCATIONS = {"Alameda"}

_model_cache = {}


# ============================================================
# DATASET
# ============================================================

def get_dataset_path(location: str) -> Path:

    if location not in SUPPORTED_LOCATIONS:
        raise ValueError(
            f"Unsupported location: {location}. "
            f"Currently supported: {sorted(SUPPORTED_LOCATIONS)}"
        )

    dataset_path = DATA_ROOT / location

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_path}"
        )

    return dataset_path


# ============================================================
# SENSOR METADATA
# ============================================================

def load_sensors(location: str) -> pd.DataFrame:

    dataset_path = get_dataset_path(location)

    sensors_file = dataset_path / "sensors.csv"

    if not sensors_file.exists():
        raise FileNotFoundError(
            f"Sensor metadata not found: {sensors_file}"
        )

    sensors = pd.read_csv(sensors_file)

    if len(sensors) == 0:
        raise RuntimeError(
            "sensors.csv contains no sensors."
        )

    return sensors


# ============================================================
# SENSOR STREET / LOCATION
# ============================================================

def get_sensor_street(
    sensors: pd.DataFrame,
    node: int
) -> str:

    row = sensors.iloc[node]

    name = str(
        row.get("Name", "")
    ).strip()

    if name:
        return name

    freeway = str(
        row.get("Fwy", "")
    ).strip()

    if freeway:
        return freeway

    return f"Sensor {node}"


# ============================================================
# SENSOR / STREET VALIDATION
# ============================================================

def sensor_matches_street(
    sensors: pd.DataFrame,
    node: int,
    street: str
) -> bool:

    if not street:
        return True

    actual_street = get_sensor_street(
        sensors,
        node
    )

    return actual_street == street


# ============================================================
# PREVIOUS / SELECTED / NEXT SENSOR
# ============================================================

def get_neighbor_nodes(
    sensors: pd.DataFrame,
    selected_node: int
):
    """
    Find previous and next sensors on the same roadway.

    XTraffic provides:
        Fwy   -> roadway / route identifier
        order -> sensor ordering along the roadway

    Therefore node-1 and node+1 are not automatically
    treated as physical previous/next sensors.
    """

    selected_row = sensors.iloc[selected_node]

    selected_fwy = str(
        selected_row.get("Fwy", "")
    ).strip()

    # --------------------------------------------------------
    # Fallback if Fwy is unavailable
    # --------------------------------------------------------

    if (
        not selected_fwy
        or selected_fwy.lower() == "nan"
    ):

        total_nodes = len(sensors)

        previous_node = max(
            0,
            selected_node - 1
        )

        next_node = min(
            total_nodes - 1,
            selected_node + 1
        )

        return (
            previous_node,
            selected_node,
            next_node
        )

    # --------------------------------------------------------
    # Find sensors on the same roadway
    # --------------------------------------------------------

    same_road = sensors[
        sensors["Fwy"].astype(str).str.strip()
        == selected_fwy
    ].copy()

    # --------------------------------------------------------
    # Fallback if only one sensor exists
    # --------------------------------------------------------

    if len(same_road) <= 1:

        total_nodes = len(sensors)

        previous_node = max(
            0,
            selected_node - 1
        )

        next_node = min(
            total_nodes - 1,
            selected_node + 1
        )

        return (
            previous_node,
            selected_node,
            next_node
        )

    # --------------------------------------------------------
    # Convert order to numeric
    # --------------------------------------------------------

    same_road["_order_numeric"] = pd.to_numeric(
        same_road["order"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Sort sensors according to road order
    # --------------------------------------------------------

    same_road = same_road.sort_values(
        by="_order_numeric",
        na_position="last"
    )

    node_list = same_road.index.tolist()

    # --------------------------------------------------------
    # Locate selected sensor
    # --------------------------------------------------------

    try:

        selected_position = node_list.index(
            selected_node
        )

    except ValueError:

        return (
            selected_node,
            selected_node,
            selected_node
        )

    # --------------------------------------------------------
    # Previous sensor
    # --------------------------------------------------------

    if selected_position > 0:

        previous_node = int(
            node_list[selected_position - 1]
        )

    else:

        previous_node = selected_node

    # --------------------------------------------------------
    # Next sensor
    # --------------------------------------------------------

    if selected_position < len(node_list) - 1:

        next_node = int(
            node_list[selected_position + 1]
        )

    else:

        next_node = selected_node

    return (
        previous_node,
        selected_node,
        next_node
    )


# ============================================================
# MODEL CONFIGURATION
# ============================================================

def build_model_config(
    location: str,
    dataset_path: Path,
    adjs,
    incident_enabled: bool
):

    return {
        "num_feat": 1,
        "num_hidden": 32,
        "node_hidden": 12,
        "time_emb_dim": 12,
        "layer": 5,
        "k_t": 3,
        "k_s": 2,
        "gap": 3,
        "tpd": 288,
        "use_pre": True,
        "dy_graph": True,
        "sta_graph": True,
        "incident": incident_enabled,
        "icsf_dim": 64,
        "lambda_incident": 1.0,
        "sigma_t": 1.0,
        "use_sensor_info": False,
        "dropout": 0.1,
        "dataset": location,
        "data_path": str(dataset_path),
        "adjs": adjs,
    }


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def _load_state_dict(
    checkpoint: Path
):

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}"
        )

    state = torch.load(
        checkpoint,
        map_location=DEVICE,
        weights_only=True
    )

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format in {checkpoint}. "
            f"Expected a model state_dict or full checkpoint."
        )

    # The Phase 1 best/last checkpoints are full training
    # checkpoints. Extract only the model weights for inference.
    if "model_state_dict" in state:
        state = state["model_state_dict"]

    if not isinstance(state, dict):
        raise RuntimeError(
            f"Invalid model_state_dict in checkpoint: {checkpoint}"
        )

    return state


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    location: str,
    incident_enabled: bool
):

    cache_key = (
        location,
        incident_enabled
    )

    if cache_key in _model_cache:
        return _model_cache[cache_key]

    dataset_path = get_dataset_path(
        location
    )

    checkpoint = (
        WITH_INCIDENT_CHECKPOINT
        if incident_enabled
        else NO_INCIDENT_CHECKPOINT
    )

    adj_np = np.load(
        dataset_path / "adj_matrix.npy",
        allow_pickle=True
    )

    adjs_np = normalize_adj_mx(
        adj_np,
        "doubletransition"
    )

    adjs = [
        torch.as_tensor(
            adj,
            dtype=torch.float32,
            device=DEVICE
        )
        for adj in adjs_np
    ]

    model_args = build_model_config(
        location,
        dataset_path,
        adjs,
        incident_enabled
    )

    model = IGSTGNN(
        model_args=model_args,
        node_num=adj_np.shape[0],
        input_dim=3,
        output_dim=1,
        seq_len=12,
        horizon=12
    )

    model.load_state_dict(
        _load_state_dict(checkpoint)
    )

    model = model.to(DEVICE)

    model.eval()

    stats = np.load(
        dataset_path / "incident_stats.npz",
        allow_pickle=True
    )

    model_info = {
        "model": model,
        "checkpoint": checkpoint,
        "stats": stats,
    }

    _model_cache[cache_key] = model_info

    return model_info


# ============================================================
# LOAD DATASET SAMPLE
# ============================================================

def load_dataset_sample(
    location: str,
    index: int = 0
):

    dataset_path = get_dataset_path(
        location
    )

    test_file = (
        dataset_path
        / "incident_test.npy"
    )

    if not test_file.exists():
        raise FileNotFoundError(
            f"Test split not found: {test_file}"
        )

    samples = np.load(
        test_file,
        allow_pickle=True
    )

    if len(samples) == 0:
        raise ValueError(
            f"No samples found in {test_file}"
        )

    if (
        index < 0
        or index >= len(samples)
    ):
        raise IndexError(
            f"Sample index {index} is outside "
            f"the dataset range 0-{len(samples) - 1}"
        )

    return samples[index]


# ============================================================
# PREPARE HISTORY
# ============================================================

def prepare_history(sample):

    x_data = np.asarray(
        sample["x_data"],
        dtype=np.float32
    )

    if (
        x_data.shape[0] != 12
        or x_data.shape[2] != 3
    ):
        raise RuntimeError(
            f"Unexpected XTraffic history shape: "
            f"{x_data.shape}; expected [12, nodes, 3]."
        )

    return (
        torch.from_numpy(x_data)
        .unsqueeze(0)
        .to(DEVICE)
    )


# ============================================================
# PREPARE INCIDENT DATA
# ============================================================

def prepare_incident_data(sample):

    incident_features = sample[
        "incident_features"
    ]

    incident_vector = np.array(
        [
            incident_features["Incident Time"],
            incident_features["Description"],
            incident_features["Type"],
            incident_features["Holiday"],
        ],
        dtype=np.float32
    )

    position = np.asarray(
        sample["incident_position"],
        dtype=np.int64
    ).reshape(())

    distances = np.asarray(
        sample["incident_distances"],
        dtype=np.float32
    )

    return {
        "incident": (
            torch.from_numpy(
                incident_vector
            )
            .unsqueeze(0)
            .to(DEVICE)
        ),

        "position": (
            torch.from_numpy(
                np.asarray(
                    position,
                    dtype=np.int64
                ).reshape(1)
            )
            .to(DEVICE)
        ),

        "distances": (
            torch.from_numpy(
                distances
            )
            .unsqueeze(0)
            .to(DEVICE)
        ),
    }


# ============================================================
# PREPARE INCIDENT INFORMATION FOR UI
# ============================================================

def prepare_incident_information(
    sample,
    dataset_path: Path
):

    incident_features = sample[
        "incident_features"
    ]

    # --------------------------------------------------------
    # Load incident type mapping
    # --------------------------------------------------------

    with open(
        dataset_path / "type_mapping.json",
        "r",
        encoding="utf-8"
    ) as file:

        type_mapping = json.load(file)

    # --------------------------------------------------------
    # Load incident description mapping
    # --------------------------------------------------------

    with open(
        dataset_path / "desc_mapping.json",
        "r",
        encoding="utf-8"
    ) as file:

        desc_mapping = json.load(file)

    # --------------------------------------------------------
    # Reverse mappings
    # --------------------------------------------------------

    type_lookup = {
        int(value): key
        for key, value in type_mapping.items()
    }

    desc_lookup = {
        int(value): key
        for key, value in desc_mapping.items()
    }

    # --------------------------------------------------------
    # Get encoded values from XTraffic
    # --------------------------------------------------------

    incident_type_id = int(
        incident_features.get(
            "Type",
            -1
        )
    )

    description_id = int(
        incident_features.get(
            "Description",
            -1
        )
    )

    holiday_value = incident_features.get(
        "Holiday",
        0
    )

    # --------------------------------------------------------
    # Return human-readable information for UI
    # --------------------------------------------------------

    return {

        "incident_type":
            type_lookup.get(
                incident_type_id,
                "Unknown"
            ),

        "description":
            desc_lookup.get(
                description_id,
                "Unknown"
            ),

        "is_holiday":
            bool(
                holiday_value
            ),

    }


# ============================================================
# INVERSE TRANSFORMATION
# ============================================================

def inverse_transform_traffic(
    values,
    stats
):

    mean = np.asarray(
        stats["mean"]
    ).reshape(-1)

    std = np.asarray(
        stats["std"]
    ).reshape(-1)

    if (
        mean.size == 0
        or std.size == 0
    ):
        raise RuntimeError(
            "incident_stats.npz does not contain mean/std."
        )

    return (
        values * float(std[0])
        + float(mean[0])
    )


# ============================================================
# TIME HELPERS
# ============================================================

def build_history_times(
    start_datetime: datetime,
    points: int = 12,
    interval_minutes: int = 5
):

    return [
        start_datetime
        - timedelta(
            minutes=(points - i)
            * interval_minutes
        )
        for i in range(points)
    ]


def build_forecast_times(
    start_datetime: datetime,
    points: int = 12,
    interval_minutes: int = 5
):

    return [
        start_datetime
        + timedelta(
            minutes=(i + 1)
            * interval_minutes
        )
        for i in range(points)
    ]


# ============================================================
# BUILD SENSOR FORECAST
# ============================================================

def build_sensor_forecast(
    role: str,
    node: int,
    sensors: pd.DataFrame,
    historical_values,
    forecast_values,
    history_times,
    forecast_times
):

    row = sensors.iloc[node]

    station_id = str(
        row.get(
            "station_id",
            node
        )
    )

    sensor_name = str(
        row.get(
            "Name",
            f"Sensor {node}"
        )
    )

    street = get_sensor_street(
        sensors,
        node
    )

    historical = [
        {
            "time": timestamp.strftime("%H:%M"),
            "traffic": float(value),
        }
        for timestamp, value in zip(
            history_times,
            historical_values
        )
    ]

    forecast = [
        {
            "time": timestamp.strftime("%H:%M"),
            "traffic": float(value),
        }
        for timestamp, value in zip(
            forecast_times,
            forecast_values
        )
    ]

    return {
        "role": role,
        "sensor_node": int(node),
        "station_id": station_id,
        "sensor_name": sensor_name,
        "street": street,
        "historical": historical,
        "forecast": forecast,
    }


# ============================================================
# MAIN PREDICTION
# ============================================================

def predict_traffic(
    date: str,
    time: str,
    location: str,
    street: str,
    sensor: int,
    incident_enabled: bool = False,
    incident_time: float | None = None,
    incident_description: int | None = None,
    incident_type: int | None = None,
    holiday: int | None = None
):

    prediction_datetime = datetime.strptime(
        f"{date} {time}",
        "%Y-%m-%d %H:%M"
    )

    if location != "Alameda":
        raise ValueError(
            "Phase 1 currently supports Alameda only. "
            "The Contra Costa and Orange models have not "
            "been trained for the current UI."
        )

    dataset_path = get_dataset_path(
        location
    )

    sensors = load_sensors(
        location
    )

    # --------------------------------------------------------
    # Selected sensor
    # --------------------------------------------------------

    try:
        selected_node = int(sensor)

    except (TypeError, ValueError):
        raise ValueError(
            "Sensor must be a numeric XTraffic sensor."
        )

    if (
        selected_node < 0
        or selected_node >= len(sensors)
    ):
        raise ValueError(
            f"Sensor node must be between "
            f"0 and {len(sensors) - 1}."
        )

    # --------------------------------------------------------
    # Validate selected road/sensor relationship
    # --------------------------------------------------------

    if street:

        if not sensor_matches_street(
            sensors,
            selected_node,
            street
        ):
            raise ValueError(
                "The selected sensor does not belong "
                "to the selected street."
            )

    # --------------------------------------------------------
    # Previous / Selected / Next
    # --------------------------------------------------------

    (
        previous_node,
        selected_node,
        next_node
    ) = get_neighbor_nodes(
        sensors,
        selected_node
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model_info = load_model(
        location,
        incident_enabled
    )

    model = model_info["model"]

    stats = model_info["stats"]

    # --------------------------------------------------------
    # Load XTraffic test sample
    # --------------------------------------------------------

    sample = load_dataset_sample(
        location,
        index=0
    )

    history_tensor = prepare_history(
        sample
    )

    # --------------------------------------------------------
    # Incident data
    # --------------------------------------------------------

    incident_data = None

    incident_information = None

    if incident_enabled:

        # ----------------------------------------------------
        # Incident data used by IGSTGNN
        # ----------------------------------------------------

        incident_data = prepare_incident_data(
            sample
        )

        # ----------------------------------------------------
        # Human-readable incident information for UI
        # ----------------------------------------------------

        incident_information = (
            prepare_incident_information(
                sample,
                dataset_path
            )
        )

    # --------------------------------------------------------
    # IGSTGNN prediction
    # --------------------------------------------------------

    with torch.inference_mode():

        prediction = model(
            history_data=history_tensor,
            incident_data=incident_data,
            sensor_data=None
        )

    if prediction.ndim != 4:
        raise RuntimeError(
            "Unexpected IGSTGNN output shape: "
            f"{tuple(prediction.shape)}"
        )

    # --------------------------------------------------------
    # Historical traffic
    # --------------------------------------------------------

    x_data = np.asarray(
        sample["x_data"],
        dtype=np.float32
    )

    historical_all = x_data[:, :, 0]

    # --------------------------------------------------------
    # Forecast traffic
    # --------------------------------------------------------

    prediction_all = (
        prediction[0, :, :, 0]
        .detach()
        .cpu()
        .numpy()
    )

    prediction_all = inverse_transform_traffic(
        prediction_all,
        stats
    )

    if not np.isfinite(
        prediction_all
    ).all():
        raise RuntimeError(
            "IGSTGNN produced NaN/Inf values."
        )

    historical_all = inverse_transform_traffic(
        historical_all,
        stats
    )

    if not np.isfinite(
        historical_all
    ).all():
        raise RuntimeError(
            "Historical XTraffic data contains NaN/Inf."
        )

    # --------------------------------------------------------
    # Display timeline
    # --------------------------------------------------------

    history_times = build_history_times(
        prediction_datetime,
        points=12,
        interval_minutes=5
    )

    forecast_times = build_forecast_times(
        prediction_datetime,
        points=12,
        interval_minutes=5
    )

    # --------------------------------------------------------
    # Three sensors
    # --------------------------------------------------------

    sensor_definitions = [
        ("previous", previous_node),
        ("selected", selected_node),
        ("next", next_node),
    ]

    sensor_results = []

    for role, node in sensor_definitions:

        sensor_results.append(
            build_sensor_forecast(
                role=role,
                node=node,
                sensors=sensors,
                historical_values=historical_all[:, node],
                forecast_values=prediction_all[:, node],
                history_times=history_times,
                forecast_times=forecast_times,
            )
        )

    # --------------------------------------------------------
    # Traffic mode
    # --------------------------------------------------------

    mode = (
        "Incident-Aware Traffic"
        if incident_enabled
        else "Normal Traffic"
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {
        "location": location,

        "street": get_sensor_street(
            sensors,
            selected_node
        ),

        "selected_sensor": int(
            selected_node
        ),

        "prediction_start":
            prediction_datetime.isoformat(),

        "horizon_minutes": 60,

        "interval_minutes": 5,

        "model": "IGSTGNN",

        "mode": mode,

        "phase": "Phase 1",

        "sensors": sensor_results,

        # ----------------------------------------------------
        # Human-readable incident information for UI
        # ----------------------------------------------------

        "incident_information":
            incident_information,
    }