from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    User-facing traffic prediction request.
    """

    date: str = Field(
        ...,
        description="Prediction reference date in YYYY-MM-DD format",
    )

    time: str = Field(
        ...,
        description="Prediction reference time in HH:MM format",
    )

    location: str = Field(
        ...,
        description="XTraffic dataset/region",
    )

    street: str = Field(
        ...,
        description="Selected street/road",
    )

    sensor: int = Field(
        ...,
        description="Selected XTraffic sensor node index",
    )

    incident_enabled: bool = Field(
        False,
        description="Whether incident-aware forecasting is enabled",
    )

    incident_time: float | None = Field(
        None,
        description="Normalized incident time from XTraffic",
    )

    incident_description: int | None = Field(
        None,
        description="XTraffic incident description code",
    )

    incident_type: int | None = Field(
        None,
        description="XTraffic incident type code",
    )

    holiday: int | None = Field(
        None,
        description="XTraffic holiday indicator",
    )


class TrafficPoint(BaseModel):
    """
    One 5-minute traffic observation.
    """

    time: str
    traffic: float


class SensorForecast(BaseModel):
    """
    Historical and forecast traffic for one sensor.
    """

    role: str
    sensor_node: int
    station_id: str
    sensor_name: str
    street: str

    historical: list[TrafficPoint]
    forecast: list[TrafficPoint]


# ============================================================
# INCIDENT INFORMATION
# UI display information returned after prediction
# ============================================================

class IncidentInformation(BaseModel):
    """
    Human-readable incident information for the frontend.

    These values are derived from the XTraffic incident
    metadata and are displayed only after prediction.
    """

    incident_type: str
    description: str
    is_holiday: bool


class PredictionResponse(BaseModel):
    """
    Response returned to the PulseTraffic AI frontend.
    """

    location: str
    street: str
    selected_sensor: int

    prediction_start: str

    horizon_minutes: int = 60
    interval_minutes: int = 5

    model: str = "IGSTGNN"
    mode: str

    phase: str = "Phase 1"

    sensors: list[SensorForecast]

    # ========================================================
    # INCIDENT INFORMATION
    # Returned only when incident-aware prediction is enabled.
    # Normal prediction returns null.
    # ========================================================

    incident_information: IncidentInformation | None = None