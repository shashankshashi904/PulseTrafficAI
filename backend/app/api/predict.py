from fastapi import APIRouter, HTTPException
from ..schemas.prediction import PredictionRequest, PredictionResponse
from ..services.prediction_service import (
    predict_traffic,
    load_sensors,
    get_sensor_street,
)

router = APIRouter(prefix="/api/v1", tags=["Traffic Prediction"])


@router.get("/metadata")
def get_metadata(location: str = "Alameda"):
    try:
        sensors = load_sensors(location)

        sensor_list = []

        for index, row in sensors.iterrows():

            node = int(index)

            sensor_list.append(
                {
                    "node": node,
                    "station_id": str(
                        row.get("station_id", node)
                    ),
                    "name": str(
                        row.get("Name", f"Sensor {node}")
                    ),
                    "street": get_sensor_street(
                        sensors,
                        node,
                    ),
                    "fwy": str(
                        row.get("Fwy", "")
                    ),
                }
            )

        streets = sorted(
            {
                sensor["street"]
                for sensor in sensor_list
                if sensor["street"]
            }
        )

        return {
            "location": location,
            "streets": streets,
            "sensors": sensor_list,
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Metadata loading failed: {exc}",
        )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:

        result = predict_traffic(
            date=request.date,
            time=request.time,
            location=request.location,
            street=request.street,
            sensor=request.sensor,
            incident_enabled=request.incident_enabled,
            incident_time=request.incident_time,
            incident_description=request.incident_description,
            incident_type=request.incident_type,
            holiday=request.holiday,
        )

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except NotImplementedError as exc:
        raise HTTPException(
            status_code=501,
            detail=str(exc),
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        )