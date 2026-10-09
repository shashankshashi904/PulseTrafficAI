from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.predict import router as prediction_router


# ============================================================
# PulseTraffic AI - FastAPI Application
# ============================================================

app = FastAPI(
    title="PulseTraffic AI",
    description=(
        "AI-Based Road Traffic Prediction for "
        "Emergency Service Facilitation"
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# Allows the frontend running from the local browser
# to communicate with the FastAPI backend.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Routes
# ============================================================

app.include_router(prediction_router)


# ============================================================
# Health Check
# ============================================================

@app.get("/")
def root():
    return {
        "application": "PulseTraffic AI",
        "status": "running",
        "model": "IGSTGNN",
        "phase": "Phase 1",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }