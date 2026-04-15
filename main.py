import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AEGIS Triple-Modal Fusion API",
    description="Advanced asteroid impact prediction using NASA NEO, USGS terrain, and spacecraft telemetry data",
    version="2.0.0"
)

# Enable CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- MODEL LOADING ---
MODEL_PATH = "models/aegis_impact_voter_v01.joblib"
PREPROCESSOR_PATH = "models/scalers/feature_preprocessor_v01.joblib"
COMPAT_MODEL_PATH = "models/aegis_ensemble.joblib"
COMPAT_SCALER_PATH = "models/scaler.joblib"

print(f"📡 Loading Model: {MODEL_PATH}")
print(f"📏 Loading Preprocessor: {PREPROCESSOR_PATH}")

model = None
preprocessor = None
feature_names = None

# Try to load the triple-modal fusion model first
if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)

        # Extract feature names from preprocessor
        if hasattr(preprocessor, 'transformers_'):
            numeric_features = ['diameter', 'velocity', 'eccentricity', 'inclination',
                                'soil_density', 'porosity', 'elevation', 'water_depth',
                                'impactor_mass', 'intercept_velocity', 'momentum_factor']

            # Get one-hot encoded feature names
            cat_encoder = preprocessor.named_transformers_['cat']
            if hasattr(cat_encoder, 'get_feature_names_out'):
                cat_features = cat_encoder.get_feature_names_out(
                    ['composition']).tolist()
            else:
                cat_features = ['composition_C-type',
                                'composition_S-type', 'composition_M-type']

            feature_names = numeric_features + cat_features

        logger.info("✅ System Ready: Triple-Modal Fusion Model is Online.")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(
            f"Input features: {len(feature_names) if feature_names else 'Unknown'}")
        logger.info(f"   Features: {feature_names}")

    except Exception as e:
        logger.error(f"❌ Error loading triple-modal model: {e}")
        model = None
        preprocessor = None

# Fallback to compatible model if triple-modal not available
if model is None and os.path.exists(COMPAT_MODEL_PATH) and os.path.exists(COMPAT_SCALER_PATH):
    try:
        model = joblib.load(COMPAT_MODEL_PATH)
        preprocessor = joblib.load(COMPAT_SCALER_PATH)
        logger.info(
            "✅ System Ready: Compatible Model is Online (3-feature version).")
        feature_names = ['diameter', 'velocity', 'composition']
    except Exception as e:
        logger.error(f"❌ Error loading compatible model: {e}")
        model = None
        preprocessor = None

if model is None:
    logger.warning("❌ No model files found. Run train_and_save.py first.")

# --- API MODELS ---


class NASAData(BaseModel):
    """NASA NEO astronomical data."""
    diameter: float = Field(..., gt=0, le=10.0,
                            description="Asteroid diameter (km)")
    velocity: float = Field(..., ge=5.0, le=70.0,
                            description="Relative velocity (km/s)")
    eccentricity: float = Field(
        0.5, ge=0, le=1.0, description="Orbital eccentricity")
    inclination: float = Field(
        10.0, ge=0, le=180, description="Orbital inclination (degrees)")
    composition: str = Field(..., pattern='^(C-type|S-type|M-type)$',
                             description="Spectral class")

    model_config = {
        "json_schema_extra": {
            "example": {
                "diameter": 1.5,
                "velocity": 25.0,
                "eccentricity": 0.5,
                "inclination": 10,
                "composition": "S-type"
            }
        }
    }


class USGSData(BaseModel):
    """USGS terrestrial data."""
    soil_density: float = Field(
        2.0, ge=1.0, le=5.0, description="Soil density (g/cm³)")
    porosity: float = Field(0.3, ge=0, le=1.0, description="Soil porosity")
    elevation: float = Field(1000.0, ge=-500, le=9000,
                             description="Impact site elevation (m)")
    water_depth: float = Field(
        0.0, ge=0, le=11000, description="Water depth (m) for maritime impacts")

    model_config = {
        "json_schema_extra": {
            "example": {
                "soil_density": 2.0,
                "porosity": 0.3,
                "elevation": 1000,
                "water_depth": 0
            }
        }
    }


class TelemetryData(BaseModel):
    """Spacecraft kinetic impactor telemetry."""
    impactor_mass: float = Field(
        750.0, ge=100, le=5000, description="Impactor mass (kg)")
    intercept_velocity: float = Field(
        7.0, ge=1, le=30, description="Intercept velocity (km/s)")
    momentum_factor: float = Field(
        3.0, ge=0.1, le=10.0, description="Momentum enhancement factor (β)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "impactor_mass": 750,
                "intercept_velocity": 7.0,
                "momentum_factor": 3.0
            }
        }
    }


class AsteroidInput(BaseModel):
    """Input data for asteroid impact prediction (simplified version)."""
    diameter: float = Field(..., gt=0, le=10.0,
                            description="Asteroid diameter in kilometers")
    velocity: float = Field(..., ge=5.0, le=70.0,
                            description="Impact velocity in km/s")
    composition: int = Field(..., ge=0, le=2,
                             description="Composition type: 0=C-type, 1=S-type, 2=M-type")

    @field_validator('diameter')
    @classmethod
    def validate_diameter(cls, v):
        if v <= 0:
            raise ValueError('Diameter must be positive')
        if v > 10.0:
            raise ValueError(
                'Diameter exceeds maximum realistic value (10 km)')
        return v

    @field_validator('velocity')
    @classmethod
    def validate_velocity(cls, v):
        if v < 5.0:
            raise ValueError('Velocity below minimum realistic value (5 km/s)')
        if v > 70.0:
            raise ValueError(
                'Velocity exceeds maximum realistic value (70 km/s)')
        return v

    @field_validator('composition')
    @classmethod
    def validate_composition(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError(
                'Composition must be 0 (C-type), 1 (S-type), or 2 (M-type)')
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "diameter": 1.5,
                "velocity": 25.0,
                "composition": 1
            }
        }
    }


class FullImpactInput(BaseModel):
    """Complete input for triple-modal fusion model."""
    nasa: NASAData
    usgs: USGSData = USGSData()
    telemetry: TelemetryData = TelemetryData()

    model_config = {
        "json_schema_extra": {
            "example": {
                "nasa": {
                    "diameter": 1.5,
                    "velocity": 25.0,
                    "eccentricity": 0.5,
                    "inclination": 10,
                    "composition": "S-type"
                },
                "usgs": {
                    "soil_density": 2.0,
                    "porosity": 0.3,
                    "elevation": 1000,
                    "water_depth": 0
                },
                "telemetry": {
                    "impactor_mass": 750,
                    "intercept_velocity": 7.0,
                    "momentum_factor": 3.0
                }
            }
        }
    }


class PredictionResponse(BaseModel):
    """Prediction results with spacecraft recommendations."""
    crater_km: float = Field(..., description="Predicted crater diameter (km)")
    seismic_mag: float = Field(..., description="Predicted seismic magnitude")
    model_version: str = Field(..., description="Model version")
    composition_type: str = Field(...,
                                  description="Human-readable composition type")
    confidence_interval: Dict[str, List[float]] = Field(
        None, description="Confidence intervals")
    model_type: str = Field("triple-modal", description="Type of model used")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    recommended_spacecraft: Optional[List[Dict[str, Any]]] = Field(
        None, description="Top spacecraft recommendations for NEO distraction/deflection")

    model_config = {
        "json_schema_extra": {
            "example": {
                "crater_km": 3.2,
                "seismic_mag": 5.7,
                "model_version": "2.0.0",
                "composition_type": "S-type (Silicaceous)",
                "confidence_interval": {
                    "crater_km": [2.8, 3.6],
                    "seismic_mag": [5.4, 6.0]
                },
                "model_type": "triple-modal",
                "timestamp": "2024-01-01T12:00:00"
            }
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_type: str
    features_count: Optional[int]
    model_info: Dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "online",
                "model_loaded": True,
                "preprocessor_loaded": True,
                "model_type": "triple-modal",
                "features_count": 14,
                "model_info": {
                    "type": "MultiOutputRegressor",
                    "features": ["diameter", "velocity", "composition"],
                    "outputs": ["crater_km", "seismic_mag"]
                }
            }
        }
    }

# --- HELPER FUNCTIONS ---


def get_composition_name(comp_code: int) -> str:
    """Convert composition code to human-readable name."""
    composition_map = {
        0: "C-type (Carbonaceous)",
        1: "S-type (Silicaceous)",
        2: "M-type (Metallic)"
    }
    return composition_map.get(comp_code, "Unknown")


def get_composition_from_string(comp_str: str) -> int:
    """Convert composition string to integer code."""
    comp_map = {
        "C-type": 0,
        "S-type": 1,
        "M-type": 2
    }
    return comp_map.get(comp_str, 0)


def prepare_full_input(input_data: FullImpactInput) -> pd.DataFrame:
    """
    Convert API input to DataFrame format expected by preprocessor.
    """
    data_dict = {
        # NASA data
        'diameter': [input_data.nasa.diameter],
        'velocity': [input_data.nasa.velocity],
        'eccentricity': [input_data.nasa.eccentricity],
        'inclination': [input_data.nasa.inclination],
        'composition': [input_data.nasa.composition],

        # USGS data
        'soil_density': [input_data.usgs.soil_density],
        'porosity': [input_data.usgs.porosity],
        'elevation': [input_data.usgs.elevation],
        'water_depth': [input_data.usgs.water_depth],

        # Telemetry data
        'impactor_mass': [input_data.telemetry.impactor_mass],
        'intercept_velocity': [input_data.telemetry.intercept_velocity],
        'momentum_factor': [input_data.telemetry.momentum_factor]
    }

    return pd.DataFrame(data_dict)


def prepare_simple_input(diameter: float, velocity: float, composition: int) -> pd.DataFrame:
    """Prepare input for the 3-feature model."""
    comp_str = get_composition_name(composition).split()[
        0]  # Extract base type

    # Create full DataFrame with defaults for other features
    data_dict = {
        'diameter': [diameter],
        'velocity': [velocity],
        'eccentricity': [0.5],
        'inclination': [10.0],
        'composition': [comp_str],
        'soil_density': [2.0],
        'porosity': [0.3],
        'elevation': [1000.0],
        'water_depth': [0.0],
        'impactor_mass': [750.0],
        'intercept_velocity': [7.0],
        'momentum_factor': [3.0]
    }

    return pd.DataFrame(data_dict)


def estimate_confidence(prediction: np.ndarray, model) -> Dict[str, List[float]]:
    """Estimate confidence intervals for predictions."""
    try:
        # Simple confidence estimation based on prediction magnitude
        std_factor = 0.1  # 10% standard deviation
        return {
            "crater_km": [
                float(prediction[0][0] * (1 - 2*std_factor)),
                float(prediction[0][0] * (1 + 2*std_factor))
            ],
            "seismic_mag": [
                float(prediction[0][1] - 0.5),
                float(prediction[0][1] + 0.5)
            ]
        }
    except:
        # Default confidence intervals (±20%)
        return {
            "crater_km": [
                float(prediction[0][0] * 0.8),
                float(prediction[0][0] * 1.2)
            ],
            "seismic_mag": [
                float(prediction[0][1] - 0.5),
                float(prediction[0][1] + 0.5)
            ]
        }


def recommend_spacecraft(nasa_data: NASAData) -> List[Dict[str, Any]]:
    """
    Rule-based spacecraft recommendation for NEO distraction/deflection.
    Returns top 3 spacecrafts ranked by match_score (0-1).
    """
    spacecrafts = [
        {
            "name": "RK-01 DART",
            "status": "🟢 ACTIVE",
            "best_for": "S-type asteroids ≤150m, low-medium velocity",
            "success_probability": 0.85
        },
        {
            "name": "RK-02 ORION",
            "status": "🟡 STANDBY",
            "best_for": "M-type metallic asteroids, high-speed intercept",
            "success_probability": 0.78
        },
        {
            "name": "HAMMER",
            "status": "🟡 STANDBY",
            "best_for": "Large asteroids (>1km), multiple penetrators",
            "success_probability": 0.72
        },
        {
            "name": "AEGIS-X",
            "status": "🔵 DEVELOPMENT",
            "best_for": "Next-gen high-risk intercepts, AI trajectory",
            "success_probability": 0.90
        }
    ]

    diameter_km = nasa_data.diameter
    velocity_kms = nasa_data.velocity
    comp = nasa_data.composition

    # Calculate match scores (0-1) based on mission specs
    for sc in spacecrafts:
        match_score = 0.0

        # Diameter matching
        if sc["name"] == "RK-01 DART":
            if diameter_km <= 0.15:
                match_score += 1.0
            elif diameter_km <= 0.5:
                match_score += 0.8
            else:
                match_score += 0.3
        elif sc["name"] in ["RK-02 ORION", "HAMMER"]:
            if diameter_km <= 2.0:
                match_score += 0.9
            else:
                match_score += 0.6
        else:  # AEGIS-X versatile
            match_score += 0.7

        # Composition matching
        if comp == "S-type" and "DART" in sc["name"]:
            match_score += 0.25
        elif comp == "M-type" and "ORION" in sc["name"]:
            match_score += 0.25
        elif comp == "C-type":
            match_score += 0.15  # Less specialized

        # Velocity matching (kinetic intercept capability)
        if velocity_kms <= 30:
            match_score += 0.2
        elif velocity_kms <= 50:
            match_score += 0.1

        # Clamp to 0-1
        sc["match_score"] = min(1.0, match_score)
        # Adjust base prob by match
        sc["success_probability"] *= sc["match_score"]

    # Sort by match_score descending, return top 3
    spacecrafts.sort(key=lambda x: x["match_score"], reverse=True)
    return spacecrafts[:3]

# --- API ENDPOINTS ---


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with health check."""
    model_type = "unknown"
    if model is not None:
        if hasattr(model, 'estimators_') and feature_names and len(feature_names) > 5:
            model_type = "triple-modal"
        else:
            model_type = "simple"

    return {
        "status": "online",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_type": model_type,
        "features_count": len(feature_names) if feature_names else 0,
        "model_info": {
            "type": str(type(model).__name__) if model else "Not loaded",
            "features": feature_names[:5] if feature_names else [],
            "outputs": ["crater_km", "seismic_mag"]
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return await root()


@app.post("/predict", response_model=PredictionResponse)
async def predict_simple(data: AsteroidInput):
    """
    Predict asteroid impact consequences using simplified input.

    Takes asteroid parameters and returns predicted crater size and seismic magnitude.
    This endpoint maintains compatibility with the existing frontend.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist and run training first."
        )

    try:
        # Prepare input data with defaults for other features
        input_df = prepare_simple_input(
            data.diameter, data.velocity, data.composition)

        # Transform using preprocessor
        processed_features = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_features)

        # Get composition name
        comp_name = get_composition_name(data.composition)

        # Estimate confidence intervals
        confidence = estimate_confidence(prediction, model)

        # Determine model type
        model_type = "triple-modal" if feature_names and len(
            feature_names) > 5 else "simple"

        logger.info(f"Prediction made - Diameter: {data.diameter}km, "
                    f"Velocity: {data.velocity}km/s, "
                    f"Composition: {comp_name}")
        logger.info(f"Results - Crater: {prediction[0][0]:.2f}km, "
                    f"Seismic: {prediction[0][1]:.2f}")

        return {
            "crater_km": float(prediction[0][0]),
            "seismic_mag": float(prediction[0][1]),
            "model_version": "2.0.0",
            "composition_type": comp_name,
            "confidence_interval": confidence,
            "model_type": model_type,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/full", response_model=PredictionResponse)
async def predict_full(data: FullImpactInput):
    """
    Full prediction endpoint using all three data modalities.

    Combines NASA NEO astronomical data, USGS terrain data,
    and spacecraft telemetry for most accurate predictions.
    """
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist and run training first."
        )

    try:
        # Prepare full input data
        input_df = prepare_full_input(data)

        # Transform using preprocessor
        processed_features = preprocessor.transform(input_df)

        # Make prediction
        prediction = model.predict(processed_features)

        # Get composition name
        comp_name = data.nasa.composition

        # Estimate confidence intervals (more confident with full data)
        confidence = estimate_confidence(prediction, model)

        logger.info(f"Full prediction made - Diameter: {data.nasa.diameter}km, "
                    f"Velocity: {data.nasa.velocity}km/s, "
                    f"Composition: {data.nasa.composition}")
        logger.info(f"Results - Crater: {prediction[0][0]:.2f}km, "
                    f"Seismic: {prediction[0][1]:.2f}")

        # Generate spacecraft recommendations for distraction/deflection
        spacecraft_recs = recommend_spacecraft(data.nasa)

        return {
            "crater_km": float(prediction[0][0]),
            "seismic_mag": float(prediction[0][1]),
            "model_version": "2.0.1",  # Updated version with spacecraft recs
            "composition_type": comp_name,
            "confidence_interval": confidence,
            "model_type": "triple-modal",
            "timestamp": datetime.now().isoformat(),
            "recommended_spacecraft": spacecraft_recs
        }

    except Exception as e:
        logger.error(f"Full prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Full prediction failed: {str(e)}"
        )


@app.post("/predict/batch")
async def predict_batch(data: List[AsteroidInput]):
    """
    Batch prediction endpoint for multiple asteroids.
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        results = []
        for input_data in data:
            input_df = prepare_simple_input(
                input_data.diameter,
                input_data.velocity,
                input_data.composition
            )
            processed = preprocessor.transform(input_df)
            prediction = model.predict(processed)

            comp_name = get_composition_name(input_data.composition)

            results.append({
                "input": input_data.model_dump(),
                "crater_km": float(prediction[0][0]),
                "seismic_mag": float(prediction[0][1]),
                "composition_type": comp_name
            })

        logger.info(f"Batch prediction completed for {len(data)} samples")

        return {
            "predictions": results,
            "count": len(results),
            "model_version": "2.0.0"
        }

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model-info")
async def model_info():
    """Get detailed information about the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model_type = "triple-modal" if feature_names and len(
        feature_names) > 5 else "simple"

    info = {
        "model_type": str(type(model).__name__),
        "model_variant": model_type,
        "model_loaded": True,
        "preprocessor_loaded": preprocessor is not None,
        "features": feature_names if feature_names else [],
        "outputs": ["crater_size (km)", "seismic_magnitude"],
        "composition_mapping": {
            "0": "C-type (Carbonaceous)",
            "1": "S-type (Silicaceous)",
            "2": "M-type (Metallic)"
        }
    }

    # Add ensemble-specific info if available
    if hasattr(model, 'estimators_'):
        info["num_outputs"] = len(model.estimators_)
        if hasattr(model.estimators_[0], 'named_estimators_'):
            info["ensemble_models"] = list(
                model.estimators_[0].named_estimators_.keys())

    return info

# Run with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
