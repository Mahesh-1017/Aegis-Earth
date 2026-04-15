"""
Configuration file for AEGIS Earth system.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
SCALERS_DIR = MODELS_DIR / "scalers"
ENSEMBLE_DIR = MODELS_DIR / "ensemble"
RESULTS_DIR = BASE_DIR / "results"

# Create directories if they don't exist
for dir_path in [MODELS_DIR, SCALERS_DIR, ENSEMBLE_DIR, RESULTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Model paths
MODEL_PATHS = {
    "triple_modal": MODELS_DIR / "aegis_impact_voter_v01.joblib",
    "preprocessor": SCALERS_DIR / "feature_preprocessor_v01.joblib",
    "compatible_model": MODELS_DIR / "aegis_ensemble.joblib",
    "compatible_scaler": MODELS_DIR / "scaler.joblib",
}

# API settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "AEGIS Triple-Modal Fusion API"
API_VERSION = "2.0.0"

# NASA API
NASA_API_KEY = "QpCinFsJT2fYXQLkkrTNwCCsOlBgW1v66T1OqFqt"
NASA_API_URL = "https://api.nasa.gov/neo/rest/v1/neo/browse"

# Model parameters
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 150,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    },
    "xgboost": {
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42
    }
}

# Feature configuration
FEATURES = {
    "numeric": [
        'diameter', 'velocity', 'eccentricity', 'inclination',
        'soil_density', 'porosity', 'elevation', 'water_depth',
        'impactor_mass', 'intercept_velocity', 'momentum_factor'
    ],
    "categorical": ['composition']
}

# Input validation ranges
VALIDATION_RANGES = {
    "diameter": {"min": 0.1, "max": 10.0},
    "velocity": {"min": 5.0, "max": 70.0},
    "eccentricity": {"min": 0.0, "max": 1.0},
    "inclination": {"min": 0.0, "max": 180.0},
    "soil_density": {"min": 1.0, "max": 5.0},
    "porosity": {"min": 0.0, "max": 1.0},
    "elevation": {"min": -500, "max": 9000},
    "water_depth": {"min": 0, "max": 11000},
    "impactor_mass": {"min": 100, "max": 5000},
    "intercept_velocity": {"min": 1.0, "max": 30.0},
    "momentum_factor": {"min": 0.1, "max": 10.0}
}

# Composition mapping
COMPOSITION_MAP = {
    "C-type": 0,
    "S-type": 1,
    "M-type": 2
}

COMPOSITION_NAMES = {
    0: "C-type (Carbonaceous)",
    1: "S-type (Silicaceous)",
    2: "M-type (Metallic)"
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["file", "stream"]
}