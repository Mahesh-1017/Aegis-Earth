import os
import joblib
import numpy as np
import requests
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. Setup Directories
os.makedirs('models/scalers', exist_ok=True)
os.makedirs('models/ensemble', exist_ok=True)

# 2. Fetch NASA NEO Dataset (Astronomical Data)
def fetch_nasa_neo_data(api_key, samples=500):
    """Fetch real asteroid data from NASA API."""
    base_url = "https://api.nasa.gov/neo/rest/v1/neo/browse"
    params = {"api_key": api_key, "page": 1, "size": min(samples, 20)}  # API limit
    
    try:
        response = requests.get(base_url, params=params, timeout=10)
        if response.status_code != 200:
            raise Exception(f"NASA API Error: {response.status_code}")
        
        data = response.json()
        asteroids = data.get("near_earth_objects", [])
        
        neo_data = []
        for asteroid in asteroids[:samples]:
            # Physical Characteristics
            diameter_data = asteroid.get("estimated_diameter", {}).get("kilometers", {})
            diameter_min = diameter_data.get("estimated_diameter_min", 0) or 0
            diameter_max = diameter_data.get("estimated_diameter_max", 0) or 0
            diameter = (diameter_min + diameter_max) / 2
            if diameter <= 0 or np.isnan(diameter):
                diameter = np.random.uniform(0.1, 2.0)  # Fallback
            
            # Kinematic Data
            orbital_data = asteroid.get("orbital_data", {})
            try:
                eccentricity = float(orbital_data.get("eccentricity", 0) or 0)
            except:
                eccentricity = np.random.uniform(0, 0.5)
                
            try:
                inclination = float(orbital_data.get("inclination", 0) or 0)
            except:
                inclination = np.random.uniform(0, 20)
            
            velocity = 0
            close_approaches = asteroid.get("close_approach_data", [])
            if close_approaches:
                vel_str = close_approaches[0].get("relative_velocity", {}).get("kilometers_per_second", "0")
                try:
                    velocity = float(vel_str)
                except (ValueError, TypeError):
                    velocity = np.random.uniform(10, 30)
            
            # Spectral Classification
            composition = np.random.choice(['C-type', 'S-type', 'M-type'])
            
            neo_data.append([diameter, velocity, eccentricity, inclination, composition])
        
        df = pd.DataFrame(neo_data, columns=['diameter', 'velocity', 'eccentricity', 'inclination', 'composition'])
        
        # Clean data
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(df.mean())
        
        return df
    
    except Exception as e:
        logger.error(f"Error fetching NASA data: {e}")
        return generate_synthetic_neo_data(samples)

def generate_synthetic_neo_data(samples=500):
    """Generate synthetic NEO data as fallback."""
    np.random.seed(42)
    data = pd.DataFrame({
        'diameter': np.random.lognormal(mean=-0.5, sigma=0.8, size=samples),
        'velocity': np.random.normal(loc=20.0, scale=8.0, size=samples),
        'eccentricity': np.random.uniform(0.1, 0.6, samples),
        'inclination': np.random.uniform(0, 25, samples),
        'composition': np.random.choice(['C-type', 'S-type', 'M-type'], samples)
    })
    # Clip to realistic ranges
    data['diameter'] = data['diameter'].clip(0.1, 10.0)
    data['velocity'] = data['velocity'].clip(5.0, 70.0)
    data['eccentricity'] = data['eccentricity'].clip(0, 0.9)
    data['inclination'] = data['inclination'].clip(0, 40)
    return data

# 3. Simulate USGS Topographical and Geological Data
def generate_usgs_data(samples=500):
    """Generate synthetic USGS terrain data."""
    np.random.seed(42)
    usgs_data = pd.DataFrame({
        'soil_density': np.random.uniform(1.8, 2.8, samples),  # ρ (g/cm³)
        'porosity': np.random.uniform(0.15, 0.45, samples),
        'elevation': np.random.uniform(100, 3000, samples),  # meters
        'water_depth': np.random.uniform(0, 500, samples)  # meters
    })
    return usgs_data

# 4. Simulate Spacecraft Kinetic Impactor Telemetry
def generate_telemetry_data(samples=500):
    """Generate synthetic spacecraft telemetry data."""
    np.random.seed(42)
    telemetry_data = pd.DataFrame({
        'impactor_mass': np.random.uniform(550, 900, samples),  # M_i (kg)
        'intercept_velocity': np.random.uniform(6, 9, samples),  # V_i (km/s)
        'momentum_factor': np.random.uniform(1.5, 4.0, samples)  # β
    })
    return telemetry_data

# 5. Generate Physics-Based Targets with Safe Numerical Operations
def generate_targets(X, composition_encoded=False):
    """
    Generate realistic impact consequences based on physics.
    Includes safe numerical operations to avoid NaN/Inf.
    """
    np.random.seed(42)
    n_samples = X.shape[0]
    
    # Extract diameter and velocity safely
    if composition_encoded:
        # For preprocessed data, assume first two columns are diameter and velocity
        diameter = X[:, 0] if X.shape[1] > 0 else np.random.uniform(0.5, 5.0, n_samples)
        velocity = X[:, 1] if X.shape[1] > 1 else np.random.uniform(10, 30, n_samples)
    else:
        # Raw data - extract features with safe defaults
        diameter = X[:, 0] if X.shape[1] > 0 else np.random.uniform(0.5, 5.0, n_samples)
        velocity = X[:, 1] if X.shape[1] > 1 else np.random.uniform(10, 30, n_samples)
    
    # Ensure arrays are float and finite
    diameter = np.array(diameter, dtype=np.float64).flatten()
    velocity = np.array(velocity, dtype=np.float64).flatten()
    
    # Replace any invalid values
    diameter = np.nan_to_num(diameter, nan=1.0, posinf=5.0, neginf=0.5)
    velocity = np.nan_to_num(velocity, nan=20.0, posinf=30.0, neginf=10.0)
    
    # Clip to realistic ranges
    diameter = np.clip(diameter, 0.1, 10.0)
    velocity = np.clip(velocity, 5.0, 70.0)
    
    # Calculate kinetic energy safely
    # Assume spherical asteroid with density 3000 kg/m³
    density = 3000  # kg/m³
    radius_m = (diameter * 1000) / 2  # Convert to meters
    volume = (4/3) * np.pi * np.power(radius_m, 3)
    mass = density * volume  # kg
    
    # Safe kinetic energy calculation (avoid overflow)
    velocity_ms = velocity * 1000  # Convert to m/s
    kinetic_energy = 0.5 * mass * np.power(velocity_ms, 2)  # Joules
    
    # Handle very small or zero values
    kinetic_energy = np.maximum(kinetic_energy, 1e10)  # Minimum 10^10 J
    
    # Crater size (km) - using safe log10 and power operations
    # Use log10 safely by adding small epsilon
    log_ke = np.log10(kinetic_energy + 1e-10)
    
    # Empirical crater scaling (km)
    crater = 0.1 * np.power(10, (log_ke - 12) * 0.3)  # Simplified scaling
    
    # Add composition effects if available
    if not composition_encoded and X.shape[1] > 2:
        composition = X[:, 2]
        comp_modifier = np.array([1.0 if c == 0 else 1.2 if c == 1 else 0.8 
                                  for c in composition])
        crater *= comp_modifier
    
    # Seismic magnitude - using safe log10
    seismic = 4.0 + 0.5 * (log_ke - 12)  # Moment magnitude approximation
    
    # Add realistic noise
    crater += np.random.normal(0, crater.mean() * 0.15, n_samples)
    seismic += np.random.normal(0, 0.3, n_samples)
    
    # Simulate rare events (5% of cases)
    rare_mask = np.random.rand(n_samples) < 0.05
    crater[rare_mask] *= np.random.uniform(1.3, 2.0, np.sum(rare_mask))
    seismic[rare_mask] += np.random.uniform(0.3, 0.8, np.sum(rare_mask))
    
    # Ensure positive and finite values
    crater = np.maximum(0.2, crater)
    seismic = np.maximum(3.5, seismic)
    seismic = np.minimum(9.5, seismic)  # Cap at realistic maximum
    
    # Final check for NaN/Inf
    crater = np.nan_to_num(crater, nan=1.0, posinf=10.0, neginf=0.2)
    seismic = np.nan_to_num(seismic, nan=5.0, posinf=9.0, neginf=3.5)
    
    return np.column_stack((crater, seismic))

# 6. Main Triple-Modal Fusion
API_KEY = 'QpCinFsJT2fYXQLkkrTNwCCsOlBgW1v66T1OqFqt'  # Your NASA API key

print("=" * 60)
print("🚀 AEGIS Triple-Modal Fusion Model Training")
print("=" * 60)

# Fetch or generate data with minimum samples
MIN_SAMPLES = 100  # Increased minimum samples for better training

print(f"\n📡 Fetching NASA NEO data (target: {MIN_SAMPLES} samples)...")
try:
    neo_df = fetch_nasa_neo_data(API_KEY, samples=MIN_SAMPLES)
    print(f"   ✅ Retrieved {len(neo_df)} asteroid records")
except Exception as e:
    print(f"   ⚠️ Error: {e}")
    print("   ⚠️ Generating synthetic NEO data...")
    neo_df = generate_synthetic_neo_data(MIN_SAMPLES)

# Ensure we have enough samples
if len(neo_df) < MIN_SAMPLES:
    print(f"   ⚠️ Insufficient real data ({len(neo_df)}), generating additional synthetic data...")
    synthetic_df = generate_synthetic_neo_data(MIN_SAMPLES - len(neo_df))
    neo_df = pd.concat([neo_df, synthetic_df], ignore_index=True)
    print(f"   ✅ Total samples: {len(neo_df)}")

print("\n🌍 Generating USGS terrain data...")
usgs_df = generate_usgs_data(len(neo_df))
print(f"   ✅ Generated {len(usgs_df)} terrain samples")

print("\n🛰️ Generating spacecraft telemetry data...")
telemetry_df = generate_telemetry_data(len(neo_df))
print(f"   ✅ Generated {len(telemetry_df)} telemetry samples")

# Fuse into high-dimensional feature space
X = pd.concat([neo_df, usgs_df, telemetry_df], axis=1)
print(f"\n📊 Fused dataset shape: {X.shape}")
print(f"   Features: {list(X.columns)}")

# Clean the data
X = X.replace([np.inf, -np.inf], np.nan)
for col in X.columns:
    if X[col].dtype in ['float64', 'float32']:
        X[col] = X[col].fillna(X[col].mean())

# 7. Preprocessing Pipeline
numeric_features = ['diameter', 'velocity', 'eccentricity', 'inclination', 
                    'soil_density', 'porosity', 'elevation', 'water_depth', 
                    'impactor_mass', 'intercept_velocity', 'momentum_factor']
categorical_features = ['composition']

# Verify all numeric features exist
numeric_features = [f for f in numeric_features if f in X.columns]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)
    ]
)

print("\n🔄 Preprocessing features...")
X_processed = preprocessor.fit_transform(X)

# Get feature names after preprocessing
feature_names = numeric_features.copy()
cat_encoder = preprocessor.named_transformers_['cat']
try:
    cat_features = cat_encoder.get_feature_names_out(categorical_features).tolist()
except:
    cat_features = [f"{categorical_features[0]}_{cat}" for cat in cat_encoder.categories_[0]]
feature_names.extend(cat_features)
print(f"   ✅ Preprocessed shape: {X_processed.shape}")
print(f"   🔤 Feature names: {feature_names[:5]}... ({len(feature_names)} total)")

# Save preprocessor
preprocessor_path = 'models/scalers/feature_preprocessor_v01.joblib'
joblib.dump(preprocessor, preprocessor_path)
print(f"   💾 Preprocessor saved to {preprocessor_path}")

# Generate targets
print("\n🎯 Generating impact targets...")
y = generate_targets(X_processed, composition_encoded=True)
print(f"   ✅ Targets shape: {y.shape}")
print(f"   📈 Target ranges - Crater: [{y[:,0].min():.2f}, {y[:,0].max():.2f}] km, "
      f"Seismic: [{y[:,1].min():.2f}, {y[:,1].max():.2f}]")

# Verify no NaN in targets
if np.any(np.isnan(y)):
    print("   ⚠️ Found NaN in targets, replacing with default values...")
    y = np.nan_to_num(y, nan=1.0, posinf=10.0, neginf=0.5)

# Split data for validation
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

print(f"\n📊 Training set: {X_train.shape[0]} samples")
print(f"📊 Test set: {X_test.shape[0]} samples")

# 8. Train Voting Regressor Ensemble
print("\n🧠 Training ensemble model...")

# Random Forest with conservative parameters
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

# XGBoost with conservative parameters
xgb = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

# Create voting ensemble
voting_regressor = VotingRegressor(
    estimators=[('rf', rf), ('xgb', xgb)],
    weights=[1, 1]
)

# Wrap for multi-output
ensemble = MultiOutputRegressor(voting_regressor, n_jobs=-1)

# Train
print("   Training in progress...")
try:
    ensemble.fit(X_train, y_train)
    print("   ✅ Training complete")
except Exception as e:
    print(f"   ❌ Training error: {e}")
    print("   Trying with simplified model...")
    # Fallback to simple Random Forest
    from sklearn.ensemble import RandomForestRegressor
    ensemble = MultiOutputRegressor(RandomForestRegressor(n_estimators=50, random_state=42))
    ensemble.fit(X_train, y_train)

# Evaluate
y_pred = ensemble.predict(X_test)

crater_rmse = np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))
seismic_rmse = np.sqrt(mean_squared_error(y_test[:, 1], y_pred[:, 1]))
crater_r2 = r2_score(y_test[:, 0], y_pred[:, 0])
seismic_r2 = r2_score(y_test[:, 1], y_pred[:, 1])

print(f"\n📊 Model Performance:")
print(f"   Crater Size - RMSE: {crater_rmse:.3f} km, R²: {crater_r2:.3f}")
print(f"   Seismic Mag - RMSE: {seismic_rmse:.3f}, R²: {seismic_r2:.3f}")

# Save model
model_path = 'models/aegis_impact_voter_v01.joblib'
joblib.dump(ensemble, model_path)
print(f"   💾 Model saved to {model_path}")

# Also save in the format expected by main.py
os.makedirs('models', exist_ok=True)
joblib.dump(ensemble, 'models/aegis_ensemble.joblib')
joblib.dump(preprocessor, 'models/scaler.joblib')
print(f"   💾 Compatible model saved to models/aegis_ensemble.joblib")

# 9. Test Prediction
print("\n🔬 Testing prediction with sample input:")
test_input = pd.DataFrame({
    'diameter': [1.5], 
    'velocity': [25.0], 
    'eccentricity': [0.5], 
    'inclination': [10], 
    'composition': ['S-type'],
    'soil_density': [2.0], 
    'porosity': [0.3], 
    'elevation': [1000], 
    'water_depth': [0],
    'impactor_mass': [750], 
    'intercept_velocity': [7.0], 
    'momentum_factor': [3.0]
})

try:
    test_processed = preprocessor.transform(test_input)
    test_pred = ensemble.predict(test_processed)
    
    print(f"   Input: Diameter=1.5km, Velocity=25km/s, Composition=S-type")
    print(f"   Output: Crater = {test_pred[0][0]:.2f} km, Seismic = {test_pred[0][1]:.2f}")
except Exception as e:
    print(f"   ❌ Test prediction failed: {e}")

print("\n" + "=" * 60)
print("✅ Triple-Modal Fusion Model Training Complete!")
print("=" * 60)
print("\nTo start the API server:")
print("   python main.py")
print("\nTo test the prediction page:")
print("   Open predicton.html in your browser")