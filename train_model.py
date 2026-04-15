import numpy as np
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime
from typing import Tuple, Dict, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ImpactModelTrainer:
    """Asteroid impact consequences prediction model trainer."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models_dir = 'models'
        self.results_dir = 'results'
        self.scaler = StandardScaler()
        self.ensemble = None
        self.training_history = {}
        
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
    
    def generate_synthetic_data(self, samples: int = 1000, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic asteroid impact data with more realistic physics.
        
        Args:
            samples: Number of samples to generate
            noise_level: Amount of noise to add (0-1)
        
        Returns:
            X: Features [diameter, velocity, composition]
            y: Targets [crater_size, seismic_magnitude]
        """
        np.random.seed(self.random_state)
        
        # Generate features with more realistic distributions
        X = np.zeros((samples, 3))
        
        # Diameter (km) - lognormal distribution (more small asteroids)
        X[:, 0] = np.random.lognormal(mean=-1.0, sigma=1.0, size=samples)
        X[:, 0] = np.clip(X[:, 0], 0.1, 10.0)  # Clip to realistic range
        
        # Velocity (km/s) - normal distribution with higher mean
        X[:, 1] = np.random.normal(loc=17.0, scale=8.0, size=samples)
        X[:, 1] = np.clip(X[:, 1], 5.0, 70.0)  # Clip to realistic range
        
        # Composition types (0-2: rocky, metallic, icy)
        X[:, 2] = np.random.choice([0, 1, 2], size=samples, p=[0.5, 0.3, 0.2])
        
        # Physics-based synthetic targets with composition effects
        # Crater size (km) - energy-based scaling
        kinetic_energy = 0.5 * (4/3 * np.pi * (X[:, 0]/2)**3) * (X[:, 1]**2)
        
        # Composition modifiers
        comp_modifier = np.array([1.0 if c == 0 else 1.2 if c == 1 else 0.8 for c in X[:, 2]])
        
        crater = 0.5 * np.power(kinetic_energy * comp_modifier, 0.3) * 1000
        crater += np.random.normal(0, noise_level * crater.mean(), samples)
        
        # Seismic magnitude - moment magnitude scale approximation
        seismic = 4.0 + 0.5 * np.log10(kinetic_energy) + 0.3 * (X[:, 2] - 1)
        seismic += np.random.normal(0, noise_level * 0.5, samples)
        
        y = np.column_stack((crater, seismic))
        
        logger.info(f"Generated {samples} samples. Features shape: {X.shape}, Targets shape: {y.shape}")
        logger.info(f"Feature ranges - Diameter: [{X[:,0].min():.2f}, {X[:,0].max():.2f}], "
                   f"Velocity: [{X[:,1].min():.2f}, {X[:,1].max():.2f}]")
        
        return X, y
    
    def create_ensemble(self) -> MultiOutputRegressor:
        """
        Create the ensemble model with optimized hyperparameters.
        """
        # Random Forest for handling non-linear relationships
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        # XGBoost for gradient boosting
        xgb_model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Voting ensemble with weights
        voting_regressor = VotingRegressor(
            estimators=[
                ('rf', rf_model),
                ('xgb', xgb_model)
            ],
            weights=[1, 1]  # Equal weighting
        )
        
        # Wrap for multi-output
        self.ensemble = MultiOutputRegressor(
            voting_regressor,
            n_jobs=-1
        )
        
        logger.info("Created ensemble with RandomForest and XGBoost models")
        return self.ensemble
    
    def preprocess_data(self, X: np.ndarray, y: np.ndarray, fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features and targets.
        
        Args:
            X: Features
            y: Targets
            fit_scaler: Whether to fit the scaler or use existing
        
        Returns:
            Preprocessed X and y
        """
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # For multi-output, keep y as is (can add target scaling if needed)
        return X_scaled, y
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the model with proper validation.
        
        Args:
            X: Features
            y: Targets
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with training metrics
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Preprocess
        X_train_scaled, y_train = self.preprocess_data(X_train, y_train, fit_scaler=True)
        X_test_scaled, y_test = self.preprocess_data(X_test, y_test, fit_scaler=False)
        
        # Create and train ensemble
        if self.ensemble is None:
            self.create_ensemble()
        
        logger.info("Starting model training...")
        start_time = datetime.now()
        
        self.ensemble.fit(X_train_scaled, y_train)
        
        training_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Evaluate
        metrics = self.evaluate(X_test_scaled, y_test)
        metrics['training_time'] = training_time
        metrics['train_samples'] = len(X_train)
        metrics['test_samples'] = len(X_test)
        
        # Cross-validation
        cv_scores = self.cross_validate(X_train_scaled, y_train)
        metrics['cv_mean'] = cv_scores['mean']
        metrics['cv_std'] = cv_scores['std']
        
        self.training_history = metrics
        return metrics
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.ensemble.predict(X_test)
        
        metrics = {}
        target_names = ['Crater Size (km)', 'Seismic Magnitude']
        
        for i, name in enumerate(target_names):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            metrics[f'{name}_RMSE'] = rmse
            metrics[f'{name}_MAE'] = mae
            metrics[f'{name}_R2'] = r2
            
            logger.info(f"{name} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        # Overall metrics
        metrics['overall_RMSE'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['overall_MAE'] = mean_absolute_error(y_test, y_pred)
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Targets
            cv_folds: Number of cross-validation folds
        
        Returns:
            Dictionary with CV scores
        """
        # For multi-output, we need to handle separately
        scores = []
        for i in range(y.shape[1]):
            cv_scores = cross_val_score(
                self.ensemble.estimators_[i], X, y[:, i],
                cv=cv_folds, scoring='r2', n_jobs=-1
            )
            scores.append(cv_scores)
        
        scores = np.array(scores)
        
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'per_target_mean': np.mean(scores, axis=1),
            'per_target_std': np.std(scores, axis=1)
        }
    
    def save_model(self, filename: str = 'aegis_ensemble.joblib') -> str:
        """
        Save the trained model and scaler.
        
        Args:
            filename: Name of the model file
        
        Returns:
            Path to the saved model
        """
        if self.ensemble is None:
            raise ValueError("No trained model to save")
        
        model_path = os.path.join(self.models_dir, filename)
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        
        # Save model
        joblib.dump(self.ensemble, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save scaler
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Scaler saved to {scaler_path}")
        
        # Save training history
        if self.training_history:
            history_path = os.path.join(self.results_dir, 'training_history.json')
            # Convert numpy values to Python types for JSON serialization
            history_json = json.dumps(self.training_history, default=str, indent=2)
            with open(history_path, 'w') as f:
                f.write(history_json)
            logger.info(f"Training history saved to {history_path}")
        
        return model_path
    
    def load_model(self, filename: str = 'aegis_ensemble.joblib'):
        """
        Load a trained model.
        
        Args:
            filename: Name of the model file
        """
        model_path = os.path.join(self.models_dir, filename)
        scaler_path = os.path.join(self.models_dir, 'scaler.joblib')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.ensemble = joblib.load(model_path)
        
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        logger.info(f"Model loaded from {model_path}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features
        
        Returns:
            Predictions
        """
        if self.ensemble is None:
            raise ValueError("Model not trained or loaded")
        
        X_scaled = self.preprocess_data(X, np.zeros((X.shape[0], 2)), fit_scaler=False)[0]
        return self.ensemble.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance from the ensemble.
        
        Returns:
            Dictionary with feature importance information
        """
        if self.ensemble is None:
            raise ValueError("Model not trained or loaded")
        
        feature_names = ['Diameter (km)', 'Velocity (km/s)', 'Composition']
        
        # Extract importance from Random Forest (first output)
        rf_model = self.ensemble.estimators_[0].named_estimators_['rf']
        rf_importance = rf_model.feature_importances_
        
        # XGBoost importance
        xgb_model = self.ensemble.estimators_[0].named_estimators_['xgb']
        xgb_importance = xgb_model.feature_importances_
        
        # Average importance
        avg_importance = (rf_importance + xgb_importance) / 2
        
        importance_dict = {
            'random_forest': {name: float(imp) for name, imp in zip(feature_names, rf_importance)},
            'xgboost': {name: float(imp) for name, imp in zip(feature_names, xgb_importance)},
            'average': {name: float(imp) for name, imp in zip(feature_names, avg_importance)}
        }
        
        return importance_dict


def main():
    """Main training pipeline."""
    
    logger.info("🚀 Starting AEGIS Impact Prediction Model Training")
    
    # Initialize trainer
    trainer = ImpactModelTrainer(random_state=42)
    
    # Generate data
    X, y = trainer.generate_synthetic_data(samples=2000, noise_level=0.1)
    
    # Train model
    metrics = trainer.train(X, y, test_size=0.2)
    
    # Save model
    model_path = trainer.save_model()
    
    # Get feature importance
    importance = trainer.get_feature_importance()
    logger.info(f"Feature importance: {importance['average']}")
    
    # Example prediction
    new_asteroid = np.array([[1.5, 25.0, 1]])  # 1.5km, 25km/s, metallic
    prediction = trainer.predict(new_asteroid)
    logger.info(f"Example prediction - Crater: {prediction[0,0]:.2f} km, "
               f"Seismic: {prediction[0,1]:.2f}")
    
    logger.info("✅ Training pipeline completed successfully")


if __name__ == "__main__":
    main()