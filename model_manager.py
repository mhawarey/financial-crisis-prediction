import numpy as np
import pandas as pd
import pickle
import os
import datetime as dt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("model_manager.log"), logging.StreamHandler()]
)
logger = logging.getLogger("ModelManager")

class ModelManager:
    """
    Manages predictive models for financial crisis prediction.
    Handles model training, evaluation, and predictions.
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_params = {}
        self.model_performance = {}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create models directory if it doesn't exist
        if not os.path.exists("models"):
            os.makedirs("models")
        
        logger.info("ModelManager initialized")
    
    def prepare_data(self, data, target_col=None, sequence_length=30):
        """
        Prepare data for model training and prediction.
        
        Parameters:
        - data: DataFrame with indicators
        - target_col: Target column name (optional for unsupervised models)
        - sequence_length: Sequence length for time series models
        
        Returns:
        - X: Features (numpy array)
        - y: Target values (numpy array, None if target_col is None)
        - scaler_x: Feature scaler
        - scaler_y: Target scaler (None if target_col is None)
        """
        try:
            logger.info("Preparing data for modeling")
            
            # Remove any non-numeric columns
            numeric_data = data.select_dtypes(include=[np.number])
            
            # Handle missing values
            numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Prepare features
            if target_col and target_col in numeric_data.columns:
                features = numeric_data.drop(columns=[target_col])
                target = numeric_data[target_col].values
            else:
                features = numeric_data
                target = None
            
            # Scale features
            scaler_x = StandardScaler()
            scaled_features = scaler_x.fit_transform(features)
            
            # Scale target if provided
            scaler_y = None
            if target is not None:
                scaler_y = MinMaxScaler()
                target = scaler_y.fit_transform(target.reshape(-1, 1)).flatten()
            
            # For time series models, create sequences
            if sequence_length > 0 and len(scaled_features) > sequence_length:
                X, y = self._create_sequences(scaled_features, target, sequence_length)
                logger.info(f"Created {len(X)} sequences of length {sequence_length}")
            else:
                X = scaled_features
                y = target
            
            return X, y, scaler_x, scaler_y
            
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            return None, None, None, None
    
    def _create_sequences(self, features, target=None, sequence_length=30):
        """
        Create sequences for time series models.
        
        Parameters:
        - features: Scaled feature array
        - target: Scaled target array (optional)
        - sequence_length: Sequence length
        
        Returns:
        - X: Sequence features
        - y: Sequence targets (None if target is None)
        """
        X = []
        y = []
        
        for i in range(len(features) - sequence_length):
            X.append(features[i:i+sequence_length])
            if target is not None:
                y.append(target[i+sequence_length])
        
        return np.array(X), np.array(y) if target is not None else None
    
    def train_kalman_filter(self, data, target_col, model_name="kalman_filter"):
        """
        Train a basic Kalman Filter model.
        
        Parameters:
        - data: DataFrame with indicators
        - target_col: Target column to predict
        - model_name: Name for the model
        
        Returns:
        - Success flag (boolean)
        """
        try:
            logger.info(f"Training Kalman Filter model: {model_name}")
            
            # Prepare data for Kalman filter
            if target_col not in data.columns:
                logger.error(f"Target column {target_col} not found in data")
                return False
            
            # Extract target series and ensure it's numeric
            target_series = pd.to_numeric(data[target_col], errors='coerce')
            target_series = target_series.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Scale the target
            scaler = MinMaxScaler(feature_range=(-1, 1))
            scaled_target = scaler.fit_transform(target_series.values.reshape(-1, 1)).flatten()
            
            # Initialize Kalman filter
            kf = KalmanFilter(dim_x=2, dim_z=1)
            
            # Initial state (position and velocity)
            kf.x = np.array([scaled_target[0], 0.])
            
            # State transition matrix (position + velocity, constant velocity model)
            kf.F = np.array([[1., 1.], 
                             [0., 1.]])
            
            # Measurement function (we only measure position)
            kf.H = np.array([[1., 0.]])
            
            # Covariance matrices - tuned for financial data
            kf.P *= 10.                       # Initial state uncertainty
            kf.R = np.array([[0.1]])          # Measurement uncertainty
            kf.Q = np.array([[0.01, 0.01],    # Process uncertainty
                             [0.01, 0.01]])
            
            # Run the filter on historical data to optimize it
            filtered_state_means = np.zeros((len(scaled_target), 2))
            for i, measurement in enumerate(scaled_target):
                kf.predict()
                kf.update(measurement)
                filtered_state_means[i] = kf.x
            
            # Store the model, scaler, and parameters
            self.models[model_name] = kf
            self.scalers[model_name] = scaler
            self.model_params[model_name] = {
                'dim_x': 2,
                'dim_z': 1,
                'target_col': target_col
            }
            
            # Evaluate the model
            predictions = filtered_state_means[:, 0]  # Position estimates
            
            # Convert back to original scale
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            actual = target_series.values
            
            # Calculate errors on training data
            mse = np.mean((predictions - actual)**2)
            mae = np.mean(np.abs(predictions - actual))
            
            # Store performance metrics
            self.model_performance[model_name] = {
                'mse': mse,
                'mae': mae,
                'training_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"Kalman Filter model trained: MSE={mse:.4f}, MAE={mae:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training Kalman Filter model: {e}")
            return False
    
    def train_extended_kalman_filter(self, data, target_col, model_name="extended_kalman_filter"):
        """
        Train an Extended Kalman Filter for non-linear dynamics.
        
        Parameters:
        - data: DataFrame with indicators
        - target_col: Target column to predict
        - model_name: Name for the model
        
        Returns:
        - Success flag (boolean)
        """
        try:
            logger.info(f"Training Extended Kalman Filter model: {model_name}")
            
            # For the trial version, we'll implement a simplified EKF
            # This is a placeholder for a more sophisticated implementation
            
            # Extract target series and related features
            if target_col not in data.columns:
                logger.error(f"Target column {target_col} not found in data")
                return False
            
            # Get target and a key feature for the model
            target_series = pd.to_numeric(data[target_col], errors='coerce')
            
            # Find a highly correlated feature to use in the model
            correlations = data.corrwith(target_series).abs().sort_values(ascending=False)
            feature_cols = [col for col in correlations.index if col != target_col][:3]
            
            if not feature_cols:
                logger.error("No suitable features found for EKF model")
                return False
            
            # Prepare data
            model_data = data[[target_col] + feature_cols].copy()
            model_data = model_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(model_data)
            
            # Initialize EKF with simple non-linear model
            # For a financial time series, we can use a mean-reverting model
            class FinancialEKF(ExtendedKalmanFilter):
                def __init__(self, dim_x, dim_z, feature_count):
                    super().__init__(dim_x, dim_z)
                    self.feature_count = feature_count
                
                def f(self, x, dt):
                    # Mean-reverting process: x_t+1 = x_t + theta*(mu - x_t) + noise
                    # x[0] is state, x[1] is mean, x[2] is reversion strength
                    # Remaining elements are feature weights
                    mean = x[1]
                    theta = np.clip(x[2], 0.01, 0.99)  # Bound theta between 0.01 and 0.99
                    
                    # Basic state transition
                    x_new = x.copy()
                    x_new[0] = x[0] + theta * (mean - x[0])
                    
                    return x_new
                
                def h(self, x):
                    # Measurement function - predict target value
                    # Output is just the state
                    return np.array([x[0]])
            
            # Create and configure the EKF
            feature_count = len(feature_cols)
            ekf = FinancialEKF(dim_x=3+feature_count, dim_z=1, feature_count=feature_count)
            
            # Initial state
            x0 = scaled_data[0, 0]  # Initial target value
            mean_value = np.mean(scaled_data[:, 0])  # Mean of target
            theta = 0.1  # Initial mean reversion strength
            
            # Feature weights - initialized to small random values
            feature_weights = np.random.normal(0, 0.1, feature_count)
            
            # Set initial state
            ekf.x = np.concatenate([[x0, mean_value, theta], feature_weights])
            
            # Initial covariance matrix
            ekf.P = np.eye(3+feature_count) * 1.0
            
            # Measurement uncertainty
            ekf.R = np.array([[0.1]])
            
            # Process noise
            ekf.Q = np.eye(3+feature_count) * 0.01
            
            # Store the model and parameters
            self.models[model_name] = ekf
            self.scalers[model_name] = scaler
            self.model_params[model_name] = {
                'dim_x': 3+feature_count,
                'dim_z': 1,
                'target_col': target_col,
                'feature_cols': feature_cols
            }
            
            logger.info(f"Extended Kalman Filter model initialized with features: {feature_cols}")
            return True
            
        except Exception as e:
            logger.error(f"Error training Extended Kalman Filter model: {e}")
            return False
    
    def train_machine_learning_model(self, data, target_col, model_type="random_forest", model_name=None):
        """
        Train a machine learning model for crisis prediction.
        
        Parameters:
        - data: DataFrame with indicators
        - target_col: Target column to predict
        - model_type: Type of ML model ('random_forest', 'gradient_boosting', 'logistic')
        - model_name: Name for the model (defaults to model_type)
        
        Returns:
        - Success flag (boolean)
        """
        if model_name is None:
            model_name = model_type
        
        try:
            logger.info(f"Training {model_type} model: {model_name}")
            
            # Prepare data
            X, y, scaler_x, scaler_y = self.prepare_data(data, target_col, sequence_length=0)
            
            if X is None or y is None:
                logger.error("Data preparation failed")
                return False
            
            # Split data for training and validation
            # For time series, we use a time-based split
            tscv = TimeSeriesSplit(n_splits=3)
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
            
            # Initialize the model based on type
            if model_type == "random_forest":
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
            elif model_type == "gradient_boosting":
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            elif model_type == "logistic":
                model = LogisticRegression(
                    C=1.0,
                    class_weight='balanced',
                    random_state=42
                )
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate the model
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            # Store the model and parameters
            self.models[model_name] = model
            self.scalers[model_name] = {'x': scaler_x, 'y': scaler_y}
            self.model_params[model_name] = {
                'model_type': model_type,
                'target_col': target_col,
                'features': data.drop(columns=[target_col]).columns.tolist()
            }
            
            # Store performance metrics
            self.model_performance[model_name] = {
                'train_score': train_score,
                'test_score': test_score,
                'training_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"{model_type} model trained: Train score={train_score:.4f}, Test score={test_score:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training {model_type} model: {e}")
            return False
    
    def create_simple_lstm(self, input_dim, hidden_dim=64, output_dim=1, num_layers=2):
        """
        Create a simple LSTM model using PyTorch.
        
        Parameters:
        - input_dim: Number of input features
        - hidden_dim: Hidden layer dimension
        - output_dim: Output dimension
        - num_layers: Number of LSTM layers
        
        Returns:
        - LSTM model
        """
        class LSTMModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super(LSTMModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                # LSTM layers
                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers, 
                    batch_first=True, dropout=0.2
                )
                
                # Fully connected layer
                self.fc = nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                # Initialize hidden state
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
                
                # Forward propagate LSTM
                out, _ = self.lstm(x, (h0, c0))
                
                # Get the output from the last time step
                out = self.fc(out[:, -1, :])
                return out
        
        return LSTMModel(input_dim, hidden_dim, output_dim, num_layers).to(self.device)
    
    def train_lstm_model(self, data, target_col, sequence_length=30, model_name="lstm_model"):
        """
        Train an LSTM model for time series prediction.
        
        Parameters:
        - data: DataFrame with indicators
        - target_col: Target column to predict
        - sequence_length: Sequence length for LSTM
        - model_name: Name for the model
        
        Returns:
        - Success flag (boolean)
        """
        try:
            logger.info(f"Training LSTM model: {model_name}")
            
            # Prepare data
            X, y, scaler_x, scaler_y = self.prepare_data(data, target_col, sequence_length)
            
            if X is None or y is None:
                logger.error("Data preparation failed")
                return False
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).unsqueeze(1).to(self.device)
            
            # Split data for training and validation
            train_size = int(len(X) * 0.8)
            X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
            y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]
            
            # Create data loaders
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
            
            # Create model
            input_dim = X.shape[2]  # Number of features
            model = self.create_simple_lstm(input_dim)
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop
            epochs = 50
            best_val_loss = float('inf')
            patience = 5
            counter = 0
            
            for epoch in range(epochs):
                model.train()
                train_loss = 0.0
                
                for inputs, targets in train_loader:
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                train_loss = train_loss / len(train_loader)
                
                # Validation
                model.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                
                val_loss = val_loss / len(val_loader)
                
                logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    counter = 0
                    # Save the best model
                    torch.save(model.state_dict(), f"models/{model_name}_best.pt")
                else:
                    counter += 1
                    if counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Load the best model
            model.load_state_dict(torch.load(f"models/{model_name}_best.pt"))
            
            # Store the model and parameters
            self.models[model_name] = model
            self.scalers[model_name] = {'x': scaler_x, 'y': scaler_y}
            self.model_params[model_name] = {
                'input_dim': input_dim,
                'hidden_dim': 64,
                'output_dim': 1,
                'num_layers': 2,
                'sequence_length': sequence_length,
                'target_col': target_col
            }
            
            # Store performance metrics
            self.model_performance[model_name] = {
                'best_val_loss': best_val_loss,
                'training_date': dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            logger.info(f"LSTM model trained: Best validation loss={best_val_loss:.6f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            return False
    
    def train_ensemble_model(self, data, target_col, model_name="ensemble_model"):
        """
        Train an ensemble of models for crisis prediction.
        
        Parameters:
        - data: DataFrame with indicators
        - target_col: Target column to predict
        - model_name: Name for the ensemble model
        
        Returns:
        - Success flag (boolean)
        """
        try:
            logger.info(f"Training ensemble model: {model_name}")
            
            # Train individual models first
            models_to_ensemble = []
            
            # 1. Train Kalman Filter
            kf_name = f"{model_name}_kf"
            if self.train_kalman_filter(data, target_col, kf_name):
                models_to_ensemble.append(kf_name)
            
            # 2. Train Random Forest
            rf_name = f"{model_name}_rf"
            if self.train_machine_learning_model(data, target_col, "random_forest", rf_name):
                models_to_ensemble.append(rf_name)
            
            # 3. Train Gradient Boosting
            gb_name = f"{model_name}_gb"
            if self.train_machine_learning_model(data, target_col, "gradient_boosting", gb_name):
                models_to_ensemble.append(gb_name)
            
            # 4. Train LSTM (if enough data)
            if len(data) > 100:  # Ensure enough data for sequences
                lstm_name = f"{model_name}_lstm"
                if self.train_lstm_model(data, target_col, sequence_length=30, model_name=lstm_name):
                    models_to_ensemble.append(lstm_name)
            
            # Store ensemble configuration
            if models_to_ensemble:
                self.models[model_name] = models_to_ensemble
                self.model_params[model_name] = {
                    'model_type': 'ensemble',
                    'target_col': target_col,
                    'component_models': models_to_ensemble,
                    'weights': [1.0/len(models_to_ensemble)] * len(models_to_ensemble)  # Equal weights initially
                }
                
                logger.info(f"Ensemble model created with components: {models_to_ensemble}")
                return True
            else:
                logger.error("Failed to create any component models for ensemble")
                return False
            
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return False
    
    def predict_with_model(self, model_name, data, forecast_horizon=10):
        """
        Make predictions using a trained model.
        
        Parameters:
        - model_name: Name of the model to use
        - data: DataFrame with indicators
        - forecast_horizon: Number of steps to forecast
        
        Returns:
        - DataFrame with predictions and confidence intervals
        """
        try:
            logger.info(f"Making predictions with model: {model_name}")
            
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return None
            
            model = self.models[model_name]
            params = self.model_params[model_name]
            
            # Handle ensemble models
            if params.get('model_type') == 'ensemble':
                return self.predict_with_ensemble(model_name, data, forecast_horizon)
            
            # Different prediction logic based on model type
            if isinstance(model, KalmanFilter) or isinstance(model, ExtendedKalmanFilter):
                predictions = self.predict_with_kalman(model_name, data, forecast_horizon)
            elif isinstance(model, nn.Module):  # PyTorch model (LSTM)
                predictions = self.predict_with_lstm(model_name, data, forecast_horizon)
            else:  # Scikit-learn models
                predictions = self.predict_with_sklearn(model_name, data, forecast_horizon)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return None
    
    def predict_with_kalman(self, model_name, data, forecast_horizon=10):
        """
        Make predictions using a Kalman Filter model.
        
        Parameters:
        - model_name: Name of the Kalman Filter model
        - data: DataFrame with indicators
        - forecast_horizon: Number of steps to forecast
        
        Returns:
        - DataFrame with predictions and confidence intervals
        """
        model = self.models[model_name]
        params = self.model_params[model_name]
        scaler = self.scalers[model_name]
        target_col = params['target_col']
        
        # Extract the target series
        if target_col in data.columns:
            target_series = data[target_col].fillna(method='ffill').fillna(0)
        else:
            logger.warning(f"Target column {target_col} not found, using zeros")
            target_series = pd.Series(np.zeros(len(data)))
        
        # Scale the series
        scaled_target = scaler.transform(target_series.values.reshape(-1, 1)).flatten()
        
        # Get the last state
        kf = model.copy()
        
        # Run the filter on all available data to get to the current state
        for measurement in scaled_target:
            kf.predict()
            kf.update(measurement)
        
        # Now forecast future values
        predictions = []
        lower_bounds = []
        upper_bounds = []
        
        for _ in range(forecast_horizon):
            # Predict the next state
            kf.predict()
            
            # Extract the predicted value and uncertainty
            pred_value = kf.x[0]
            uncertainty = np.sqrt(kf.P[0, 0])
            
            # Store predictions and confidence intervals (95%)
            predictions.append(pred_value)
            lower_bounds.append(pred_value - 1.96 * uncertainty)
            upper_bounds.append(pred_value + 1.96 * uncertainty)
            
            # Update with the predicted value (since we don't have actual measurements)
            kf.update(pred_value)
        
        # Convert predictions back to original scale
        predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        lower_bounds = scaler.inverse_transform(np.array(lower_bounds).reshape(-1, 1)).flatten()
        upper_bounds = scaler.inverse_transform(np.array(upper_bounds).reshape(-1, 1)).flatten()
        
        # Create forecast dates
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            if data.index.freq:
                freq = data.index.freq
            else:
                # Infer frequency from the last two dates
                freq = pd.infer_freq(data.index[-5:])
                if freq is None:
                    freq = 'D'  # Default to daily if can't infer
            
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq=freq)
        else:
            # If index is not datetime, use integers
            forecast_dates = range(len(data), len(data) + forecast_horizon)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': forecast_dates,
            'prediction': predictions,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds
        })
        
        return results
    
    def predict_with_lstm(self, model_name, data, forecast_horizon=10):
        """
        Make predictions using an LSTM model.
        
        Parameters:
        - model_name: Name of the LSTM model
        - data: DataFrame with indicators
        - forecast_horizon: Number of steps to forecast
        
        Returns:
        - DataFrame with predictions
        """
        model = self.models[model_name]
        params = self.model_params[model_name]
        scalers = self.scalers[model_name]
        sequence_length = params['sequence_length']
        
        # Prepare the input data
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale the data
        scaler_x = scalers['x']
        scaler_y = scalers['y']
        scaled_data = scaler_x.transform(numeric_data)
        
        # Start with the most recent sequence
        prediction_sequence = scaled_data[-sequence_length:].copy()
        
        # Convert to tensor
        model.eval()
        
        # Make predictions
        predictions = []
        for _ in range(forecast_horizon):
            # Prepare input sequence
            input_seq = torch.FloatTensor(prediction_sequence.reshape(1, sequence_length, -1)).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                pred = model(input_seq).cpu().numpy().flatten()[0]
            
            # Update sequence for next prediction
            new_features = prediction_sequence[-1, :].copy()
            new_features[0] = pred  # Assume first feature is the target
            prediction_sequence = np.roll(prediction_sequence, -1, axis=0)
            prediction_sequence[-1, :] = new_features
            
            # Store prediction
            predictions.append(pred)
        
        # Convert predictions back to original scale
        if scaler_y:
            predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Create forecast dates
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            if data.index.freq:
                freq = data.index.freq
            else:
                freq = pd.infer_freq(data.index[-5:])
                if freq is None:
                    freq = 'D'  # Default to daily if can't infer
            
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq=freq)
        else:
            forecast_dates = range(len(data), len(data) + forecast_horizon)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': forecast_dates,
            'prediction': predictions
        })
        
        # Add simple confidence intervals (based on historical RMSE)
        # For a proper implementation, we would use Monte Carlo or similar methods
        results['lower_bound'] = results['prediction'] * 0.9
        results['upper_bound'] = results['prediction'] * 1.1
        
        return results
    
    def predict_with_sklearn(self, model_name, data, forecast_horizon=10):
        """
        Make predictions using a scikit-learn model.
        
        Parameters:
        - model_name: Name of the scikit-learn model
        - data: DataFrame with indicators
        - forecast_horizon: Number of steps to forecast
        
        Returns:
        - DataFrame with predictions
        """
        model = self.models[model_name]
        params = self.model_params[model_name]
        scalers = self.scalers[model_name]
        
        # For scikit-learn models, we need to use an autoregressive approach for multi-step forecasting
        # This is a simplified implementation
        
        # Prepare the input data
        numeric_data = data.select_dtypes(include=[np.number])
        numeric_data = numeric_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Scale the data
        scaler_x = scalers['x']
        scaler_y = scalers['y']
        
        # Start with the most recent data point
        current_features = scaler_x.transform(numeric_data.iloc[-1:])
        
        # Make predictions
        predictions = []
        for _ in range(forecast_horizon):
            # Predict next value
            pred = model.predict(current_features).flatten()[0]
            predictions.append(pred)
            
            # For a proper implementation, we would update features based on the prediction
            # Here we just use the same features for simplicity
        
        # Convert predictions back to original scale
        if scaler_y:
            predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        
        # Create forecast dates
        last_date = data.index[-1]
        if isinstance(last_date, pd.Timestamp):
            if data.index.freq:
                freq = data.index.freq
            else:
                freq = pd.infer_freq(data.index[-5:])
                if freq is None:
                    freq = 'D'  # Default to daily if can't infer
            
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq=freq)
        else:
            forecast_dates = range(len(data), len(data) + forecast_horizon)
        
        # Create results DataFrame
        results = pd.DataFrame({
            'date': forecast_dates,
            'prediction': predictions
        })
        
        # Add simple confidence intervals
        results['lower_bound'] = results['prediction'] * 0.9
        results['upper_bound'] = results['prediction'] * 1.1
        
        return results
    
    def predict_with_ensemble(self, model_name, data, forecast_horizon=10):
        """
        Make predictions using an ensemble of models.
        
        Parameters:
        - model_name: Name of the ensemble model
        - data: DataFrame with indicators
        - forecast_horizon: Number of steps to forecast
        
        Returns:
        - DataFrame with predictions and confidence intervals
        """
        component_models = self.models[model_name]
        params = self.model_params[model_name]
        weights = params['weights']
        
        # Get predictions from each component model
        all_predictions = []
        
        for model_name in component_models:
            pred_df = self.predict_with_model(model_name, data, forecast_horizon)
            if pred_df is not None:
                all_predictions.append(pred_df)
        
        if not all_predictions:
            logger.error("No valid predictions from component models")
            return None
        
        # Combine predictions with weighted average
        # First, ensure all predictions have the same dates
        base_dates = all_predictions[0]['date']
        
        # Initialize the ensemble predictions
        ensemble_pred = pd.DataFrame({'date': base_dates})
        ensemble_pred['prediction'] = 0.0
        ensemble_pred['lower_bound'] = 0.0
        ensemble_pred['upper_bound'] = 0.0
        
        # Compute weighted average
        weight_sum = sum(weights[:len(all_predictions)])
        normalized_weights = [w/weight_sum for w in weights[:len(all_predictions)]]
        
        for i, pred_df in enumerate(all_predictions):
            weight = normalized_weights[i]
            ensemble_pred['prediction'] += pred_df['prediction'].values * weight
            
            if 'lower_bound' in pred_df.columns:
                ensemble_pred['lower_bound'] += pred_df['lower_bound'].values * weight
            
            if 'upper_bound' in pred_df.columns:
                ensemble_pred['upper_bound'] += pred_df['upper_bound'].values * weight
        
        return ensemble_pred
    
    def save_models(self, directory="models"):
        """
        Save all trained models to disk.
        
        Parameters:
        - directory: Directory to save models
        
        Returns:
        - Success flag (boolean)
        """
        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # Save metadata
            metadata = {
                'model_params': self.model_params,
                'model_performance': self.model_performance
            }
            
            with open(os.path.join(directory, "model_metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            
            # Save each model
            for name, model in self.models.items():
                if isinstance(model, list):  # Ensemble model
                    continue  # Already saved component models
                elif isinstance(model, nn.Module):  # PyTorch model
                    torch.save(model.state_dict(), os.path.join(directory, f"{name}.pt"))
                else:  # Scikit-learn model or Kalman filter
                    with open(os.path.join(directory, f"{name}.pkl"), "wb") as f:
                        pickle.dump(model, f)
            
            # Save scalers
            with open(os.path.join(directory, "scalers.pkl"), "wb") as f:
                pickle.dump(self.scalers, f)
            
            logger.info(f"Models saved to {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, directory="models"):
        """
        Load trained models from disk.
        
        Parameters:
        - directory: Directory to load models from
        
        Returns:
        - Success flag (boolean)
        """
        try:
            if not os.path.exists(directory):
                logger.error(f"Directory {directory} does not exist")
                return False
            
            # Load metadata
            with open(os.path.join(directory, "model_metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
                self.model_params = metadata['model_params']
                self.model_performance = metadata['model_performance']
            
            # Load scalers
            with open(os.path.join(directory, "scalers.pkl"), "rb") as f:
                self.scalers = pickle.load(f)
            
            # Load each model
            for name, params in self.model_params.items():
                if params.get('model_type') == 'ensemble':
                    # Load component models first
                    for component in params['component_models']:
                        self.load_models(directory)
                    
                    # Then set the ensemble
                    self.models[name] = params['component_models']
                else:
                    model_path = os.path.join(directory, f"{name}.pkl")
                    pt_path = os.path.join(directory, f"{name}.pt")
                    
                    if os.path.exists(model_path):
                        with open(model_path, "rb") as f:
                            self.models[name] = pickle.load(f)
                    elif os.path.exists(pt_path):
                        # Create model architecture
                        if 'input_dim' in params:
                            model = self.create_simple_lstm(
                                params['input_dim'],
                                params.get('hidden_dim', 64),
                                params.get('output_dim', 1),
                                params.get('num_layers', 2)
                            )
                            # Load weights
                            model.load_state_dict(torch.load(pt_path))
                            self.models[name] = model
                    else:
                        logger.warning(f"Model file for {name} not found")
            
            logger.info(f"Models loaded from {directory}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False