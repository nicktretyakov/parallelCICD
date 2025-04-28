# Placeholder for ML model for task prediction or resource optimization.

import logging
import pandas as pd
from sklearn.linear_model import LinearRegression # Example regression model
from sklearn.model_selection import train_test_split # Example for splitting data
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import joblib # For saving/loading models
import os # To check if model file exists
import json # Import json for deserializing parameters

class TaskPredictor:
    def __init__(self):
        self._logger = logging.getLogger(__name__)
        self._model_pipeline = None # Use a pipeline to include preprocessing and model
        self._is_trained = False
        self._data = None # Placeholder for loaded historical data (full dataset)
        self._training_data = None # Data filtered for successful runs, used for training duration model
        self._trained_features = None # Store the list of feature columns used for training
        self._numerical_features_means = None # Store means for imputation during prediction

    def load_data(self, data_source):
        """Loads historical task execution data from a CSV file.

        Assumes CSV has columns: 'task_name', 'status', 'duration_seconds', 'timestamp', 'parameters', 'cpu_usage', 'memory_usage'.
        'parameters' column is expected to be a JSON string.

        Args:
            data_source: Path to the CSV data file.
        """
        self._logger.info(f"Loading historical task data from {data_source}.")
        try:
            expected_columns = ['task_name', 'status', 'duration_seconds', 'timestamp', 'parameters', 'cpu_usage', 'memory_usage']
            if not os.path.exists(data_source):
                 self._logger.warning(f"Data file not found at {data_source}. Cannot load data.")
                 self._data = None
                 self._training_data = None
                 return

            self._data = pd.read_csv(data_source)

            if not all(col in self._data.columns for col in expected_columns):
                self._logger.error(f"Data file {data_source} is missing expected columns. Expected: {expected_columns}. Found: {list(self._data.columns)}")
                self._data = None
                self._training_data = None
                return

            # Deserialize the 'parameters' column from JSON strings to dictionaries
            self._data['parameters'] = self._data['parameters'].apply(lambda x: json.loads(x) if pd.notna(x) else {}, errors='coerce')

            # Keep only successful task runs for duration prediction training
            self._training_data = self._data[self._data['status'] == 'success'].copy()

            self._logger.info(f"Loaded {len(self._data)} total data points. {len(self._training_data)} successful tasks for training.")
            # TODO: Add more sophisticated data validation and cleaning (e.g., handling outliers, ensuring data types)

        except pd.errors.EmptyDataError:
            self._logger.warning(f"Data file {data_source} is empty or contains no successful tasks for training.")
            self._data = pd.DataFrame(columns=expected_columns) # Full data
            self._training_data = pd.DataFrame(columns=expected_columns) # Training data
        except Exception as e:
            self._logger.error(f"Error loading or processing historical data from {data_source}: {e}")
            self._data = None
            self._training_data = None


    def train_model(self):
        """Trains the ML model on the loaded data.

        The model is trained to predict task duration based on task characteristics.
        It uses task name, specific numerical parameters, and resource usage from
        historical *successful* task runs as features.
        """
        self._logger.info("Attempting to train ML model.")
        if self._training_data is None or self._training_data.empty:
             self._logger.warning("Cannot train model: no successful task data loaded or data is empty.")
             self._is_trained = False
             self._model_pipeline = None
             self._trained_features = None
             self._numerical_features_means = None
             return False

        # --- Feature Engineering and Target Definition ---
        # Target variable: task duration for successful runs
        target = self._training_data['duration_seconds'].copy()

        # Features:
        # - Task name (categorical)
        # - Numerical parameters extracted from the 'parameters' dictionary
        # - Historical Resource usage (numerical) - Note: For prediction, we might use requested/estimated resources.

        # Example: Extracting specific numerical parameters and handling potential missing/non-numeric values
        def safe_extract_numerical_param(params_dict, key):
             value = params_dict.get(key, None)
             if pd.isna(value): return None
             try: return float(value)
             except (ValueError, TypeError): return None

        # TODO: Dynamically identify or explicitly list relevant parameters across all task types
        # For example, iterate through all 'parameters' dicts in self._training_data to find common keys
        # For now, hardcoding the parameters found in your example config:
        relevant_params = ['dataset_size_gb', 'epochs', 'learning_rate', 'processing_intensity'] # Add all potential parameters here

        for param_key in relevant_params:
             # Create a new column for each relevant parameter, prefixing to avoid name conflicts
             self._training_data[f'param_{param_key}'] = self._training_data['parameters'].apply(lambda x: safe_extract_numerical_param(x, param_key))

        # Define the list of feature columns to use for training
        # Include 'task_name', extracted numerical parameters (now prefixed), and resource usage
        # Filter for columns that actually exist after extraction (in case some params weren't present in data)
        base_features = ['task_name', 'cpu_usage', 'memory_usage']
        param_features = [f'param_{param_key}' for param_key in relevant_params if f'param_{param_key}' in self._training_data.columns]

        feature_columns = base_features + param_features
        self._trained_features = feature_columns # Store feature names for prediction

        # Ensure all defined feature columns are available
        if not all(col in self._training_data.columns for col in feature_columns):
             missing = [col for col in feature_columns if col not in self._training_data.columns]
             self._logger.error(f"Required feature columns not found in training data after extraction: {missing}")
             self._is_trained = False
             self._model_pipeline = None
             self._trained_features = None
             self._numerical_features_means = None
             return False

        features = self._training_data[feature_columns].copy()

        # ---------------------------------------------------

        # --- Preprocessing Steps ---
        # Identify categorical and numerical features
        categorical_features = ['task_name']
        numerical_features = [col for col in feature_columns if col not in categorical_features]

        # Store the means/modes of features from the training data for imputation during prediction
        self._numerical_features_means = features[numerical_features].mean().to_dict()
        # TODO: Store modes for categorical features for imputation if necessary
        # TODO: Consider a more robust imputation strategy (e.g., K-nearest neighbors imputation)

        # Create preprocessing pipelines for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            # Impute numerical features with the mean from the training data
            # Using SimpleImputer is more robust than fillna before the pipeline
            ('imputer', 
             # Use the calculated means for imputation
             # SimpleImputer(strategy='constant', fill_value=0) # Or another strategy
             # A better approach is to calculate and use means *within* the pipeline fitting
             # For simplicity here, we assume means are pre-calculated
             # Alternatively, use a custom transformer or lambda within the pipeline if means are stored externally
             # For now, rely on fillna before prediction, but ideally, imputation is part of the pipeline
             'passthrough' # Placeholder - actual imputation will be handled before prediction for simplicity
             ),
            ('scaler', StandardScaler()) # Scale numerical features
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore')) # One-hot encode categorical features
        ])

        # Combine preprocessing steps using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ],
            remainder='drop' # Drop columns not specified in transformers
        )
        # --------------------------

        # --- Model Selection and Training ---
        # TODO: Split data into training and validation/test sets *before* fitting the pipeline
        # This allows for unbiased evaluation and prevents data leakage during preprocessing.
        # X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # TODO: Experiment with different regression models (e.g., RandomForestRegressor, GradientBoostingRegressor, SVR, XGBoost, LightGBM)
        # TODO: Implement hyperparameter tuning (e.g., GridSearchCV, RandomizedSearchCV) on the training set using cross-validation
        model = LinearRegression() # Example model - replace with your chosen model

        # Create the final pipeline: Preprocessing -> Model
        self._model_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

        try:
            self._logger.info("Training ML model pipeline.")
            # Fit the pipeline on the training data (or X_train if splitting)
            # self._model_pipeline.fit(X_train, y_train)
            self._model_pipeline.fit(features, target) # Fitting on full data for simplicity here

            # --------------------------------------

            # --- Model Evaluation ---
            self._logger.info("Evaluating ML model.")
            # Evaluate on the held-out test set if data was split
            # if 'X_test' in locals() and 'y_test' in locals():
            #     predictions = self._model_pipeline.predict(X_test)
            #     mse = mean_squared_error(y_test, predictions)
            #     r2 = r2_score(y_test, predictions)
            #     self._logger.info(f"Model Evaluation - Mean Squared Error: {mse:.4f}, R-squared: {r2:.4f}")
            # else:
            #     self._logger.warning("Data not split into train/test. Skipping formal evaluation on test set.")

            # TODO: Add cross-validation for more robust evaluation on the training data
            # from sklearn.model_selection import cross_val_score
            # scores = cross_val_score(self._model_pipeline, features, target, cv=5, scoring='neg_mean_squared_error')
            # rmse_scores = (-scores)**0.5
            # self._logger.info(f"Cross-validation RMSE: {rmse_scores.mean():.4f} (+/- {rmse_scores.std():.4f})")
            # ------------------------

            self._is_trained = True
            self._logger.info("ML model training complete.")
            return True

        except Exception as e:
            self._logger.error(f"Error during model training: {e}")
            self._is_trained = False
            self._model_pipeline = None
            self._trained_features = None
            self._numerical_features_means = None
            return False

    # --- Continuous Training Idea ---
    # Continuous training could be implemented as a separate scheduled task or triggered
    # by new data arriving in the history file. It would involve:
    # 1. Loading the latest data.
    # 2. Re-training the model (potentially with a sliding window of data).
    # 3. Saving the new model version.
    # The main pipeline execution would then load the latest trained model.
    # --------------------------------


    def _prepare_features_for_prediction(self, task, task_parameters):
         """Prepares features for a single task prediction, including imputation."
         if self._trained_features is None or self._numerical_features_means is None:
              self._logger.warning("Training features or means not available for prediction.")
              return None

         # --- Feature Extraction for Prediction ---
         # Extract the same parameters as used in training
         def safe_extract_numerical_param_predict(params_dict, key):
              value = params_dict.get(key, None)
              if pd.isna(value): return None
              try: return float(value)
              except (ValueError, TypeError): return None

         # TODO: Ensure this extraction logic matches train_model and handles all relevant_params
         sample_data = {
             'task_name': task.name,
             # Initialize with None for parameters that might not be in task_parameters
             **{f'param_{param_key}': None for param_key in [p.replace('param_', '') for p in self._trained_features if p.startswith('param_')]}
         }

         # Populate extracted parameters if available in task_parameters
         if task_parameters:
             for param_key in [p.replace('param_', '') for p in self._trained_features if p.startswith('param_')]:
                 sample_data[f'param_{param_key}'] = safe_extract_numerical_param_predict(task_parameters, param_key)

         # Resource usage features at prediction time would ideally be estimated/requested resources
         # If not available, you might impute or use defaults.
         # For this example, we'll use placeholders, assuming these would come from the task definition or a resource request.
         # TODO: Replace with actual estimated/requested resource values for the task.
         sample_data['cpu_usage'] = task_parameters.get('estimated_cpu_request', 1.0) if task_parameters else 1.0 # Placeholder
         sample_data['memory_usage'] = task_parameters.get('estimated_memory_request', 1024) if task_parameters else 1024 # Placeholder (in MB)

         task_features_df = pd.DataFrame([sample_data])

         # Handle missing values in prediction features using the means from training data
         try:
             # Ensure the columns are in the same order as training features before imputation and prediction
             task_features_df = task_features_df[self._trained_features]

             # Impute numerical features with the mean from training data
             numerical_cols_to_impute = [col for col in self._trained_features if col in self._numerical_features_means]
             for col in numerical_cols_to_impute:
                 if pd.isna(task_features_df.loc[0, col]):
                      task_features_df.loc[0, col] = self._numerical_features_means.get(col, 0.0) # Use mean or 0.0 if mean not found

             # TODO: Impute categorical features if they can be missing at prediction time (e.g., with mode)

         except KeyError as e:
              self._logger.error(f"Prediction features do not match training features after preparation. Missing: {e}. Cannot predict.")
              return None
         except Exception as e:
              self._logger.error(f"Error during prediction feature preparation: {e}. Cannot predict.")
              return None

         return task_features_df


    def predict_duration(self, task, task_parameters=None) -> float:
        """Predicts the execution duration for a given task.

        Args:
            task: The Task object.
            task_parameters: Dictionary of parameters specific to the task.

        Returns:
            Predicted duration in seconds, or a default value if prediction is not possible.
        """
        if not self._is_trained or self._model_pipeline is None or self._trained_features is None:
            self._logger.warning(f"Model not trained/loaded or features not defined for task {task.name}. Returning default prediction (60.0s).")
            return 60.0 # Return a reasonable default

        self._logger.debug(f"Predicting duration for task: {task.name}")

        task_features_df = self._prepare_features_for_prediction(task, task_parameters)
        if task_features_df is None:
             return 60.0 # Return default if feature preparation failed

        try:
            # Predict using the trained pipeline (preprocessing + model prediction)
            prediction = self._model_pipeline.predict(task_features_df)[0] # Predict expects a DataFrame

            self._logger.debug(f"Predicted duration for task {task.name}: {prediction:.2f}s")
            # Ensure prediction is non-negative and at least a small value
            return max(0.1, prediction) # Return at least 0.1 seconds
        except Exception as e:
            self._logger.error(f"Error during duration prediction for task {task.name}: {e}. Returning default.")
            return 60.0 # Return default on prediction error


    def optimize_resources(self, task, task_parameters=None):
        """Suggests resource allocation for a task (e.g., CPU, memory).

        This method is a placeholder. Implementing actual resource optimization
        would typically require a different ML model trained on historical data
        where the target variables are optimal resource configurations (CPU, memory) for tasks.

        For this example, it provides illustrative recommendations based on a simple rule
        related to the predicted duration (using the existing duration prediction model).

        Args:
            task: The Task object.
            task_parameters: Dictionary of parameters specific to the task.

        Returns:
            A dictionary of recommended resources (e.g., {'cpu': 2, 'memory': '4GB'}),
            or an empty dictionary if recommendations are not available.
        """
        self._logger.debug(f"Attempting to optimize resources for task: {task.name}")

        # This example uses the duration prediction model to inform a simple resource rule.
        # A real implementation would use a dedicated resource optimization model.
        predicted_duration = self.predict_duration(task, task_parameters)

        # Simple illustrative rule based on predicted duration
        if predicted_duration > 120:
            recommendations = {'cpu': 4, 'memory': '16GB'}
        elif predicted_duration > 60:
            recommendations = {'cpu': 2, 'memory': '8GB'}
        elif predicted_duration > 30:
            recommendations = {'cpu': 1, 'memory': '4GB'}
        else:
            recommendations = {'cpu': 1, 'memory': '2GB'}

        self._logger.debug(f"Resource optimization for task {task.name}: Predicted Duration {predicted_duration:.2f}s, Recommendations: {recommendations}")
        return recommendations


    def save_model(self, path):
        """Saves the trained model pipeline and trained features/means to disk using joblib."
        if self._model_pipeline is None or not self._is_trained or self._trained_features is None or self._numerical_features_means is None:
            self._logger.warning("Nothing to save: model not trained or components missing.")
            return
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Save the pipeline
            joblib.dump(self._model_pipeline, path)
            # Save the feature list and numerical feature means separately
            joblib.dump(self._trained_features, path.replace('.joblib', '_features.joblib'))
            joblib.dump(self._numerical_features_means, path.replace('.joblib', '_means.joblib'))

            self._logger.info(f"ML model pipeline, features, and means saved to {path} and associated files.")
        except Exception as e:
            self._logger.error(f"Error saving model components to {path}: {e}")

    def load_model(self, path):
        """Loads a trained model pipeline, features, and means from disk using joblib."
        features_path = path.replace('.joblib', '_features.joblib')
        means_path = path.replace('.joblib', '_means.joblib')

        if not os.path.exists(path) or not os.path.exists(features_path) or not os.path.exists(means_path):
             self._logger.warning(f"Model components not found at {path} or associated files. Cannot load model.")
             self._model_pipeline = None
             self._is_trained = False
             self._trained_features = None
             self._numerical_features_means = None
             return
        try:
            self._model_pipeline = joblib.load(path)
            self._trained_features = joblib.load(features_path)
            self._numerical_features_means = joblib.load(means_path)

            self._is_trained = True
            self._logger.info(f"ML model pipeline, features, and means loaded from {path} and associated files.")
        except Exception as e:
            self._logger.error(f"Error loading model components from {path}: {e}")
            self._model_pipeline = None
            self._is_trained = False
            self._trained_features = None
            self._numerical_features_means = None
