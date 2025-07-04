import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
import warnings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_classification_type(y):
    """
    Detect if the problem is binary or multiclass classification.
    
    Args:
        y: Target variable (pandas Series or numpy array)
    
    Returns:
        tuple: (classification_type, n_classes)
    """
    unique_values = pd.Series(y).unique()
    n_classes = len(unique_values)
    
    if n_classes == 2:
        return "binary", n_classes
    else:
        return "multiclass", n_classes

def get_class_distribution(y):
    """
    Get class distribution as a dictionary.
    
    Args:
        y: Target variable (pandas Series or numpy array)
    
    Returns:
        dict: Class distribution
    """
    return pd.Series(y).value_counts().to_dict()

def handle_class_imbalance(X_train, y_train, strategy):
    """
    Handle class imbalance using various strategies.
    
    Args:
        X_train: Training features
        y_train: Training labels
        strategy: Strategy to use ('auto', 'smote', 'class_weights', 'none')
    
    Returns:
        tuple: (X_balanced, y_balanced)
    """
    if strategy == "none":
        return X_train, y_train
    
    # Get class distribution
    class_counts = pd.Series(y_train).value_counts()
    imbalance_ratio = min(class_counts) / max(class_counts)
    
    if strategy == "auto":
        if imbalance_ratio < 0.3:  # Severe imbalance
            strategy = "smote"
        elif imbalance_ratio < 0.7:  # Moderate imbalance
            strategy = "class_weights"
        else:
            return X_train, y_train  # No handling needed
    
    if strategy == "class_weights":
        # Calculate class weights
        class_weights = dict(zip(
            class_counts.index, 
            len(y_train) / (len(class_counts) * class_counts)
        ))
        return X_train, y_train, class_weights
    
    elif strategy == "smote":
        # Simple oversampling for now (can be enhanced with SMOTE later)
        from sklearn.utils import resample
        
        # Find the majority class
        majority_class = class_counts.idxmax()
        minority_classes = class_counts[class_counts < class_counts.max()].index
        
        X_balanced = [X_train]
        y_balanced = [y_train]
        
        for minority_class in minority_classes:
            # Get minority class samples
            minority_mask = y_train == minority_class
            X_minority = X_train[minority_mask]
            y_minority = y_train[minority_mask]
            
            # Oversample to match majority class
            n_samples_needed = class_counts[majority_class] - len(y_minority)
            X_oversampled = resample(
                X_minority, 
                n_samples=n_samples_needed,
                random_state=42
            )
            y_oversampled = pd.Series([minority_class] * n_samples_needed)
            
            X_balanced.append(X_oversampled)
            y_balanced.append(y_oversampled)
        
        return pd.concat(X_balanced), pd.concat(y_balanced)
    
    return X_train, y_train

class ModelFactory:
    """
    Factory class for creating and configuring machine learning models.
    """
    
    def __init__(self, classification_type, n_classes, config):
        self.classification_type = classification_type
        self.n_classes = n_classes
        self.config = config
        self.label_encoder = LabelEncoder()
        self.is_label_encoder_fitted = False
    
    def fit_label_encoder(self, y):
        """Fit the label encoder on the training data."""
        self.label_encoder.fit(y)
        self.is_label_encoder_fitted = True
    
    def transform_labels(self, y):
        """Transform labels using the fitted encoder."""
        if not self.is_label_encoder_fitted:
            raise ValueError("Label encoder must be fitted before transforming labels")
        return self.label_encoder.transform(y)
    
    def inverse_transform_labels(self, y):
        """Inverse transform labels back to original format."""
        if not self.is_label_encoder_fitted:
            raise ValueError("Label encoder must be fitted before inverse transforming labels")
        return self.label_encoder.inverse_transform(y)
    
    def get_model_config(self, model_name):
        """Get model configuration and hyperparameter grid."""
        
        if model_name == "linear":
            return {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "param_grid": {
                    "C": [0.1, 1, 10, 100],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear", "saga"]
                }
            }
        
        elif model_name == "random_forest":
            return {
                "model": RandomForestClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
        
        elif model_name == "xgboost":
            return {
                "model": xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }
            }
        
        elif model_name == "lightgbm":
            return {
                "model": lgb.LGBMClassifier(random_state=42, verbose=-1),
                "param_grid": {
                    "n_estimators": [50, 100],
                    "max_depth": [3, 6, 9],
                    "learning_rate": [0.01, 0.1, 0.3],
                    "num_leaves": [31, 63, 127],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }
            }
        
        elif model_name == "svm":
            return {
                "model": SVC(random_state=42, probability=True),
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto", 0.001, 0.01]
                }
            }
        
        elif model_name == "neural_net":
            return {
                "model": MLPClassifier(random_state=42, max_iter=500),
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 25)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate_init": [0.001, 0.01]
                }
            }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")

def train_model_with_cv(model_name, model_config, X_train, y_train, cv_folds=5, n_jobs=-1):
    """
    Train a model using cross-validation and hyperparameter tuning.
    
    Args:
        model_name: Name of the model
        model_config: Model configuration dictionary
        X_train: Training features
        y_train: Training labels
        cv_folds: Number of CV folds
        n_jobs: Number of parallel jobs
    
    Returns:
        dict: Training results
    """
    try:
        logger.info(f"Training {model_name} with cross-validation...")
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=model_config["model"],
            param_grid=model_config["param_grid"],
            cv=cv_folds,
            scoring='f1_weighted' if model_config["model"].__class__.__name__ in ['XGBClassifier', 'LGBMClassifier'] else 'f1_weighted',
            n_jobs=n_jobs,
            verbose=0
        )
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        logger.info(f"{model_name} best score: {grid_search.best_score_:.3f}")
        
        return {
            "status": "success",
            "model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": grid_search.cv_results_
        }
        
    except Exception as e:
        logger.error(f"Error training {model_name}: {str(e)}")
        return {
            "status": "failed",
            "error": str(e)
        }


def evaluate_model(model, X_train, y_train, X_test, y_test, classification_type):
    """
    Evaluate a trained model on training and test sets.
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        classification_type: Type of classification ('binary' or 'multiclass')
    
    Returns:
        dict: Evaluation metrics
    """
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    if classification_type == "binary":
        train_f1 = f1_score(y_train, y_train_pred, average='binary')
        test_f1 = f1_score(y_test, y_test_pred, average='binary')
        train_precision = precision_score(y_train, y_train_pred, average='binary')
        test_precision = precision_score(y_test, y_test_pred, average='binary')
        train_recall = recall_score(y_train, y_train_pred, average='binary')
        test_recall = recall_score(y_test, y_test_pred, average='binary')
    else:
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
    
    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_f1": train_f1,
        "test_f1": test_f1,
        "train_precision": train_precision,
        "test_precision": test_precision,
        "train_recall": train_recall,
        "test_recall": test_recall,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred
    }