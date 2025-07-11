import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
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
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, VarianceThreshold



def feature_selection(X, y, method, scoring, n_features):
    if method == 'select_k_best':
        if scoring == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif scoring == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            raise ValueError(f"Unknown scoring function for select_k_best: {scoring}")
        selector.fit(X, y)
        scores = selector.scores_
        feature_scores = pd.Series(scores, index=X.columns)
        selected_features = X.columns[selector.get_support()]
        return feature_scores, selected_features, scoring

    elif method == 'model_importance':
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        importances = rf.feature_importances_
        feature_scores = pd.Series(importances, index=X.columns)
        selected_features = feature_scores.nlargest(n_features).index
        return feature_scores, selected_features, 'Random Forest Importance'

    elif method == 'variance':
        selector = VarianceThreshold()
        selector.fit(X)
        variances = selector.variances_
        feature_scores = pd.Series(variances, index=X.columns)
        selected_features = feature_scores.nlargest(n_features).index
        return feature_scores, selected_features, 'Variance'

    else:
        raise ValueError(f"Unknown feature selection method: {method}")


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
        return X_train, y_train
    
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
                },
                "fixed_params": {}
            }
        
        elif model_name == "random_forest":
            return {
                "model": RandomForestClassifier(random_state=42),
                "param_grid": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },
                "fixed_params": {}
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
                },
                "fixed_params": {}
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
                },
                "fixed_params": {}
            }
        
        elif model_name == "svm":
            return {
                "model": SVC(random_state=42, probability=True),
                "param_grid": {
                    "C": [0.1, 1, 10],
                    "kernel": ["rbf", "linear"],
                    "gamma": ["scale", "auto", 0.001, 0.01]
                },
                "fixed_params": {}
            }
        
        elif model_name == "neural_net":
            return {
                "model": MLPClassifier(random_state=42, max_iter=500),
                "param_grid": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 25)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate_init": [0.001, 0.01]
                },
                "fixed_params": {}
            }
        
        else:
            raise ValueError(f"Unknown model: {model_name}")

def nested_cv_train_model(
    model_name,
    model_config,
    X,
    y,
    method,
    scoring,
    n_features,
    outer_folds=5,
    inner_folds=3,
    n_jobs=-1
):
    """
    Train a model using nested cross-validation with feature selection in the outer loop
    and hyperparameter tuning in the inner loop.

    Returns:
        dict with model, best score, selected features, and evaluation details.
    """
    try:
        print(f"Training {model_name} using nested cross-validation...")

        outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=42)
        outer_scores = []
        outer_best_params = []
        outer_selected_features = []
        outer_models = []
        outer_cv_results = []
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
            print(f"Outer Fold {fold_idx}/{outer_folds}")

            X_train_outer, X_val_outer = X.iloc[train_idx], X.iloc[val_idx]
            y_train_outer, y_val_outer = y[train_idx], y[val_idx]

            # --- Feature Selection in the outer training fold ---
            feature_scores, selected_features, importance_type = feature_selection(
                X_train_outer, y_train_outer, method, scoring, n_features
            )

            X_train_fs = X_train_outer[selected_features]
            X_val_fs = X_val_outer[selected_features]

            # --- Grid SearchCV on the selected features (inner CV loop) ---
            inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=clone(model_config["model"]),
                param_grid=model_config["param_grid"],
                cv=inner_cv,
                scoring='f1_weighted',
                n_jobs=n_jobs,
                verbose=0
            )
            grid_search.fit(X_train_fs, y_train_outer)

            # --- Evaluate on outer validation fold ---
            best_model = grid_search.best_estimator_
            y_val_pred = best_model.predict(X_val_fs)
            score = f1_score(y_val_outer, y_val_pred, average='weighted')

            print(f"  Outer Fold {fold_idx} Score: {score:.4f}")

            outer_scores.append(score)
            outer_best_params.append(grid_search.best_params_)
            outer_selected_features.append(selected_features)
            outer_models.append(best_model)
            outer_cv_results.append(grid_search.cv_results_)

        # --- Final Model Training on Full Dataset ---
        best_fold_idx = int(np.argmax(outer_scores))
        best_params = outer_best_params[best_fold_idx]
        best_features = outer_selected_features[best_fold_idx]
        best_cv_results = outer_cv_results[best_fold_idx]

        print(f"Best outer fold: {best_fold_idx+1} with score {outer_scores[best_fold_idx]:.4f}")

        # Global feature selection for final model
        global_feature_scores, global_features, global_importance_type = feature_selection(
            X, y, method, scoring, n_features
        )
        X_final = X[global_features]

        final_model = clone(model_config["model"])
        final_model.set_params(**best_params)
        final_model.fit(X_final, y)

        return {
            "status": "success",
            "model": final_model,
            "best_score": outer_scores[best_fold_idx],
            "all_scores": outer_scores,
            "selected_features": global_features,
            "feature_scores": global_feature_scores,
            "importance_type": global_importance_type,
            "best_params": best_params,
            "cv_results": best_cv_results,
            'label_encoder': le
        }

    except Exception as e:
        print(f"Error in nested CV for {model_name}: {str(e)}")
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