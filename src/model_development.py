"""
This module implements comprehensive model development pipeline following 
industry best practices for customer churn prediction.

Key Features:
- Multiple algorithm implementations
- Advanced ensemble methods
- Hyperparameter optimization
- Cross-validation strategies
- Model interpretation and explainability
- Business impact analysis

Author: Ramu Nalla
Date: 27-09-2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier,
    VotingClassifier, StackingClassifier, BaggingClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, log_loss
)
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import logging
from datetime import datetime
from pathlib import Path
import joblib
import json
import optuna
from scipy import stats
import shap
import lime
import lime.lime_tabular

# Configure warnings and plotting
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedModelDeveloper:       # Advanced Model Development Pipeline
    
    """
    This class implements comprehensive model development including:
        - Multiple algorithm implementations
        - Advanced ensemble methods  
        - Hyperparameter optimization
        - Model interpretation and explainability
        - Business impact analysis
    """

    def __init__(self, output_dir: str = "models/", 
                 config_path: str = "configs/model_config.yaml",
                 random_state: int = 42):
        """
        Initialize Advanced Model Developer
        
        Args:
            output_dir (str): Directory to save models and results
            config_path (str): Path to configuration file
            random_state (int): Random state for reproducibility
        """
       
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Always resolve config_path relative to the parent directory of this file unless absolute
        if not Path(config_path).is_absolute():
            # Get parent directory of this file (src/)
            parent_dir = Path(__file__).parent.parent
            self.config_path = str(parent_dir / "configs" / config_path)
        else:
            self.config_path = config_path

        print(self.config_path)
        print(Path(self.output_dir))        # This is the folder in the directory from where you are executing this script

        self.random_state = random_state
        self.models = {}
        self.model_results = {}
        self.ensemble_models = {}
        self.best_model = None
        self.feature_names = None
        self.target_name = None
        
        self._setup_logging()           # Setup logging
        
        self._load_config()             # Load configuration


    def _setup_logging(self):               # Setup logging configuration
        
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/phase3_model_development_{datetime.now().strftime("%Y%m%d")}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self):                 # Load configuration from YAML file
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                self.logger.info(f"Configuration loaded from {self.config_path}")
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
            self.config = self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:        # Get default configuration
        
        return {
            'random_state': 42,
            'test_size': 0.2,
            'validation_size': 0.2,
            'cv_folds': 5,
            'optimization': {
                'hyperparameter_tuning': 'optuna',
                'n_trials': 50,
                'cv_scoring': 'roc_auc',
                'n_jobs': -1
            },
            'evaluation': {
                'primary_metric': 'roc_auc',
                'secondary_metrics': ['precision', 'recall', 'f1', 'accuracy'],
                'business_metrics': True,
                'interpretability': True
            },
            'models': {
                'logistic_regression': True,
                'random_forest': True,
                'xgboost': True,
                'lightgbm': True,
                'catboost': True,
                'svm': False,  # Can be slow on large datasets
                'naive_bayes': True,
                'knn': False    # Can be slow on large datasets
            },
            'ensemble': {
                'voting_classifier': True,
                'stacking_classifier': True,
                'bagging_classifier': True
            }
        }

    def load_data(self, data_path: str, target_col: str = 'Churn') -> Tuple[pd.DataFrame, pd.Series]:   # Load preprocessed data
        
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            df = pd.read_csv(data_path)
            
            if target_col in df.columns:            # Separate features and target
                X = df.drop(columns=[target_col])
                y = df[target_col]
                
                # Convert target to binary if needed
                if y.dtype == 'object':
                    y = (y == 'Yes').astype(int)
                elif set(y.unique()) == {0, 1}:
                    y = y.astype(int)
                else:
                    raise ValueError(f"Target column {target_col} has unexpected values: {y.unique()}")
            else:
                raise ValueError(f"Target column {target_col} not found in data")
            
            self.feature_names = list(X.columns)
            self.target_name = target_col
            
            self.logger.info(f"Data loaded successfully. Shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
            return X, y
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    
    def create_train_validation_test_split(self, X: pd.DataFrame, y: pd.Series) -> Tuple:   # Create train/validation/test splits
        
        self.logger.info("Creating train/validation/test splits...")
        
        test_size = self.config['test_size']
        val_size = self.config['validation_size']

        X_temp, X_test, y_temp, y_test = train_test_split(      # First split: separate test set
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate train and validation from remaining data
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.random_state, stratify=y_temp
        )
        
        self.logger.info(f"Data splits created:")
        self.logger.info(f"  Train: {X_train.shape[0]} samples ({y_train.sum()} positive)")
        self.logger.info(f"  Validation: {X_val.shape[0]} samples ({y_val.sum()} positive)")
        self.logger.info(f"  Test: {X_test.shape[0]} samples ({y_test.sum()} positive)")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def initialize_base_models(self) -> Dict[str, Any]:     # Initialize base machine learning models
        
        self.logger.info("Initializing base models...")
        
        models = {}
        
        # Logistic Regression
        if self.config['models']['logistic_regression']:
            models['logistic_regression'] = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                n_jobs=self.config['optimization'].get('n_jobs', -1)
            )
        
        # Random Forest
        if self.config['models']['random_forest']:
            models['random_forest'] = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=self.config['optimization'].get('n_jobs', -1)
            )
        
        # XGBoost
        if self.config['models']['xgboost']:
            models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=self.config['optimization'].get('n_jobs', -1),
                eval_metric='logloss'
            )
        
        # LightGBM
        if self.config['models']['lightgbm']:
            models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_jobs=self.config['optimization'].get('n_jobs', -1),
                verbose=-1
            )
        
        # CatBoost
        if self.config['models']['catboost']:
            models['catboost'] = CatBoostClassifier(
                random_state=self.random_state,
                verbose=False
            )
        
        # Support Vector Machine
        if self.config['models']['svm']:
            models['svm'] = SVC(
                random_state=self.random_state,
                probability=True  # Enable probability prediction
            )
        
        # Naive Bayes
        if self.config['models']['naive_bayes']:
            models['naive_bayes'] = GaussianNB()
        
        # K-Nearest Neighbors
        if self.config['models']['knn']:
            models['knn'] = KNeighborsClassifier(
                n_jobs=self.config['optimization'].get('n_jobs', -1)
            )
        
        # Extra Trees
        models['extra_trees'] = ExtraTreesClassifier(
            random_state=self.random_state,
            n_jobs=self.config['optimization'].get('n_jobs', -1)
        )
        
        # Decision Tree (for ensemble diversity)
        models['decision_tree'] = DecisionTreeClassifier(
            random_state=self.random_state
        )
        
        self.logger.info(f"Initialized {len(models)} base models: {list(models.keys())}")
        return models
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict]:      # Train baseline models with default parameters
        
        self.logger.info("Training baseline models...")
        
        models = self.initialize_base_models()
        results = {}
        
        for name, model in models.items():
            self.logger.info(f"Training baseline {name}...")
            
            try:
                # Train model
                start_time = datetime.now()
                
                # Handle models that support early stopping
                if name in ['xgboost', 'lightgbm', 'catboost']:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    model.fit(X_train, y_train)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_pred = model.predict(X_val)
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics['training_time'] = training_time
                
                results[name] = {
                    'model': model,
                    'metrics': metrics,
                    'predictions': {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                self.logger.info(f"  {name}: AUC = {metrics['roc_auc']:.4f}, "
                               f"Precision = {metrics['precision']:.4f}, "
                               f"Recall = {metrics['recall']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                continue
        
        self.models.update({f"{name}_baseline": result['model'] for name, result in results.items()})
        self.model_results.update({f"{name}_baseline": result for name, result in results.items()})
        
        self.logger.info(f"Baseline training completed for {len(results)} models")
        return results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                          y_pred_proba: np.ndarray) -> Dict[str, float]:    # Calculate comprehensive evaluation metrics
        
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_true, y_pred_proba),
                'log_loss': log_loss(y_true, y_pred_proba)
            }
            
            # Calculate additional metrics
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            metrics.update({
                'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'balanced_accuracy': (metrics['recall'] + metrics['specificity']) / 2,
                'true_positives': int(tp),
                'true_negatives': int(tn),
                'false_positives': int(fp),
                'false_negatives': int(fn)
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            return {}


    def hyperparameter_optimization(self, X_train: pd.DataFrame, y_train: pd.Series,
                                  model_name: str, n_trials: int = 50) -> Dict[str, Any]:       # Perform hyperparameter optimization using Optuna
        
        self.logger.info(f"Starting hyperparameter optimization for {model_name} ({n_trials} trials)...")
        
        def objective(trial):
            # Define hyperparameter spaces for different models
            if model_name == 'random_forest':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_name == 'xgboost':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'eval_metric': 'logloss'
                }
                model = xgb.XGBClassifier(**params)
                
            elif model_name == 'lightgbm':
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
                    'random_state': self.random_state,
                    'n_jobs': -1,
                    'verbose': -1
                }
                model = lgb.LGBMClassifier(**params)
                
            elif model_name == 'catboost':
                params = {
                    'iterations': trial.suggest_int('iterations', 100, 500),
                    'depth': trial.suggest_int('depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
                    'random_state': self.random_state,
                    'verbose': False
                }
                model = CatBoostClassifier(**params)
                
            elif model_name == 'logistic_regression':
                params = {
                    'C': trial.suggest_float('C', 0.001, 100, log=True),
                    'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
                    'solver': trial.suggest_categorical('solver', ['liblinear', 'saga']),
                    'max_iter': 1000,
                    'random_state': self.random_state,
                    'n_jobs': -1
                }
                
                # Handle penalty-solver compatibility
                if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
                    params['solver'] = 'saga'
                if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                    params['solver'] = 'saga'
                    
                model = LogisticRegression(**params)
                
            else:
                raise ValueError(f"Hyperparameter optimization not implemented for {model_name}")
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
                scoring=self.config['optimization']['cv_scoring'],
                n_jobs=-1
            )
            
            return cv_scores.mean()
        
        # Run optimization
        study = optuna.create_study(direction='maximize', 
                                  study_name=f"{model_name}_optimization")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Get best parameters and train final model
        best_params = study.best_params
        
        # Create and train optimized model
        if model_name == 'random_forest':
            optimized_model = RandomForestClassifier(**best_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'xgboost':
            optimized_model = xgb.XGBClassifier(**best_params, random_state=self.random_state, n_jobs=-1, eval_metric='logloss')
        elif model_name == 'lightgbm':
            optimized_model = lgb.LGBMClassifier(**best_params, random_state=self.random_state, n_jobs=-1, verbose=-1)
        elif model_name == 'catboost':
            optimized_model = CatBoostClassifier(**best_params, random_state=self.random_state, verbose=False)
        elif model_name == 'logistic_regression':
            optimized_model = LogisticRegression(**best_params, random_state=self.random_state, n_jobs=-1, max_iter=1000)
        
        self.logger.info(f"Hyperparameter optimization completed for {model_name}")
        self.logger.info(f"Best score: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {best_params}")
        
        return {
            'model': optimized_model,
            'best_params': best_params,
            'best_score': study.best_value,
            'study': study
        }
    
    def train_optimized_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series,
                             models_to_optimize: List[str] = None) -> Dict[str, Dict]:  # Train models with optimized hyperparameters
               
        if models_to_optimize is None:
            models_to_optimize = ['random_forest', 'xgboost', 'lightgbm']
        
        self.logger.info(f"Training optimized models: {models_to_optimize}")
        
        results = {}
        n_trials = self.config['optimization'].get('n_trials', 50)
        
        for model_name in models_to_optimize:
            self.logger.info(f"Optimizing {model_name}...")
            
            try:
                # Hyperparameter optimization
                optimization_result = self.hyperparameter_optimization(
                    X_train, y_train, model_name, n_trials
                )
                
                # Train optimized model
                optimized_model = optimization_result['model']
                
                start_time = datetime.now()
                
                # Handle models with early stopping
                if model_name in ['xgboost', 'lightgbm', 'catboost']:
                    optimized_model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=50,
                        verbose=False
                    )
                else:
                    optimized_model.fit(X_train, y_train)
                
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                y_pred = optimized_model.predict(X_val)
                y_pred_proba = optimized_model.predict_proba(X_val)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics['training_time'] = training_time
                
                results[model_name] = {
                    'model': optimized_model,
                    'metrics': metrics,
                    'best_params': optimization_result['best_params'],
                    'optimization_score': optimization_result['best_score'],
                    'predictions': {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                self.logger.info(f"  {model_name} optimized: AUC = {metrics['roc_auc']:.4f}, "
                               f"Precision = {metrics['precision']:.4f}, "
                               f"Recall = {metrics['recall']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error optimizing {model_name}: {str(e)}")
                continue
        
        # Update model storage
        self.models.update({f"{name}_optimized": result['model'] for name, result in results.items()})
        self.model_results.update({f"{name}_optimized": result for name, result in results.items()})
        
        self.logger.info(f"Optimized training completed for {len(results)} models")
        return results
    

    def create_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Dict]:     # Create and train ensemble models
        
        self.logger.info("Creating ensemble models...")
        
        # Get best performing base models for ensembles
        base_models = []
        model_names = []
        
        # Select top performing models from optimized models
        optimized_results = {k: v for k, v in self.model_results.items() if 'optimized' in k}
        
        if len(optimized_results) < 2:
            self.logger.warning("Not enough optimized models for ensemble creation")
            return {}
        
        # Sort by AUC score and take top models
        sorted_models = sorted(optimized_results.items(), 
                             key=lambda x: x[1]['metrics']['roc_auc'], 
                             reverse=True)
        
        # Take top 3-5 models for ensemble
        top_models = sorted_models[:min(5, len(sorted_models))]
        
        for name, result in top_models:
            base_models.append((name.replace('_optimized', ''), result['model']))
            model_names.append(name)
        
        self.logger.info(f"Using {len(base_models)} models for ensemble: {[name for name, _ in base_models]}")
        
        ensemble_results = {}
        
        # 1. Voting Classifier
        if self.config['ensemble']['voting_classifier']:
            self.logger.info("Training Voting Classifier...")
            
            try:
                voting_clf = VotingClassifier(
                    estimators=base_models,
                    voting='soft'  # Use predicted probabilities
                )
                
                start_time = datetime.now()
                voting_clf.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Predictions
                y_pred = voting_clf.predict(X_val)
                y_pred_proba = voting_clf.predict_proba(X_val)[:, 1]
                
                # Metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics['training_time'] = training_time
                
                ensemble_results['voting_classifier'] = {
                    'model': voting_clf,
                    'metrics': metrics,
                    'base_models': model_names,
                    'predictions': {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                self.logger.info(f"  Voting Classifier: AUC = {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training Voting Classifier: {str(e)}")
        
        # 2. Stacking Classifier
        if self.config['ensemble']['stacking_classifier']:
            self.logger.info("Training Stacking Classifier...")
            
            try:
                # Use Logistic Regression as meta-learner (base models predictions are used as input features for the meta-learner)
                meta_learner = LogisticRegression(random_state=self.random_state, max_iter=1000)
                
                stacking_clf = StackingClassifier(
                    estimators=base_models,
                    final_estimator=meta_learner,
                    cv=5,  # Cross-validation for meta-features
                    n_jobs=-1
                )
                
                start_time = datetime.now()
                stacking_clf.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Predictions
                y_pred = stacking_clf.predict(X_val)
                y_pred_proba = stacking_clf.predict_proba(X_val)[:, 1]
                
                # Metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics['training_time'] = training_time
                
                ensemble_results['stacking_classifier'] = {
                    'model': stacking_clf,
                    'metrics': metrics,
                    'base_models': model_names,
                    'meta_learner': 'LogisticRegression',
                    'predictions': {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                self.logger.info(f"  Stacking Classifier: AUC = {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training Stacking Classifier: {str(e)}")
        
        # 3. Bagging with best model
        if self.config['ensemble']['bagging_classifier'] and len(base_models) > 0:
            self.logger.info("Training Bagging Classifier...")
            
            try:
                # Use the best individual model as base estimator
                best_model = base_models[0][1]  # First model is best (sorted by AUC)
                
                bagging_clf = BaggingClassifier(        # Trains multiple instances of a single base model
                    base_estimator=best_model,
                    n_estimators=10,
                    random_state=self.random_state,
                    n_jobs=-1
                )
                
                start_time = datetime.now()
                bagging_clf.fit(X_train, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Predictions
                y_pred = bagging_clf.predict(X_val)
                y_pred_proba = bagging_clf.predict_proba(X_val)[:, 1]
                
                # Metrics
                metrics = self._calculate_metrics(y_val, y_pred, y_pred_proba)
                metrics['training_time'] = training_time
                
                ensemble_results['bagging_classifier'] = {
                    'model': bagging_clf,
                    'metrics': metrics,
                    'base_estimator': base_models[0][0],
                    'predictions': {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                self.logger.info(f"  Bagging Classifier: AUC = {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training Bagging Classifier: {str(e)}")
        
        # Update ensemble model storage
        self.ensemble_models.update(ensemble_results)
        self.model_results.update({f"ensemble_{name}": result for name, result in ensemble_results.items()})
        
        self.logger.info(f"Ensemble training completed for {len(ensemble_results)} ensembles")
        return ensemble_results


    def evaluate_all_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict]:      # Evaluate all trained models on test set
        
        self.logger.info("Evaluating all models on test set...")
        
        test_results = {}
        all_models = {**self.model_results, **{f"ensemble_{k}": v for k, v in self.ensemble_models.items()}}
        
        for model_name, model_data in all_models.items():
            self.logger.info(f"Evaluating {model_name}...")
            
            try:
                model = model_data['model']
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate comprehensive metrics
                test_metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                # Calculate business metrics if enabled
                if self.config['evaluation']['business_metrics']:
                    business_metrics = self._calculate_business_metrics(y_test, y_pred, y_pred_proba)
                    test_metrics.update(business_metrics)
                
                test_results[model_name] = {
                    'metrics': test_metrics,
                    'predictions': {
                        'y_pred': y_pred,
                        'y_pred_proba': y_pred_proba
                    }
                }
                
                self.logger.info(f"  {model_name}: Test AUC = {test_metrics['roc_auc']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        # Find best model
        if test_results:
            primary_metric = self.config['evaluation']['primary_metric']
            best_model_name = max(test_results.keys(), 
                                key=lambda k: test_results[k]['metrics'].get(primary_metric, 0))
            self.best_model = {
                'name': best_model_name,
                'model': all_models[best_model_name]['model'],
                'metrics': test_results[best_model_name]['metrics']
            }
            
            self.logger.info(f"Best model: {best_model_name} ({primary_metric} = {self.best_model['metrics'][primary_metric]:.4f})")
        
        return test_results
    

    def _calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_pred_proba: np.ndarray) -> Dict[str, float]:        # Calculate business-specific metrics
        
        try:
            # Confusion matrix components
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            # Business metrics
            business_metrics = {
                'customers_at_risk_identified': int(tp),
                'customers_missed': int(fn),
                'false_alarms': int(fp),
                'true_negatives': int(tn),
                'churn_detection_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
                'precision_in_targeting': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'cost_savings_ratio': (tp - fp) / (tp + fn) if (tp + fn) > 0 else 0
            }
            
            # Calculate lift and gain metrics
            # Sort by predicted probability (descending)
            sorted_indices = np.argsort(y_pred_proba)[::-1]
            sorted_actual = y_true[sorted_indices]
            
            # Calculate cumulative gains
            total_positives = np.sum(y_true)
            decile_size = len(y_true) // 10
            
            gains = []
            for i in range(1, 11):
                end_idx = min(i * decile_size, len(y_true))
                positives_in_decile = np.sum(sorted_actual[:end_idx])
                gain = positives_in_decile / total_positives if total_positives > 0 else 0
                gains.append(gain)
            
            business_metrics.update({
                'top_decile_lift': gains[0] * 10 if gains[0] > 0 else 0,  # Lift in top 10%
                'top_quintile_gain': gains[1] if len(gains) > 1 else 0,    # Gain in top 20%
                'gini_coefficient': 2 * roc_auc_score(y_true, y_pred_proba) - 1
            })
            
            return business_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating business metrics: {str(e)}")
            return {}


    def cross_validate_models(self, X: pd.DataFrame, y: pd.Series, 
                            models_to_validate: List[str] = None) -> Dict[str, Dict]:       # Perform cross-validation on selected models
        
        self.logger.info("Performing cross-validation...")
        
        if models_to_validate is None:
            # Use optimized models and best ensemble
            models_to_validate = [name for name in self.model_results.keys() if 'optimized' in name]
            if 'ensemble_stacking_classifier' in self.model_results:
                models_to_validate.append('ensemble_stacking_classifier')
        
        cv_results = {}
        cv_folds = self.config['cv_folds']
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        for model_name in models_to_validate:
            if model_name not in self.model_results:
                self.logger.warning(f"Model {model_name} not found, skipping...")
                continue
                
            self.logger.info(f"Cross-validating {model_name}...")
            
            try:
                model = self.model_results[model_name]['model']
                
                # Cross-validation for multiple metrics
                scoring_metrics = [
                    self.config['evaluation']['primary_metric'],
                    'precision', 'recall', 'f1'
                ]
                
                cv_scores = {}
                for metric in scoring_metrics:
                    scores = cross_val_score(model, X, y, cv=cv, scoring=metric, n_jobs=-1)
                    cv_scores[metric] = {
                        'mean': scores.mean(),
                        'std': scores.std(),
                        'scores': scores.tolist()
                    }
                
                cv_results[model_name] = cv_scores
                
                primary_metric = self.config['evaluation']['primary_metric']
                self.logger.info(f"  {model_name}: {primary_metric} = "
                               f"{cv_scores[primary_metric]['mean']:.4f} (+/- {cv_scores[primary_metric]['std']*2:.4f})")
                
            except Exception as e:
                self.logger.error(f"Error in cross-validation for {model_name}: {str(e)}")
                continue
        
        return cv_results


    def model_interpretation(self, model_name: str, X_sample: pd.DataFrame, 
                           sample_size: int = 100) -> Dict[str, Any]:       # Generate model interpretation using SHAP and LIME
        """Generate model interpretation using SHAP and LIME"""
        if not self.config['evaluation']['interpretability']:
            return {}
            
        self.logger.info(f"Generating model interpretation for {model_name}...")
        
        if model_name not in self.model_results:
            self.logger.error(f"Model {model_name} not found")
            return {}
        
        model = self.model_results[model_name]['model']
        interpretation_results = {}
        
        # Limit sample size for performance
        if len(X_sample) > sample_size:
            sample_indices = np.random.choice(len(X_sample), sample_size, replace=False)
            X_interpret = X_sample.iloc[sample_indices]
        else:
            X_interpret = X_sample
        
        try:
            # SHAP Analysis
            self.logger.info("Computing SHAP values...")
            
            # Choose appropriate explainer based on model type
            if hasattr(model, 'predict_proba'):
                if 'tree' in model_name.lower() or 'forest' in model_name.lower() or 'xgb' in model_name.lower():
                    explainer = shap.TreeExplainer(model)
                else:
                    # Use a smaller background dataset for other models
                    background_sample = X_interpret.sample(min(50, len(X_interpret)), random_state=self.random_state)
                    explainer = shap.KernelExplainer(model.predict_proba, background_sample)
            else:
                explainer = shap.KernelExplainer(model.predict, X_interpret.sample(min(50, len(X_interpret))))
            
            shap_values = explainer.shap_values(X_interpret)
            
            # For binary classification, use positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Positive class
            
            # Calculate feature importance from SHAP values
            feature_importance = np.abs(shap_values).mean(axis=0)
            importance_dict = dict(zip(self.feature_names, feature_importance))
            
            interpretation_results['shap'] = {
                'feature_importance': importance_dict,
                'shap_values': shap_values.tolist() if isinstance(shap_values, np.ndarray) else shap_values,
                'expected_value': getattr(explainer, 'expected_value', None)
            }
            
        except Exception as e:
            self.logger.error(f"Error in SHAP analysis: {str(e)}")
        
        try:
            # LIME Analysis (for a few sample predictions)
            self.logger.info("Computing LIME explanations...")
            
            lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_interpret.values,
                feature_names=self.feature_names,
                class_names=['No Churn', 'Churn'],
                mode='classification',
                random_state=self.random_state
            )
            
            # Explain a few sample instances
            lime_explanations = []
            sample_instances = min(5, len(X_interpret))
            
            for i in range(sample_instances):
                exp = lime_explainer.explain_instance(
                    X_interpret.iloc[i].values,
                    model.predict_proba,
                    num_features=10
                )
                
                lime_explanations.append({
                    'instance_id': i,
                    'explanation': exp.as_list(),
                    'prediction_proba': model.predict_proba(X_interpret.iloc[i:i+1])[0].tolist()
                })
            
            interpretation_results['lime'] = {
                'explanations': lime_explanations,
                'n_features': 10
            }
            
        except Exception as e:
            self.logger.error(f"Error in LIME analysis: {str(e)}")
        
        # Traditional feature importance (if available)
        try:
            if hasattr(model, 'feature_importances_'):
                traditional_importance = dict(zip(self.feature_names, model.feature_importances_))
                interpretation_results['traditional'] = {
                    'feature_importance': traditional_importance
                }
            elif hasattr(model, 'coef_'):
                coef_importance = dict(zip(self.feature_names, np.abs(model.coef_[0])))
                interpretation_results['traditional'] = {
                    'feature_importance': coef_importance
                }
        except Exception as e:
            self.logger.error(f"Error extracting traditional feature importance: {str(e)}")
        
        return interpretation_results

sample = AdvancedModelDeveloper()
