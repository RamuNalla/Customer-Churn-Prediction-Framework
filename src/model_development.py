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
        


sample = AdvancedModelDeveloper()
