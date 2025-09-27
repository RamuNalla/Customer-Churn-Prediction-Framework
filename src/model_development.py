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
        


sample = AdvancedModelDeveloper()
