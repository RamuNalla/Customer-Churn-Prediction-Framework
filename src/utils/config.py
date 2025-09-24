import os
import yaml
from pathlib import Path
from typing import Dict, Any
import logging
from dotenv import load_dotenv

class Config:               # configuration management class for entire project

    def __init__(self, config_dir: str = "../configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)

        load_dotenv()

        self.model_config = self._load_config("model_config.yaml")
        self.data_config = self._load_config("data_config.yaml")
        self.deployment_config = self._load_config("deployment_config.yaml")

    def _load_config(self, filename: str) -> Dict[str, Any]:            # load configuration files from YAML file

        config_path = self.config_dir / filename
        print(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        else:
            default_config = self._get_default_config(filename)
            self._save_config(config_path, default_config)
            return default_config
        
    def _save_config(self, filename: str, config: Dict[str, Any]):     # save configuration to YAML file
        config_path = self.config_dir / filename
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def _get_default_config(self, filename: str) -> Dict[str, Any]:    # provide default configuration values

        if filename == "model_config.yaml":
            return{
                'random_state': 42,
                'test_size': 0.2,
                'validation_size': 0.2,
                'cv_folds': 5,
                'models': {
                    'logistic_regression': {
                        'C': [0.001, 0.01, 0.1, 1, 10],
                        'penalty': ['l1', 'l2', 'elasticnet'],
                        'solver': ['liblinear', 'saga']
                    },
                    'random_forest': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [5, 10, 15, None],
                        'min_samples_split': [2, 5, 10],
                        'min_samples_leaf': [1, 2, 4]
                    },
                    'xgboost': {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                },
                'ensemble': {
                    'voting_type': 'soft',
                    'stacking_cv': 5
                }
            }
        
        elif filename == "data_config.yaml":
            return {
                'data_paths': {
                    'raw_data': 'data/raw/Telco-Customer-data.csv',
                    'processed_data': 'data/processed/',
                    'features_data': 'data/features/',
                    'external_data': 'data/external/'
                },
                'preprocessing': {
                    'handle_missing': True,
                    'missing_strategy': 'median',
                    'outlier_method': 'iqr',
                    'scaling_method': 'standard',
                    'encoding_method': 'onehot'
                },
                'feature_engineering': {
                    'create_interactions': True,
                    'polynomial_features': False,
                    'polynomial_degree': 2,
                    'feature_selection': True,
                    'selection_method': 'rfe',
                    'max_features': 50
                },
                'target_column': 'Churn',
                'positive_class': 'Yes'
            }
        
        elif filename == "deployment_config.yaml":
            return {
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'workers': 4,
                    'timeout': 30
                },
                'model_serving': {
                    'model_registry_path': 'models/trained_models/',
                    'default_model': 'ensemble_final.pkl',
                    'batch_size': 1000,
                    'prediction_threshold': 0.5
                },
                'monitoring': {
                    'enable_logging': True,
                    'log_predictions': True,
                    'drift_detection': True,
                    'performance_tracking': True
                },
                'database': {
                    'host': os.getenv('DB_HOST', 'localhost'),
                    'port': os.getenv('DB_PORT', 5432),
                    'database': os.getenv('DB_NAME', 'teleretain'),
                    'username': os.getenv('DB_USER', 'postgres'),
                    'password': os.getenv('DB_PASSWORD', 'password')
                }
            }
        
        return {}
    

config = Config()       # instantiate configuration object for use throughout the project

    



