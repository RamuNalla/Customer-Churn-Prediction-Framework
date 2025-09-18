
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from src.utils.logger import setup_logger
from src.utils.config import config

class DataLoader:           # Data loading utility function

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.data_config = config.data_config

    def load_raw_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        
        if filepath is None:
            filepath = self.data_config['data_paths']['raw_data']

        try:
            self.logger.info(f"Loading data from {filepath}")
            df = pd.read_csv(filepath)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def load_processed_data(self, train: bool = True) -> pd.DataFrame:

        processed_dir = Path(self.data_config['data_paths']['processed_data'])
        filename = "train_processed.csv" if train else "test_processed.csv"
        filepath = processed_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Processed data not found: {filepath}")
            
        return pd.read_csv(filepath)
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:

        processed_dir = Path(self.data_config['data_paths']['processed_data'])
        processed_dir.mkdir(parents=True, exist_ok=True)

        filepath = processed_dir / filename

        df.to_csv(filepath, index=False)
        self.logger.info(f"Processed data saved to {filepath}")

    def create_train_test_split(self, df: pd.DataFrame, 
                                target_col: Optional[str] = None, 
                                test_size: float = 0.2, random_state: int = 42, 
                                stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:

        from sklearn.model_selection import train_test_split

        if target_col is None:
            target_col = self.data_config["target_column"]

        if stratify and target_col in df.columns:
            stratify_column = df[target_col]
        else:
            stratify_column =  None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_column
        )

        self.logger.info(f"Data split created - Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df
