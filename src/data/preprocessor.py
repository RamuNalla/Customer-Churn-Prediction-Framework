import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Dict, List, Any, Optional, Tuple
from src.utils.logger import setup_logger
from src.utils.config import config

class DataPreprocessor:         # data proprocessing utilities

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.data_config = config.data_config
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:          # handle missing values

        self.logger.info("Handling missing values...")

        df_processed = df.copy()

        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns

        if len(numeric_cols) > 0:                                               # handle numerical missing values
            strategy = self.data_config['preprocessing']['missing_strategy']
            imputer = SimpleImputer(strategy=strategy)
            df_processed[numeric_cols] = imputer.fit_transform(df_processed[numeric_cols])
            self.imputers['numeric'] = imputer

        if len(categorical_cols) > 0:                                           # handle categorical missing values
            imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = imputer.fit_transform(df_processed[categorical_cols])
            self.imputers['categorical'] = imputer

        self.logger.info("Missing values handled successfully")
        return df_processed
    
    def handle_outliers(self, df: pd.DataFrame, method: str = None) -> pd.DataFrame:        # handle outliers in numerical columns

        if method is None:
            method = self.data_config['preprocessing']['outlier_method']

        self.logger.info(f"Handling outliers using {method} method...") 

        df_processed = df.copy()
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if method == 'iqr':
                Q1 = df_processed[col].quantile(0.25)
                Q3 = df_processed[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers (on both sides)
                df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)        # any value lower than lower bound will be replaced by lower bound
                
            elif method == 'zscore':
                from scipy import stats
                z_scores = np.abs(stats.zscore(df_processed[col]))
                df_processed = df_processed[z_scores < 3]                   # remove the outlier rows from the Dataframe
                
        self.logger.info("Outliers handled successfully")
        return df_processed
    
    def encode_categorical_variables(self, df: pd.DataFrame, target_col: str = None) -> pd.DataFrame:   # encoding categorical variables

        encoding_method = self.data_config['preprocessing']['encoding_method']
        self.logger.info(f"Encoding categorical variables using {encoding_method}...")

        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()

        if target_col and target_col in categorical_cols:
            categorical_cols.remove(target_col)

        if encoding_method == 'onehot':
            df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)







