import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from typing import Dict, List, Tuple, Any
import os
from datetime import datetime
import logging
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import json

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class DataExplorer:                     # comprehensive data exploration class

    def __init__(self, data_path: str, output_dir: str = "reports/"):
        self.data_path = data_path
        self.output_dir = output_dir    
        self.df = None
        self.data_qualitty_report = {}
        self.business_insights = {}

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/figures", exist_ok=True)

        self._setup_logging()               # run this function immediately after an object is created


    def _setup_logging(self):   
        logging.basicConfig(                # main function that configures the root logger
            level=logging.INFO,             # set the logging level to INFO (captures INFO, WARNING, ERROR, CRITICAL)
            format = '%(asctime)s - %(levelname)s - %(message)s',       # log message format (timestamp, level, message)
            handlers=[
                logging.FileHandler(f"{self.output_dir}/data_exploration.log"),     # log messages to a file
                logging.StreamHandler()                                             # also print log messages to the console
            ]
        )
        self.logger = logging.getLogger(__name__)       # it helps include the name of the module where the logger is used during logging


    def load_data(self) -> pd.DataFrame:                # load and perform initial data validation
        try:
            self.logger.info(f"Loading data from {self.data_path}")
            self.df = pd.read_csv(self.data_path)
            
            self.logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
            self.data_quality_report['shape'] = self.df.shape
            self.data_quality_report['columns'] = list(self.df.columns)
            self.data_quality_report['dtypes'] = self.df.dtypes.to_dict()

            return self.df
        
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise

    
    def data_quality_assessment(self) -> Dict[str, Any]:          # assess data quality 
        
        self.logger.info("Starting data quality assessment...  ")

        self.data_quality_report['basic_stats'] = {                 # basic statistics about the dataset
            'total_records': len(self.df),
            'total_features': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2      # Total memory usage by DF in MB
        } 

        missing_data = self.df.isnull().sum()                           # count of missing values per column
        missing_percentage = (missing_data / len(self.df)) * 100        # percentage of missing values per column

        self.data_quality_report['missing_values'] = {
            'columns_with_missing': missing_data[missing_data > 0].to_dict(),
            'missing_percentage': missing_percentage[missing_percentage > 0].to_dict(),
            'total_missing_cells': int(missing_data.sum()),
            'percentage_missing_cells': (missing_data.sum() / (self.df.shape[0] * self.df.shape[1])) * 100
        }

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()

        self.data_quality_report['data_types'] = {
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'numeric_count': len(numeric_cols),
            'categorical_count': len(categorical_cols)
        }

        duplicates = self.df.duplicated().sum()               # count of duplicate rows
        self.data_quality_report['duplicates'] = {
            'duplicate_count': duplicates,
            'duplicate_percentage': (duplicates / len(self.df)) * 100
        }

        unique_values = {}                              # count and percentage of unique values per column  
        for col in self.df.columns:
            unique_values[col] = {
                'unique_count': self.df[col].nunique(),
                'unique_percentage': (self.df[col].nunique() / len(self.df)) * 100
            }
        
        self.data_quality_report['unique_values'] = unique_values

        issues = []

        all_missing = [col for col in self.df.columns if self.df[col].isnull().all()]
        if all_missing:
            issues.append(f"Columns with all missing values: {all_missing}")

        high_cardinality = [col for col in categorical_cols if self.df[col].nunique() > 50]
        if high_cardinality:
            issues.append(f"High cardinality categorical columns: {high_cardinality}")
        
        self.data_quality_report['potential_issues'] = issues
        
        self.logger.info("Data quality assessment completed")
        return self.data_quality_report

    
    def outlier_detection(self) -> Dict[str, Any]:              # detect outliers in numberic columns using multiple methods
        
        self.logger.info("Starting outlier detection...")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()

        if 'customerID' in numeric_cols:                        # Remove ID column if numeric
            numeric_cols.remove('customerID')

        outlier_report = {}

        for col in numeric_cols:
            col_data = self.df[col].dropna()
            
            Q1 = col_data.quantile(0.25)             # IQR Method
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            iqr_outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()

            z_scores = np.abs(stats.zscore(col_data))       # Z-Score Method
            zscore_outliers = (z_scores > 3).sum()

            median = np.median(col_data)                    # Modified Z-Score Method
            mad = np.median(np.abs(col_data - median))
            modified_z_scores = 0.6745 * (col_data - median) / mad
            modified_zscore_outliers = (np.abs(modified_z_scores) > 3.5).sum()

            outlier_report[col] = {
                'iqr_outliers': iqr_outliers,
                'iqr_percentage': (iqr_outliers / len(col_data)) * 100,
                'zscore_outliers': zscore_outliers,
                'zscore_percentage': (zscore_outliers / len(col_data)) * 100,
                'modified_zscore_outliers': modified_zscore_outliers,
                'modified_zscore_percentage': (modified_zscore_outliers / len(col_data)) * 100,
                'bounds': {
                    'iqr_lower': lower_bound,
                    'iqr_upper': upper_bound
                }
            }

        self.data_quality_report['outliers'] = outlier_report
        self.logger.info("Outlier detection completed")
        return outlier_report
    

    def target_variable_analysis(self, target_col: str = 'Churn') -> Dict[str, Any]:        # analyze the target variable distribution and its characteristics

        self.logger.info(f"Analyzing target variable: {target_col}")

        if target_col not in self.df.columns:
            self.logger.error(f"Target column '{target_col}' not found in dataset")
            return {}
        
        target_analysis = {}

        value_counts = self.df[target_col].value_counts()                       # basic distribution
        percentage = self.df[target_col].value_counts(normalize=True) * 100 

        target_analysis['distribution'] = {
            'value_counts': value_counts.to_dict(),
            'percentages': percentage.to_dict(),
            'total_samples': len(self.df[target_col])
        }

        if len(value_counts) == 2:              # class imbalance analysis for Binary classification
            minority_class = value_counts.min()
            majority_class = value_counts.max()
            imbalance_ratio = majority_class / minority_class
            
            target_analysis['class_balance'] = {
                'minority_class_count': minority_class,
                'majority_class_count': majority_class,
                'imbalance_ratio': imbalance_ratio,
                'is_imbalanced': imbalance_ratio > 1.5
            }
        
        missing_target = self.df[target_col].isnull().sum()         # missing values in target column
        target_analysis['missing_values'] = {
            'count': missing_target,
            'percentage': (missing_target / len(self.df)) * 100
        }

        self.data_quality_report['target_analysis'] = target_analysis
        self.logger.info("Target variable analysis completed")
        return target_analysis
        