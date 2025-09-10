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

