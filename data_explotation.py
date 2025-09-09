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


