import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.utils.logger import setup_logger

class DataValidator:

    def __init__(self):
        self.logger = setup_logger(__name__)
        self.validation_report = {}

    def validate_schema(self, df: pd.DataFrame, expected_columns: List[str]) -> bool:       # validate dataframe schema

        missing_columns = set(expected_columns) - set(df.columns)
        extra_columns = set(df.columns) - set(expected_columns)

        self.validation_report["schema"] = {
            'missing_columns': list(missing_columns),
            'extra_columns': list(extra_columns),
            'is_valid': len(missing_columns) == 0
        }

        if missing_columns:
            self.logger.warning(f"Missing columns: {missing_columns}")
        if extra_columns:
            self.logger.info(f"Extra columns found: {extra_columns}")
            
        return len(missing_columns) == 0
    
    def validate_data_quality(self, df: pd.DataFrame,                       # validate overall data quiality
                            max_missing_percentage: float = 0.3,
                            min_rows: int = 1000) -> bool:
        
        has_enough_rows = len(df) >= min_rows
        
        missing_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        acceptable_missing = missing_percentage <= max_missing_percentage
        
        is_not_empty = not df.empty
        
        self.validation_report['data_quality'] = {
            'total_rows': len(df),
            'min_rows_required': min_rows,
            'has_enough_rows': has_enough_rows,
            'missing_percentage': missing_percentage,
            'max_missing_allowed': max_missing_percentage,
            'acceptable_missing': acceptable_missing,
            'is_not_empty': is_not_empty,
            'overall_valid': has_enough_rows and acceptable_missing and is_not_empty
        }
        
        return has_enough_rows and acceptable_missing and is_not_empty
    
    def get_validation_report(self) -> Dict[str, Any]:          # get complete validation report
        return self.validation_report
    


