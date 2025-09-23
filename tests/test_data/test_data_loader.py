import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from src.data.data_loader import DataLoader

class TestDataLoader:                   # Test cases for DataLoader class
        
    def setup_method(self):             # Set up test fixtures before each test method
        
        self.data_loader = DataLoader()
        
        self.test_data = pd.DataFrame({     # Create temporary test data
            'feature1': [1, 2, 3, 4, 5],
            'feature2': ['A', 'B', 'C', 'D', 'E'],
            'Churn': ['No', 'Yes', 'No', 'Yes', 'No']
        })
        
    def test_load_raw_data_success(self):           # Test successful data loading
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.test_data.to_csv(f.name, index=False)
            temp_path = f.name
            
        try:
            # Test loading
            loaded_data = self.data_loader.load_raw_data(temp_path)
            
            # Assertions
            assert isinstance(loaded_data, pd.DataFrame)
            assert loaded_data.shape == self.test_data.shape
            assert list(loaded_data.columns) == list(self.test_data.columns)
            
        finally:
            # Cleanup
            os.unlink(temp_path)
            
    def test_load_raw_data_file_not_found(self):
        """Test data loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            self.data_loader.load_raw_data("non_existent_file.csv")
            
    def test_create_train_test_split(self):
        """Test train-test split functionality"""
        train_df, test_df = self.data_loader.create_train_test_split(
            self.test_data, 
            target_col='Churn',
            test_size=0.4,
            random_state=42
        )
        
        # Assertions
        assert isinstance(train_df, pd.DataFrame)
        assert isinstance(test_df, pd.DataFrame)
        assert len(train_df) + len(test_df) == len(self.test_data)
        assert train_df.columns.tolist() == self.test_data.columns.tolist()
        assert test_df.columns.tolist() == self.test_data.columns.tolist()