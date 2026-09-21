# This file is responsible for loading the data 

#Importing the libraries

import os
import pandas as pd
from typing import Tuple, Optional

# This class is responsible for callling and loading the data 

class DataLoader:
    '''This class is responsible for loading the data set'''
    def __init__(self, train_path: str, test_path: str ):

        self.train_path = train_path
        self.test_path = test_path
        
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def _validate_path(self, path: str, label: str) -> None:
        """Raises FileNotFoundError if the given path doesn't exist. Shared by all load_* methods to avoid repeating the same check."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} file not found at: {path}")

    def load_train_data(self) -> pd.DataFrame:

        self._validate_path(self.train_path, "Train")
        
        return pd.read_csv(self.train_path)
    
    def load_test_data(self) -> pd.DataFrame:

        self._validate_path(self.test_path, "Test")
        
        return pd.read_csv(self.test_path)

    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:

        return self.load_train_data(), self.load_test_data()