# utils/preprocessing.py
import numpy as np
import pandas as pd
import torch
import logging

def preprocess_input(data):
    try:
        if data is None or not isinstance(data, pd.DataFrame):
            raise ValueError("Invalid data provided for preprocessing. Expected a pandas DataFrame.")
        data = pd.get_dummies(data, drop_first=True).fillna(0)
        data = data.to_numpy().astype(np.float32)
        return torch.tensor(data, dtype=torch.float32)
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

