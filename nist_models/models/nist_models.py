import json
from pathlib import Path
from typing import List, Callable
import pandas as pd
import numpy as np

class NISTModel:
  _models_data = None
  
  def __init__(self, name: str):
    self.name = name
    self._load_models_data()
    self._validate_model()
    
    data = self._models_data[name]
    self._difficulty = data["difficulty"]
    self._observations_no = data["observations_no"]
    self._parameters_no = data["parameters_no"]
    self._start_1 = data["start_1"]
    self._start_2 = data["start_2"]
    self._certified = data["certified"]
    self._residual_sum_squares = data["residual_sum_squares"]
    self._model_function = eval(data["model_function"])

  @classmethod
  def _load_models_data(cls):
    if cls._models_data is None:
      json_path = Path(__file__).parent / "data" / "models" / "nist_models.json"
      with open(json_path) as f:
        cls._models_data = json.load(f)

  def _validate_model(self):
    if self.name not in self._models_data:
      available = ", ".join(self._models_data.keys())
      raise ValueError(f"Unknown model '{self.name}'. Available models: {available}")

  # Individual getter methods
  def get_difficulty(self) -> str:
    """Returns the difficulty level of the model"""
    return self._difficulty

  def get_observations_count(self) -> int:
    """Returns the number of observations in the dataset"""
    return self._observations_no

  def get_parameters_count(self) -> int:
    """Returns the number of parameters in the model"""
    return self._parameters_no

  def get_certified_values(self) -> List[float]:
    """Returns the certified parameter values"""
    return self._certified

  def get_residual_sum_squares(self) -> float:
    """Returns the residual sum of squares for the certified fit"""
    return self._residual_sum_squares

  def get_start_values(self, set_num: int = 1) -> List[float]:
    """Returns starting values for specified set (1 or 2)"""
    if set_num == 1:
        return self._start_1
    elif set_num == 2:
        return self._start_2
    raise ValueError("set_num must be 1 or 2")

  def model(self) -> Callable:
    """Returns the model function"""
    return self._model_function
  
  def data(self) -> pd.DataFrame:
    """Loads the dataset from CSV"""
    csv_path = Path(__file__).parent / "data" / "processed" / f"{self.name}.csv"
    if not csv_path.exists():
      raise FileNotFoundError(f"Data file not found: {csv_path}")
    return pd.read_csv(csv_path)

  def evaluate(self, x: float, params: List[float]) -> float:
    """Evaluates the model at given x values with specified parameters"""
    return self._model_function(x, *params)