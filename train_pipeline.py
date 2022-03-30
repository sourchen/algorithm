import numpy as np
from processing.data_management import (load_dataset, save_model)
from model import learning_algorithm
from config import config

def run_training() -> None:
  """Train the model."""
  
  # read training data
  data = load_dataset(file_name=config.TRAINING_DATA_FILE)
  X_train, y_train = data[config.FEATURES], data[config.TARGET]

  # train model
  trained_model = learning_algorithm(X_train, y_train)
  save_model(trained_model, save_file_name='trained_model')

  if __name__ == '__main__':
      run_training()