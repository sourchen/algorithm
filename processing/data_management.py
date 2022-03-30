import pandas as pd
from config import config


def load_dataset(file_name: str):
    data = pd.read_csv(f'{config.DATASET_DIR}/{file_name}')
    return data


def save_model(model, save_file_name: str):
    """Persist the model."""

    save_path = f'{config.TRAINED_MODEL_DIR}/'
    model.save(f'{save_path}/{save_file_name}')

    print('saved model')
