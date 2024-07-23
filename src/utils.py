import logging
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
import pandas as pd
import json
import sys
import numpy as np
import requests
from typing import List, Tuple

def setup_logging(log_level:str):
    """Set up logging configuration."""
    logging.basicConfig(level=log_level, stream=sys.stdout)

def load_txt(txt_file:str):
    return Path(txt_file).read_text()

def load_txt_as_array(file):
    return np.loadtxt(file, dtype=str, delimiter="\t")

def load_config(config_file:str):
    """Load config file as omegaconf"""
    return OmegaConf.load(config_file)

def ensure_dir(file_path):
    """
    Ensures that the directory for the given file path exists.
    If it doesn't exist, it creates it.
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def download_dataset(url: str, save_folder: str):
    """Download dataset from url to save_file."""
    dataset_file = os.path.join(save_folder, os.path.relpath(url, start="https://huggingface.co/datasets"))
    ensure_dir(dataset_file)
    if os.path.exists(dataset_file):
        return dataset_file
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(dataset_file, 'wb') as file:
            file.write(response.content)
        print(f"File saved to {dataset_file}")
    else:
        print(f"Failed to download file. Status code: {response.status_code}")
    return dataset_file


def load_parquet(parquet_path:str):
    """Load parquet data from a file."""
    df = pd.read_parquet(parquet_path)
    df = expand_columns(df)
    return df

def load_json(json_path:str):
    """Load JSON data from a file."""
    with open(json_path, "r", encoding='utf-8') as file:
        first_char = file.read(1)
        file.seek(0)

        if first_char == '[':
            data = json.load(file)
        else:
            data = []
            for line in file:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON on line: {line}")
                        print(f"Error: {e}")
                        continue

        if len(data) == 1:
            return data[0]
        return data


def save_dataset(save_file:str, data:pd.DataFrame):
    """Save output data to a json list of dicts."""
    with open(save_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def create_dir(folder_path):
    """Create folder if does not exist yet"""
    if not os.path.exists(folder_path):
        logging.info("Creating save folder at %s", folder_path)
        os.makedirs(folder_path)
    else:
        logging.info("Save folder already exists: %s", folder_path)

def reverse(mylist: List):
    return list(reversed(mylist))

def flatten(data, parent_key='', sep='.'):
    items = []
    if isinstance(data, dict):
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten(v, new_key, sep=sep).items())
    elif isinstance(data, (list, np.ndarray)):
        for i, v in enumerate(data):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.extend(flatten(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, data))
    return dict(items)

def expand_columns(df):
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (dict, list))).any():
            expanded_data = df[col].apply(lambda x: pd.Series(flatten(x)))
            expanded_data.columns = [f"{col}.{sub_col}" for sub_col in expanded_data.columns]
            df = pd.concat([df.drop(columns=[col]), expanded_data], axis=1)
    return df

def load_dataset(dataset_file:str):
    if dataset_file.endswith(".json"):
        return pd.json_normalize(load_json(dataset_file))
    elif dataset_file.endswith(".parquet"):
        return load_parquet(dataset_file)
    
def process_dataset(dataset_file:str, dataset_name: str, split: str, dataset_cols: DictConfig, dataset_options: DictConfig):
    data = load_dataset(dataset_file)
    data = data.rename(columns=dataset_cols)
    data = data[list(dataset_cols.values())]
    data["cop"] = data["cop"].map(dataset_options)
    data["dataset"] = dataset_name
    data["split"] = split
    if "id" not in data.columns:
        data["id"] = range(len(data))
    return data