import logging
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
import pandas as pd
import json
import sys
import numpy as np
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


def save_csv(csv_path, df):
    df.to_csv(csv_path, sep=",")


def save_dataset(save_file:str, data:pd.DataFrame):
    """Save output data to a json list of dicts."""
    with open(save_file, 'w') as outfile:
        json.dump(data, outfile, indent=4)


def create_folder(folder_path):
    """Create folder if does not exist yet"""
    if not os.path.exists(folder_path):
        logging.info("Creating save folder at %s", folder_path)
        os.makedirs(folder_path)
    else:
        logging.info("Save folder already exists: %s", folder_path)


def get_nested(dict_, keys):
    """Recursive funtion to access nested dictionaries"""
    if len(keys) == 1:
        return dict_[keys[0]]
    return get_nested(dict_[keys[0]], keys[1:])

def reverse(mylist: List):
    return list(reversed(mylist))

def load_partial_results(results: np.ndarray, save_file: str) -> Tuple[np.ndarray, int]:
    """Load partial results from a file if it exists."""
    if os.path.isfile(save_file):
        partial_results = pd.read_csv(save_file).values
        num_options_done = partial_results.shape[1]
        results[:, :num_options_done] = partial_results
    else:
        num_options_done = 0
    return results, num_options_done

def save_partial_results(results: np.ndarray, options: List[str], idx_option: int, save_file: str) -> None:
    """Save intermediate results."""
    pd.DataFrame(results[:, :idx_option + 1], columns=options[:idx_option + 1]).to_csv(save_file, index=False)


def load_dataset(dataset_config: DictConfig, columns: List[str], options: List[str]):
    dataset_cols = {dataset_config[col]: col for col in columns}
    dataset_options = {dataset_config[op]: op for op in options}
    data = pd.json_normalize(load_json(dataset_config["file"]))
    data = data.rename(columns=dataset_cols)
    data = data[list(dataset_cols.values())]
    data["cop"] = data["cop"].map(dataset_options)
    data["dataset"] = dataset_config['name']
    return data