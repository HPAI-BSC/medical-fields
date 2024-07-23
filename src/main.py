import argparse
import rootutils
from omegaconf import DictConfig
import hydra
from hydra import initialize, compose
import logging
import itertools
from pandas import DataFrame
from omegaconf import DictConfig, ListConfig
import os
import json

from typing import  Union, Optional, List

from utils import (
    create_dir,
    setup_logging, 
    save_dataset,
    download_dataset, 
    process_dataset
)
from inference import Inference

rootutils.setup_root(__file__, indicator="project-root", pythonpath=True)

def save_results(dataset: DataFrame, save_folder: str, dataset_name: str, options: List[str], split: str):
    dataset = dataset.dropna() # Drop NA values that are misprocessed samples (CareqA id: d50092cd-b772-4ca7-a17c-74062af1f76a).
    for option in options: 
        # Check if data is already processed
        output_file = os.path.join(save_folder, f"{dataset_name}_{option}_{split}.json")
        if os.path.isfile(output_file):
            continue
        
        tmp_dataset = dataset[dataset["medical_field"]== option]      
        filtered_dataset = tmp_dataset.reset_index(drop=True).to_dict('records')
        save_dataset(output_file, filtered_dataset)


def join_results_by_field(save_folder: str, options: List[str], split: str):
    """Join results by field."""
    for option in options:
        results_folder = os.path.join(save_folder, "results", option)
        os.makedirs(results_folder, exist_ok=True)
        output_file = os.path.join(results_folder, f"{split}.json")
        if os.path.exists(output_file):
            return
        combined_datasets = []
        for filename in os.listdir(save_folder):
            if filename.endswith(f'{option}_{split}.json'):
                with open(os.path.join(save_folder, filename), 'r') as file:
                    combined_datasets.append(json.load(file))
        combined_datasets = list(itertools.chain.from_iterable(combined_datasets))
        save_dataset(output_file, combined_datasets)



def run_inference(
    save_folder: str,
    inference_engine: Inference,
    datasets: Union[ListConfig, DictConfig],
    datasets_dir: str, 
    split: str,
    num_samples: Optional[int],
    block_size: int,
    merged_dataset_columns: List[str], 
    merged_dataset_options: List[str]
):

    # Iterate over datasets
    medical_specialities = inference_engine.options
    for dataset_config in datasets:
        dataset_name = dataset_config.name
        logging.info(f" Evaluating dataset: {dataset_name}")

        # Check if all files already exist
        all_files_exist = all(
            os.path.isfile(os.path.join(save_folder, f"{dataset_name}_{option}_{split}.json"))
            for option in medical_specialities
        )
        if all_files_exist:
            logging.info(f"All result files for {dataset_name} already exist. Skipping inference.")
            continue
        
        dataset_file = download_dataset(dataset_config.url, datasets_dir)
        dataset_cols = {dataset_config[col]: col for col in merged_dataset_columns if col in dataset_config}
        dataset_options = {dataset_config[op]: op for op in merged_dataset_options}
        dataset = process_dataset(dataset_file, dataset_name, dataset_config.split, dataset_cols, dataset_options).iloc[:num_samples]

        # Run inference
        predictions, logprobs, cot_prediction = inference_engine.predict(dataset['question'], block_size)

        # Save intermediate results
        dataset["medical_field"] = predictions
        dataset["cot_medical_field"] = cot_prediction
        dataset["cumulative_logprob_cot_medical_field"] = logprobs
        save_results(dataset, save_folder, dataset_name, medical_specialities, split)
    join_results_by_field(save_folder, medical_specialities, split)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """
    Classify questions of datasets in medical categories
    """

    # Datasets configuration
    datasets = cfg.datasets.datasets
    if type(datasets) == DictConfig:
        datasets = [datasets]
    datasets_dir = cfg.datasets.datasets_dir

    # Setup output directories
    setup_logging(cfg.log_level)
    create_dir(cfg.save_folder)
    create_dir(datasets_dir)

    # Run classification
    inference_engine = hydra.utils.instantiate(cfg.inference_engine)
    run_inference(
        cfg.save_folder, inference_engine, 
        datasets, datasets_dir, cfg.datasets.target_split, 
        cfg.num_samples, cfg.block_size,
        cfg.merged_dataset.columns, cfg.merged_dataset.options
    )


if __name__ == "__main__":
    main()
