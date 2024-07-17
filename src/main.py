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
    create_folder,
    setup_logging, 
    create_folder,
    save_dataset,
    load_dataset
)
from inference import Inference

rootutils.setup_root(__file__, indicator="project-root", pythonpath=True)

def save_results(dataset: DataFrame, save_folder: str, dataset_name: str, options: List[str]):
    for option in options: 
        # Check if data is already processed
        output_file = os.path.join(save_folder, f"{dataset_name}_{option}.json")
        if os.path.isfile(output_file):
            continue
        
        tmp_dataset = dataset[dataset["medical_field"]== option]      
        filtered_dataset = tmp_dataset.reset_index(drop=True).to_dict('records')
        save_dataset(output_file, filtered_dataset)


def join_results_by_field(save_folder: str, options: List[str]):
    """Join results by field."""
    results_folder = os.path.join(save_folder, "results")
    os.makedirs(results_folder, exist_ok=True)
    for option in options:
        output_file = os.path.join(results_folder, f"{option}.json")
        if os.path.exists(output_file):
            return
        combined_datasets = []
        for filename in os.listdir(save_folder):
            if filename.endswith(f'_{option}.json'):
                with open(os.path.join(save_folder, filename), 'r') as file:
                    combined_datasets.append(json.load(file))
        combined_datasets = list(itertools.chain.from_iterable(combined_datasets))
        save_dataset(output_file, combined_datasets)



def run_inference(
    save_folder: str,
    inference_engine: Inference,
    datasets: Union[ListConfig, DictConfig],
    num_samples: Optional[int],
    dataset_columns: List[str], 
    dataset_options: List[str]
):

    # Iterate over datasets
    medical_specialities = inference_engine.options
    for dataset_config in datasets:
        dataset_name = dataset_config.name
        logging.info(f" Evaluating dataset: {dataset_name}")
        
        dataset = load_dataset(dataset_config, dataset_columns, dataset_options).iloc[:num_samples]
        # Check if all files already exist
        all_files_exist = all(
            os.path.isfile(os.path.join(save_folder, f"{dataset_name}_{option}.json"))
            for option in medical_specialities
        )
        if all_files_exist:
            logging.info(f"All result files for {dataset_name} already exist. Skipping inference.")
            continue
        
        # Run inference
        predictions, logprobs, cot_prediction = inference_engine.predict(dataset['question'])
        dataset["medical_field"] = predictions
        dataset["cot_medical_field"] = cot_prediction
        dataset["cumulative_logprob_cot_medical_field"] = logprobs
        save_results(dataset, save_folder, dataset_name, medical_specialities)
    join_results_by_field(save_folder, medical_specialities)


@hydra.main()
def main(cfg: DictConfig):
    """
    Classify questions of datasets in medical categories
    """
    # Setup output directories
    setup_logging(cfg.log_level)
    create_folder(cfg.save_folder)

    # Run classification
    datasets = cfg.datasets.datasets
    if type(datasets) == DictConfig:
        datasets = [datasets]
    inference_engine = hydra.utils.instantiate(cfg.inference_engine)
    run_inference(
        cfg.save_folder, inference_engine, datasets,cfg.num_samples,
        cfg.merged_dataset.columns, cfg.merged_dataset.options
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Parquet file to JSON.')
    parser.add_argument('--config_name', dest="config_name", type=str, help='Config to apply: main.yaml or test.yaml', required=True)
    
    args = parser.parse_args()

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name=args.config_name)
        main(cfg)
