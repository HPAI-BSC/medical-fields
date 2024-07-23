import argparse
import rootutils
from omegaconf import DictConfig
import hydra
from hydra import initialize, compose


from utils import (
    download_dataset, 
    load_dataset,
    create_dir
)

rootutils.setup_root(__file__, indicator="project-root", pythonpath=True)


@hydra.main()
def main(cfg: DictConfig):
    """
    Classify questions of datasets in medical categories
    """

    # Run classification
    datasets = cfg.datasets.datasets
    datasets_dir = cfg.datasets.datasets_dir

    create_dir(datasets_dir)
    if type(datasets) == DictConfig:
        datasets = [datasets]

    for dataset in datasets:
        dataset_file = download_dataset(dataset.url, datasets_dir)
        load_dataset(dataset_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert Parquet file to JSON.')
    parser.add_argument('--config_name', dest="config_name", type=str, help='Config to apply: main.yaml or test.yaml', required=True)
    
    args = parser.parse_args()

    with initialize(config_path="../configs", version_base="1.3"):
        cfg = compose(config_name=args.config_name)
        main(cfg)
