# Medical Question Classification

This project aims to classify medical questions into their respective medical fields using LLMs.

## Table of Contents

- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)


## Project Overview

The primary goal of this project is to develop a model that can accurately classify medical questions into predefined medical fields. This is particularly useful for organizing large datasets of medical questions and for improving the efficiency of medical information retrieval systems.

This project downloads datasets from HuggingFace, classifies the questions into the medical fields specified in the configuration file, and creates a merged dataset. The merged dataset includes the original questions, their options, the correct option, the predicted medical field, the chain of thought (CoT) for the medical field, and the log probability of the CoT medical field.

## Directory Structure

```
medical-fields/
├── configs             # Configuration files for the project
├── containers          # Containerization files (e.g., Docker)
├── project-root        # Project root directory
├── src                 # Source code for the project
├── tests               # Unit tests
├── README.md           # Project README file
├── requirements.txt    # Python dependencies
```

## Installation

To install the necessary dependencies, run:

```
pip install -r requirements.txt
```

## Usage

To use the classification model, follow these steps:

1. **Prepare your dataset:**
    - Include the datasets from HuggingFace in the `configs/datasets` configuration file. Make sure each dataset is properly listed and configured.

2. **Run the main script:**
    - Use the main script to dclassify the questions and create the merged dataset:

    ```
    python src/main.py --config configs/main.yaml
    ```

3. **Configuration details:**
    - The `main.yaml` configuration file should include all necessary parameters for downloading the datasets, specifying the medical fields, and any other required configurations. Ensure this file is correctly set up before running the script.

4. **Prompt Configuration:**
    - The prompt configuration allows for chain of thought (CoT) and few-shot learning setups. These settings can be found and adjusted in the `configs/prompt` directory.
