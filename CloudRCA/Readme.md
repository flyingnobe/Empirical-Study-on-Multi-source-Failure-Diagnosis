# CloudRCA Of Empirical-Study-on-Multi-source-Failure-Diagnosis

## Artifact Description
A simple demo for CloudRCA was provided, skipping the raw data processing section and offering pre-processed data instead.  

## Structure

+  `./parsed_data`: it stores the preprocessed data for CloudRCA
+  `main.py`: lanuch the framework 

## Environment
-   Linux Server 20.04.1 LTS
-   Intel(R) Xeon(R) CPU E5-2650 v4@ 2.20GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv  

1. `pip install -r requirements.txt`
2. `run python main.py --data <dataset>`

Eg. python main.py --data aiops22

Four datasets' name are 'platform' 'gaia' 'aiops21' and 'aiops22', refer to files in './parsed_data'

## Reproducibility Instructions
> We offer four preprocessed datasets, eliminating the need for the intricate preprocessing section.   
> You may create your own dataset based on the methodology outlined in [this paper](https://dl.acm.org/doi/abs/10.1145/3459637.3481903), utilizing the raw data we provide.

