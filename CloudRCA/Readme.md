# CloudRCA Of Empirical-Study-on-Multi-source-Failure-Diagnosis

## Description
A easy start demo. 

## Structure

+  `./parsed_data`: it stores the preprocessed data for CloudRCA
+  `main.py`: lanuch the framework 

## Environment
-   Linux Server 20.04.1 LTS
-   Intel(R) Xeon(R) CPU E5-2650 v4@ 2.20GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv  
> We provide three dataset, all have been pre-processed, thus the complicated preprocessing section can be skipped  
> Make your own dataset refer to [this paper](https://dl.acm.org/doi/abs/10.1145/3459637.3481903)

1. `pip install -r requirements.txt`
2. `run python main.py --data <dataset>`

Eg. python main.py --data aiops22

Four datasets' name are 'gaia' 'aiops21' and 'aiops22', refer to files in './parsed_data'