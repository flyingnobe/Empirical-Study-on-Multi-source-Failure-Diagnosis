# TrinityRCL Of Empirical-Study-on-Multi-source-Failure-Diagnosis

## Description

### Folder Structure

-   `./drain3`: it is used to extract log template, which is cloned from [drain3](https://github.com/logpai/Drain3)

### File Description

-   `main.py`：the entry file of TrinityRCL
-   `aiops22.json`：the intermediary file of preprocessed aiops22 dataset
-   `aiops22.yaml`：the config file of aiops22 dataset

## Environment

-   Linux Server 20.04.1 LTS
-   Intel(R) Xeon(R) CPU E5-2650 v4@ 2.20GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv

Run the following commands in turn at bash or shell

1. `git clone https://github.com/flyingnobe/Empirical-Study-on-Multi-source-Failure-Diagnosis`
2. `cd TrinityRCL`
3. `pip install -r requirements.txt`
4. `python main.py`

The running log will be saved at `logs/__main__.log`

## Reproducibility

1. download dataset from we provide for you
2. change `dataset_dir: "path to the entry of aiops22 dataset you download"` in `aiops22.yaml`
3. remove `aiops22.json`
4. run `python main.py`
