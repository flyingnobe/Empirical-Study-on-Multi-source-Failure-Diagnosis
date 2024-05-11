# Diagfusion Of Empirical-Study-on-Multi-source-Failure-Diagnosis

## Description

### Folder Structure

-   `./config`: it stores the config files of each dataset
-   `./data`: it stores the intermediary files of each preprocessed dataset
-   `./detector`: it is used to perform 3-sigma anomaly detection
-   `./drain3`: it is used to extract log template, which is cloned from [drain3](https://github.com/logpai/Drain3)
-   `./models`: it stores models used in experiment
-   `./tmp_exp`: it stores the experiment results
-   `./transform`: it stores the preprocess program of each dataset

### File Description

-   `batch_test.py`：the entry file of data volume experiment
-   `eval_test.py`：the entry file of [using all modal, removing one modal, removing two modal] experiment
-   `main_preprocess.py`：the entry file of preprocess data of each dataset
-   `main.py`：the entry file of using all modal experiment
-   `public_function.py`：the utils file
-   `test.sh`：the entry of evaluating all dataset

## Environment

-   Linux Server 20.04.1 LTS
-   Intel(R) Xeon(R) CPU E5-2650 v4@ 2.20GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv

Run the following commands in turn at bash or shell

1. `git clone https://github.com/flyingnobe/Empirical-Study-on-Multi-source-Failure-Diagnosis`
2. `cd Diagfusion`
3. `pip install -r requirements.txt`
    > If you have trouble with fasttext, download [whl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#fasttext).
4. `bash test.sh`

All result will be saved in `tmp_exp/`

## Reproducibility

1. download dataset we provide for you
2. change all five dataset config in `./config/` at the position described below:
    ```yaml
    ...
    raw_data:
        dataset_entry: 'path to the entry of this type dataset you download'
        ...
    ...
    ```
3. run `bash preprocess.sh`
4. run `bash test.sh`

All result will be saved in `tmp_exp/`
