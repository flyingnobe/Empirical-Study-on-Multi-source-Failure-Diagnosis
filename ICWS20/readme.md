# Root-Cause Metric Location for Microservice Systems via Log Anomaly Detection

## Description

### Folder Structure

-   `./drain3`: it is used to extract log template, which is cloned from [drain3](https://github.com/logpai/Drain3)
-   `./ICWS20`: it is the folder of RAMLD

### File Description

-   `ICWS20/model_p.py`、`ICWS20/model_t.py`：the code of DeepLog
-   `ICWS20/kpi_selection.py`：the code of RCA

## Environment

-   Linux Server 20.04.1 LTS
-   Intel(R) Xeon(R) CPU E5-2650 v4@ 2.20GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv

Run the following commands in turn at bash or shell

1. `pip install -r requirements.txt`
2. `python Drain3/drain.py`
3. `python ICWS20/preprocess.py`
4. `python ICWS20/model_p.py`
5. `python ICWS20/model_t.py`
6. `python ICWS20/kpi_selection.py`

## Reproduce

1. download dataset from we provide for you
2. change the path in the code to the dataset you need
3. follow the Getting Started
