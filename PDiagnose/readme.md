# PDiagnose Of Empirical-Study-on-Multi-source-Failure-Diagnosis

## Description

### Folder Structure

-   `21aiops` `22aiops` `gaia` `platform`: they store the results of each dataset

### File Description

-   `22AIOps_run_table.csv` : the run_table file of the 22AIOps dataset
-   `aiops21_groundtruth.csv` : the run_table file of the 21AIOps dataset
-   `gaia_resplit.csv` : the run_table file of the gaia dataset
-   `run_table.csv` : the run_table file of the MicroServo dataset
-   `kpi_detection.ipynb` : the entry file of kpi detection
-   `log_analysis.ipynb` : the entry file of log analysis
-   `trace_analysis.ipynb` : the entry file of trace analysis
-   `localization.ipynb` : the entry file of localization

## Environment

-   Linux Server 20.04.1 LTS
-   Intel(R) Xeon(R) CPU E5-2650 v4@ 2.20GHz
-   Python version 3.9

## Getting Started

> Recommend conda environment or venv

Run the following commands in turn at bash or shell

1. `git clone https://github.com/flyingnobe/Empirical-Study-on-Multi-source-Failure-Diagnosis`
2. `cd PDiagnose`
3. preprocess datasets
4. run `kpi_detection.ipynb` `log_analysis.ipynb` `trace_analysis.ipynb`
5. run `localization.ipynb` 
