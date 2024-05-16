import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import argparse
import yaml
from datetime import datetime


def get_data(data_entry, keyword):

    # 获取指定文件夹下的所有以 "log" 结尾的 CSV 文件
    log_files = []

    if config['dataset'] == 'platform':
        for root, dirs, files in os.walk(data_entry):
            for file in files:
                if keyword == file:
                    log_files.append(os.path.join(root, file))
    else:
        for root, dirs, files in os.walk(data_entry):
            for file in files:
                if keyword in file:
                    log_files.append(os.path.join(root, file))

    # 如果没有找到符合条件的文件，打印提示信息并返回
    if not log_files:
        print("No log files found in the specified folder.")
        return None

    # 初始化一个空的 DataFrame 用于存储合并后的数据
    merged_df = pd.DataFrame()

    # 遍历每个 log 文件，读取数据并追加到 merged_data
    for log_file in log_files:
        print("Reading", log_file)
        file_path = log_file
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)

    if keyword == 'business':
        merged_df = merged_df[merged_df['message'].map(lambda x:type(x) == str)]
        times = [time.split('|')[0].split(',')[0] for time in merged_df['message'].values.tolist()]
        timestamps = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S").timestamp() for time in times]
        merged_df = merged_df.rename(columns={'service': 'cmdb_id'})
        merged_df['timestamp'] = timestamps
    merged_df = merged_df.sort_values(by=["timestamp"]).reset_index(drop=True)
    # print(merged_df.columns)
    # merged_df.to_csv("datasets/aiops2022-pre/log/log.csv")

    return merged_df


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="platform.yaml")
    args = parser.parse_args()
    with open(
        os.path.join("ICWS20/ICWS20/config", args.config),
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    return config


IGNORE_INSTANCE = ["redis-cart-0", "redis-cart-1", "redis-cart-2", "redis-cart2-0"]
config = get_config()

if config["dataset"] == "aiops22":
    log_df = get_data(config["dataset_entry"],'log')
    log_df = log_df.query(f"cmdb_id not in {IGNORE_INSTANCE}")
    run_table_path = config["run_table"]
    run_table_df = pd.read_csv(run_table_path)
    ft = run_table_df["failure_type"].unique()[0]
    run_table_df = run_table_df[run_table_df["failure_type"].map(lambda x: x.find('ft')==-1)]
    log_df = log_df[log_df["failure_type"].map(lambda x: x.find('ft')==-1)]
    anormaly_times = run_table_df["timestamp"].values.tolist()
elif config["dataset"] == "aiops21":
    log_df = get_data(config["dataset_entry"],'log')
    run_table_path = config["run_table"]
    run_table_df = pd.read_csv(run_table_path)
    ft = run_table_df["failure_type"].unique()[0]
    run_table_df = run_table_df[run_table_df["故障类别"].map(lambda x: x.find('ft')==-1)]
    log_df = log_df[log_df["故障类别"].map(lambda x: x.find('ft')==-1)]
    anormaly_times = [time/1000 for time in run_table_df["time"].values.tolist()]
elif config['dataset'] == 'gaia':
    log_df = get_data(config["dataset_entry"],'business')
    run_table_path = config["run_table"]
    run_table_df = pd.read_csv(run_table_path)
    ft = run_table_df["failure_type"].unique()[0]
    run_table_df = run_table_df[run_table_df["message"].map(lambda x: x.find('ft')==-1)]
    log_df = log_df[log_df["message"].map(lambda x: x.find('ft')==-1)]
    times = [time.split('|')[0].split(',')[0] for time in run_table_df['message'].values.tolist()]
    anormaly_times = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S").timestamp() for time in times]
elif config['dataset'] == 'platform':
    log_df = get_data(config["dataset_entry"],'log.csv')
    run_table_path = config["run_table"]
    run_table_df = pd.read_csv(run_table_path)
    anormaly_times = run_table_df["起始时间戳"].values.tolist()
    
    


window_size = 1000
column_uniques = log_df["cmdb_id"].unique()
id = 0
for column in column_uniques:
    id = id + 1
    data = []
    temp_data = log_df[log_df["cmdb_id"].isin([column])]
    interval_data = temp_data.assign(interval=temp_data[["timestamp"]].diff())
    timestamp_data = temp_data["timestamp"].values.tolist()
    

    for i in tqdm(
        range(0,len(interval_data),window_size),
        desc=f"Loading {column}"   
    ):
        # print(len(interval_data))
        if i+window_size>len(interval_data):
            break
        interval_df = interval_data.iloc[i:i+window_size]
        intervals = interval_df["interval"].values.tolist()
        intervals[0] = 0
        window_start = interval_data.iloc[i]['timestamp']
        window_end = interval_data.iloc[i+window_size-1]['timestamp']
        label = (
            1
            if any((window_start < at) & (window_end > at) for at in anormaly_times)
            else 0
        )
        data.append([intervals,label])

    log_path = config["log_path"]
    np.save(f"{log_path}{column}_interval.npy", data)
