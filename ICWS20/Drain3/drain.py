r"""

源文件（包含log\service）：
  ,log_id,              ,timestamp      ,cmdb_id           ,log_name                            ,value
0  k3aBt38B4OJPLTSquRM4  1651334403      cartservice-0      log_cartservice-service_application  [40minfo: Microsoft.AspNet...

目标npy文件：
    [
        [
            ... log:list
            # 格式
            (timestamp(10), 'instance', 'template', 'flag')
        ] // case 0
    ]
"""

# 配置
IGNORE_INSTANCE = ["redis-cart-0", "redis-cart-1", "redis-cart-2", "redis-cart2-0"]

# 代码
from pdb import run
from sklearn.model_selection import train_test_split
from log.logparser.logparser.Drain import Drain

import json
import logging
import sys
import time
import pandas as pd
import numpy as np
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm
import argparse
import os
import yaml
from datetime import datetime


def hash_decimal_to_hex(decimal):
    # 使用Python内置哈希函数将整数哈希为一个大整数
    hashed_decimal = abs(hash(str(decimal)))
    # 将大整数转换为16进制字符串
    hex_string = hex(hashed_decimal)
    # 取字符串末尾8个字符作为哈希值，即一个长度为8的16进制数
    hash_value = hex_string[:8]
    # 将16进制数转换为整数并返回
    return hash_value


def get_drain_template(log_df):
    logger = logging.getLogger(__name__)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    config = TemplateMinerConfig()
    config.load("ICWS20/Drain3/drain3/drain.ini")
    config.profiling_enabled = True
    template_miner = TemplateMiner(config=config)

    line_count = 0

    lines = log_df["message"].values.tolist()

    start_time = time.time()
    batch_start_time = start_time
    batch_size = 10000
    result_json_list = []
    for line in tqdm(lines, desc="draining"):
        line = str(line).rstrip()
        line = line.split(",", 5)[-1]
        result = template_miner.add_log_message(line)
        line_count += 1
        if line_count % batch_size == 0:
            time_took = time.time() - batch_start_time
            rate = batch_size / time_took
            batch_start_time = time.time()
        if result["change_type"] != "none":
            result_json = json.dumps(result)
        result_json_list.append(result)

    time_took = time.time() - start_time
    rate = line_count / time_took

    sorted_clusters = sorted(
        template_miner.drain.clusters, key=lambda it: it.size, reverse=True
    )
    for cluster in sorted_clusters:
        logger.info(cluster)

    # with open('logstash_structured.json', 'w+') as f:
    #     json.dump(result_json_list, f)

    # with open('logstash_templates.txt', 'w+') as f:
    #     for cluster in sorted_clusters:
    #         f.write(str(cluster))
    #         f.write('\n')

    EID_list = []
    for logdict in result_json_list:
        EID_list.append(hash_decimal_to_hex(logdict["cluster_id"]))

    log_df["EventId"] = EID_list

    return log_df


def sampling(df: pd.DataFrame, window_size, run_table, config, column):
    save_path = config["log_path"]
    logs_list = []
    if config["dataset"] == "aiops22":
        anormaly_times = run_table["timestamp"].values.tolist()
    elif config["dataset"] == "aiops21":
        anormaly_times = [time/1000 for time in run_table["time"].values.tolist()]
    elif config['dataset'] == "gaia":
        times = [time.split('|')[0].split(',')[0] for time in run_table['message'].values.tolist()]
        anormaly_times = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S").timestamp() for time in times]
        
        for time in df['message'].values.tolist():
            if type(time) == list:
                print(time)
        times = [time.split('|')[0].split(',')[0] for time in df['message'].values.tolist()]
        timestamps = [datetime.strptime(time, "%Y-%m-%d %H:%M:%S").timestamp() for time in times]
        df['timestamp'] = timestamps
    elif config['dataset'] == "platform":
        anormaly_times = run_table["起始时间戳"].values.tolist()
        

    for i in tqdm(range(0, len(df), window_size), desc=f"日志采样{column}"):
        service_list = []
        if i+window_size>len(df):
            break
        temp_df = df.iloc[i:i+window_size]
        timestamp_list = temp_df["timestamp"].values.tolist()
        cmdb_id_list = temp_df["cmdb_id"].values.tolist()
        event_id_list = temp_df["EventId"].values.tolist()

        window_start = df.iloc[i]['timestamp']
        window_end = df.iloc[i+window_size-1]['timestamp']


        label = (
            1
            if any((window_start < at) & (window_end > at) for at in anormaly_times)
            else 0
        )

        for timestamp, cmdb_id, event_id in zip(
            timestamp_list, cmdb_id_list, event_id_list
        ):
            service_list.append([timestamp, cmdb_id, event_id])
        logs_list.append([service_list, label])
    np.save(save_path + f"{column}_template.npy", logs_list)


def process_log(config):
    print("处理log")
    print("读取日志")
    import os

    log_df = None
    log_file_path_list = []
    keyword = "log"
    if config['dataset'] == 'gaia':
        keyword = 'business'
    run_table = pd.read_csv(config["run_table"])
    if config['dataset'] == 'platform':
        for dirpath, _, filenames in os.walk(config["dataset_entry"]):
            for filename in filenames:
                if filename == 'log.csv':
                    full_log_path = os.path.join(dirpath, filename)
                    log_file_path_list.append(full_log_path)
                    print(full_log_path)
    else:
        for dirpath, _, filenames in os.walk(config["dataset_entry"]):
            for filename in filenames:
                if filename.find(keyword) != -1:
                    full_log_path = os.path.join(dirpath, filename)
                    log_file_path_list.append(full_log_path)
    for path in tqdm(log_file_path_list, total=len(log_file_path_list), desc="Reading"):
        log_data = pd.read_csv(path)
        if log_data is None:
            log_df = log_data
        else:
            log_df = pd.concat([log_df, log_data])
        # print(f"成功处理{path}")
            
    if config['dataset'] == 'gaia':
        run_table = run_table[run_table["message"].map(lambda x: x.find('|')!=-1)]
        log_df = log_df.rename(columns={'service':'cmdb_id'})
        log_df = log_df[log_df['message'].map(lambda x:type(x) == str)]
    else:
        log_df = log_df.rename(columns={"value": "message"})
    log_df = log_df.query(f"cmdb_id not in {IGNORE_INSTANCE}")
    print("提取模板，准备drain")
    log_template_df = get_drain_template(log_df)
    print("准备采样")
    column_uniques = log_template_df["cmdb_id"].unique()
    for column in column_uniques:
        temp_data = log_df[log_df["cmdb_id"].isin([column])]
        sampling(temp_data, 1000, run_table, config, column)
    # normaly_sampling(log_template_df, 3000, config["normaly_path"])
    print("处理完成")


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="platform.yaml")
    args = parser.parse_args()
    with open(
        os.path.join("ICWS20/Drain3/config", args.config),
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    return config


config = get_config()
process_log(config)
