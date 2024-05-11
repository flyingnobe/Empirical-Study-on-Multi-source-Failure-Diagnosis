r"""
处理groundtruth至run_table
(初赛)22AIops Challenge Dataset Anomaly Type:
pod-failure
memory
loss
delay
corrupt

文件范围：
    groundtruth文件夹下，包含groundtruth的文件
文件输入格式:
    (JSON) timestamp(10), service, cmdb_id, failure_type
文件输出格式:
    ( CSV) index, service, instance, anomaly_type([login failure]), st_time(timestamp-10), ed_time(timestamp-10), duration(600), data_type(train/test)

"""

# 相关文件配置

# CASE_ANOMALY_DICT = {
#     "[cpu]": 40,
#     "[io]": 50,
#     "[memory]": 40,
#     "[network]": 60,
#     "[process]": 27,
# }
GROUNDTRUTH_PATH = "platform_data/groundtruth.json"
TEST_RATE = 0.3

# TEST_RATE_DICT = {
#     "[cpu]": 0.2,
#     "[io]": 0.1,
#     "[memory]": 0.3,
#     "[network]": 0.1,
#     "[process]": 0.1,
# }


# 执行代码
def generate_run_table(config):
    print("开始生成run_table")
    import os
    import json
    import random
    import numpy as np
    import pandas as pd

    ANOMALY_DICT = {
        "cpu anomaly": "cpu_anomaly",
        "http/grpc request abscence": "http/grpc_request_abscence",
        "http/grpc requestdelay": "http/grpc_request_delay",
        "memory overload": "memory_overload",
        "network delay": "network_delay",
        "network loss": "network_loss",
        "pod anomaly": "pod_anomaly",
    }

    groundtruth_paths = []
    run_table = None
    for dirpath, _, filenames in os.walk(config["raw_data"]["dataset_entry"]):
        for filename in filenames:
            if filename.find("groundtruth.csv") != -1:
                full_log_path = os.path.join(dirpath, filename)
                groundtruth_paths.append(full_log_path)
    for path in groundtruth_paths:
        groundtruth_df = pd.read_csv(path)
        if groundtruth_df is None:
            run_table = groundtruth_df
        else:
            run_table = pd.concat([run_table, groundtruth_df])
        print(f"成功处理{path}")

    run_table = run_table.rename(
        columns={
            "故障类型": "failure_type",
            "对应服务": "cmdb_id",
            "起始时间戳": "st_time",
            "截止时间戳": "ed_time",
            "持续时间": "duration",
        }
    )

    def meta_transfer(item):
        if item.find("(") != -1:
            item = eval(item)
            item = item[0]
        return item

    run_table.loc[:, "cmdb_id"] = run_table["cmdb_id"].apply(meta_transfer)
    run_table = run_table.rename(columns={"st_time": "timestamp"})
    run_table = run_table.reset_index(drop=True)
    run_table.loc[:, "failure_type"] = run_table["failure_type"].apply(
        lambda x: ANOMALY_DICT[x]
    )


    # 重命名
    run_table = run_table.rename(
        columns={
            "timestamp": "st_time",
            "failure_type": "anomaly_type",
            "cmdb_id": "instance",
        }
    )
    # 按时间排序
    run_table = run_table.sort_values(by="st_time")

    run_table["anomaly_type"] = run_table["anomaly_type"].apply(lambda x: "[" + x + "]")
    print(" ".join(sorted(list(set(run_table["anomaly_type"].tolist())))))

    run_table["service"] = run_table["instance"]
    run_table["ed_time"] = run_table["st_time"] + run_table["duration"]
    run_table.reset_index(drop=True, inplace=True)

    # 划分训练测试集
    run_table["data_type"] = ["train"] * len(run_table)
    anomaly_cnt = run_table.groupby(by="anomaly_type")["instance"].count()

    anomaly_cnt_dict = anomaly_cnt.to_dict()
    for anomaly, group in run_table.groupby(by="anomaly_type"):
        sample_cnt = int(anomaly_cnt_dict[anomaly] * TEST_RATE)
        group_index = group.index.to_list()
        test_choices = group_index[-sample_cnt:]
        for choice in test_choices:
            run_table.loc[choice, "data_type"] = "test"

    # run_table.drop(columns=["level"], inplace=True)

    run_table_dir = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
    )
    run_table_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )
    if not os.path.exists(run_table_dir):
        os.makedirs(run_table_dir)
    run_table.to_csv(run_table_path)
    print("生成完毕")
