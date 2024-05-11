TEST_RATE = 0.2
# TEST_CASES = {
#     "[cpu]": 9,  # .2
#     "[io]": 16,  # .2
#     "[memory]": 9,  # .2
#     "[network]": 15,  #  数量太多, 挑15
#     "[process]": 5,  # .2
# }

# 54


# 执行代码
def generate_run_table(config):
    print("开始生成run_table")
    import os
    import json
    import pandas as pd
    from datetime import datetime

    run_table = pd.read_csv(
        os.path.join(config["raw_data"]["dataset_entry"], "aiops21_groundtruth_new.csv"),
        index_col=0,
    )

    # 2021-03-04 11:50:00.000000
    run_table["st_time"] = run_table["st_time"].apply(
        lambda x: int(
            datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp()
        )
    )
    run_table["ed_time"] = run_table["ed_time"].apply(
        lambda x: int(
            datetime.strptime(x.split(".")[0], "%Y-%m-%d %H:%M:%S").timestamp()
        )
    )

    # 按时间排序
    run_table = run_table.sort_values(by="st_time")

    run_table["anomaly_type"] = run_table["anomaly_type"].apply(
        lambda x: "[" + x.replace("\n", "") + "]"
    )

    import re

    # 划分训练测试集
    run_table["data_type"] = ["train"] * len(run_table)

    anomaly_cnt = run_table.groupby(by="anomaly_type")["instance"].count()
    # 固定选取每个 anomaly 的后 30% 的作为测试
    anomaly_cnt_dict = anomaly_cnt.to_dict()
    for anomaly, group in run_table.groupby(by="anomaly_type"):
        sample_cnt = int(anomaly_cnt_dict[anomaly] * TEST_RATE)
        group_index = group.index.to_list()
        test_choices = group_index[-sample_cnt:]
        for choice in test_choices:
            run_table.loc[choice, "data_type"] = "test"

    run_table = run_table.sort_values(by="st_time")
    run_table = run_table.reset_index(drop=True).drop(
        columns=["id", "故障类别", "故障内容", "time"]
    )

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
