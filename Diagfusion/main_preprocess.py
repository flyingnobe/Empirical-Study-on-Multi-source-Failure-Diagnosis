from transforms import process_on_aiops21
from transforms import process_on_aiops22
from transforms import process_on_platform
from transforms import process_on_tt
from public_function import get_config, deal_config
import pandas as pd
import torch
import numpy as np
import random
import time


def set_seed(config):
    seed = config["fasttext"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


if __name__ == "__main__":
    import os

    shot = time.time()
    config = get_config()
    set_seed(config)

    run_table_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )

    if config["dataset"] == "platform":
        proxy = process_on_platform
    elif config["dataset"] == "aiops22":
        proxy = process_on_aiops22
    elif config["dataset"] == "aiops21":
        proxy = process_on_aiops21
    elif config["dataset"] == "tt":
        proxy = process_on_tt
    else:
        raise Exception("unknown dataset")

    proxy.generate_run_table(config)
    run_table = pd.read_csv(run_table_path)

    # 创建文件夹
    store_dir = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["raw_data"]["store_dir"],
    )
    if not os.path.exists(store_dir):
        os.makedirs(store_dir)
    # 处理log
    proxy.process_log(deal_config(config, "raw_data"), run_table)
    # 处理metric
    proxy.process_metric(deal_config(config, "raw_data"), run_table)
    # 处理trace
    proxy.process_trace(deal_config(config, "raw_data"), run_table_path)
    print(f"Cost: {(time.time() - shot) / float(len(run_table))} s/per-case")
