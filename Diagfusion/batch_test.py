from transforms.events import (
    fasttext_with_DA,
    sententce_embedding,
    metric_trace_log_parse,
)
from models import He_DGL
from public_function import deal_config, get_config
import os
import pandas as pd
import torch
import numpy as np
import random
from datetime import datetime
import json


def set_seed(config):
    seed = config["fasttext"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def exec(config):
    label_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )
    labels = pd.read_csv(label_path, index_col=0)

    print("[parse]")
    metric_trace_log_parse.run_parse(deal_config(config, "parse"), labels)

    print("[fasttext]")
    fasttext_with_DA.run_fasttext(deal_config(config, "fasttext"), labels)

    print("[sentence_embedding]")
    sententce_embedding.run_sentence_embedding(
        deal_config(config, "sentence_embedding")
    )

    print("[dgl]")
    lab_id = 9  # 实验唯一编号
    he_dgl = He_DGL.UnircaLab(deal_config(config, "he_dgl"))
    return he_dgl.train_test(lab_id)


def load_data(config):
    trace = None
    metric = None
    logs = None
    if config["log_path"]:
        logs = np.load(config["log_path"], allow_pickle=True)
    if config["metric_path"]:
        with open(config["metric_path"], "r", encoding="utf8") as fp:
            metric = json.load(fp)
    if config["trace_path"]:
        with open(config["trace_path"], "r", encoding="utf8") as fp:
            trace = json.load(fp)
    return logs, metric, trace


def reset_config(config):
    config["parse"]["metric_path"] = "anomalies/scope_metric.json"
    config["parse"]["trace_path"] = "anomalies/scope_trace.json"
    config["parse"]["log_path"] = "anomalies/scope_logs.npy"
    config["he_dgl"]["run_table"] = "scope_demo.csv"
    return config


def save_scope(run_table, log, metric, trace, config, run_table_save_path):
    new_log_path = config["log_path"]
    new_metric_path = config["metric_path"]
    new_trace_path = config["trace_path"]
    new_log = []
    new_metric = {}
    new_trace = {}
    for idx in run_table.index.tolist():
        new_log.append(log[idx])
        new_metric[str(idx)] = metric[str(idx)]
        new_trace[str(idx)] = trace[str(idx)]
    run_table.to_csv(run_table_save_path)
    np.save(new_log_path, new_log)
    with open(new_metric_path, "w") as w:
        json.dump(new_metric, w)
    with open(new_trace_path, "w") as w:
        json.dump(new_trace, w)


if __name__ == "__main__":
    split = 5
    config = get_config()
    set_seed(config)
    # load data
    label_path = os.path.join(
        config["base_path"],
        config["demo_path"],
        config["label"],
        config["he_dgl"]["run_table"],
    )
    run_table = pd.read_csv(label_path, index_col=0)
    log, metric, trace = load_data(deal_config(config, "parse"))

    train_table = run_table.query(f"data_type == 'train'")
    test_table = run_table.query(f"data_type == 'test'")

    # reset config
    config = reset_config(config)
    record = []
    for i in range(split):
        print(f"\033[35m{i + 1} / {split} * dataset\033[0m")

        ed_idx = int(len(train_table) * (i + 1) / split)
        new_run_table = pd.concat([train_table.iloc[0:ed_idx, :], test_table])
        save_scope(
            new_run_table,
            log,
            metric,
            trace,
            deal_config(config, "parse"),
            os.path.join(
                config["base_path"],
                config["demo_path"],
                config["label"],
                config["he_dgl"]["run_table"],
            ),
        )
        metrics, _, _ = exec(config)
        record.append([f"{i + 1} / {split}"] + metrics)
    if not os.path.exists("tmp_exp"):
        os.mkdir("tmp_exp")
    pd.DataFrame(
        record,
        columns=[
            "data size",
            "service top@1",
            "service top@2",
            "service top@3",
            "service top@4",
            "service top@5",
            "instance top@1",
            "instance top@2",
            "instance top@3",
            "instance top@4",
            "instance top@5",
            "weighted precision",
            "weighted recall",
            "weighted f1score",
            "micro precision",
            "micro recall",
            "micro f1score",
            "macro precision",
            "macro recall",
            "macro f1score",
        ],
    ).to_csv(os.path.join("tmp_exp", config["dataset"] + f"_split_{split}.csv"))
