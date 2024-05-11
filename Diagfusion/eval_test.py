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
from sklearn.metrics import precision_score, f1_score, recall_score


def set_seed(config):
    seed = config["fasttext"]["seed"]
    np.random.seed(seed)
    random.seed(seed)
    torch.random.manual_seed(seed)


def get_metric(y_pred, y_true, average):
    pre = precision_score(y_pred, y_true, average=average)
    rec = recall_score(y_pred, y_true, average=average)
    f1 = f1_score(y_pred, y_true, average=average)
    return [pre, rec, f1]


def process_each_failure(anomalys, y_pred, y_true):
    record = []
    data = pd.DataFrame({"pred": y_pred, "true": y_true})
    for anomaly, group in data.groupby("true"):
        pred = [1 if p == anomaly else 0 for p in group["pred"]]
        true = [1 if p == anomaly else 0 for p in group["true"]]
        record.append(
            [anomalys[anomaly]]
            + [0] * 10
            + get_metric(pred, true, "weighted")
            + get_metric(pred, true, "micro")
            + get_metric(pred, true, "macro")
        )
    return record


def eval_test(config, rm):
    print(f"\033[35mrm {rm}\033[0m".center(100, "*"))
    config["parse"]["rm"] = rm
    set_seed(config)

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
    record, y_pred, y_true = he_dgl.train_test(lab_id)
    record = [["remove " + rm] + record]
    if rm == "none":
        anomalys = [
            a.lstrip(" ") + "]"
            for a in config["he_dgl"]["anomaly"].split("]")
            if a != "" and a != " "
        ]
        anomalys.sort()
        record.extend(process_each_failure(anomalys, y_pred, y_true))
    return record


if __name__ == "__main__":
    config = get_config()
    rm_list = [
        "none",
        "log",
        "metric",
        "trace",
        "log,metric",
        "log,trace",
        "metric,trace",
    ]
    records = []
    for rm in rm_list:
        record = eval_test(config, rm)
        records.extend(record)

    if not os.path.exists("tmp_exp"):
        os.mkdir("tmp_exp")

    pd.DataFrame(
        records,
        columns=[
            "experiment type",
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
    ).to_csv(os.path.join("tmp_exp", config["dataset"] + ".csv"))
