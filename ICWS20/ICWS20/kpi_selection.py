from datetime import datetime
import os
import numpy as np
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
import time


def is_anomaly(metric_values):
    """
    k-sigma检测指标值是否异常。
    :param metric_values: 指标值的时间序列矩阵
    :return: True or False
    """
    if np.all(metric_values == metric_values[0]):
        return False
    z_scores = zscore(metric_values)
    # print(abs(z_scores))
    return any(abs(z_scores) > 3)


def get_log_score(ts, ps):
    w = 0.5  # 权重
    # 计算异常得分
    AS = w * ts[:, 1] + (1 - w) * ps[:, 1]
    return AS


def get_22aiops_kpi():
    kpi_list = []
    merge_dict = {}
    for dp, dn, fn in tqdm(
        os.walk("codebase/dataset/aiops2022-pre"), desc="Loading KPIs"
    ):
        for f in fn:
            if f.startswith("kpi_container"):
                if merge_dict.get(f) is None:
                    merge_dict[f] = pd.read_csv(os.path.join(dp, f))
                else:
                    merge_dict[f] = (
                        pd.concat(
                            [merge_dict[f], pd.read_csv(os.path.join(dp, f))],
                            ignore_index=True,
                        )
                        .sort_values(by=["timestamp"])
                        .reset_index(drop=True)
                    )

    for f, df in merge_dict.items():
        grouped_data = (
            df.groupby("cmdb_id")
            .apply(lambda group: group.to_dict(orient="records"))
            .to_dict()
        )

        for cmdb_id, metrics in grouped_data.items():
            cmdb_id = cmdb_id.split(".")[-1].split("_")[-1]
            if cmdb_id.endswith("2-0"):
                continue

            kpi_list.append(
                {
                    "cmdb_id": cmdb_id,
                    "kpi_name": f,
                    "data": metrics,
                }
            )

    return kpi_list


def get_21aiops_kpi():
    kpi_list = []
    merge_dict = {}
    for dp, dn, fn in tqdm(
        os.walk("codebase/dataset/aiops2021-2"), desc="Loading KPIs"
    ):
        for f in fn:
            if f.startswith("metric"):
                if merge_dict.get(f) is None:
                    merge_dict[f] = pd.read_csv(os.path.join(dp, f))
                else:
                    merge_dict[f] = (
                        pd.concat(
                            [merge_dict[f], pd.read_csv(os.path.join(dp, f))],
                            ignore_index=True,
                        )
                        .sort_values(by=["timestamp"])
                        .reset_index(drop=True)
                    )

    for f, df in merge_dict.items():
        grouped_data = (
            df.groupby(["cmdb_id", "kpi_name"])
            .apply(lambda group: group.to_dict(orient="records"))
            .to_dict()
        )

        for (cmdb_id, kpi_name), metrics in grouped_data.items():
            detectable = (
                cmdb_id.startswith("Tomcat")
                or cmdb_id.startswith("Mysql")
                or cmdb_id.startswith("apache")
            )
            if not detectable:
                continue
            kpi_list.append(
                {
                    "cmdb_id": cmdb_id,
                    "kpi_name": kpi_name,
                    "data": metrics,
                }
            )

    return kpi_list

def get_gaia_kpi():
    kpi_list = []
    merge_dict = {}
    for dp, dn, fn in tqdm(
        os.walk("codebase/dataset/MicroSS/metric"), desc="Loading KPIs"
    ):
        for f in fn:
            if f.endswith(".csv"):
                if f.startswith('system') or f.startswith('zookeeper'):
                    continue
                if merge_dict.get(f) is None:
                    merge_dict[f.split('2021')[0]] = pd.read_csv(os.path.join(dp, f))
                else:
                    merge_dict[f.split('2021')[0]] = (
                        pd.concat(
                            [merge_dict[f], pd.read_csv(os.path.join(dp, f))],
                            ignore_index=True,
                        )
                        .sort_values(by=["timestamp"])
                        .reset_index(drop=True)
                    )
    for f, df in merge_dict.items():  
        cmdb_id = f.split('_')[0]
        kpi_name = 'xxx'
        df['timestamp'] = df['timestamp'] / 1000

        kpi_list.append(
            {
                "cmdb_id": cmdb_id,
                "kpi_name": kpi_name,
                "data": df.values.tolist(),
            }
        )

    return kpi_list

def get_platform_kpi():
    kpi_list = []
    merge_dict = {}
    for dp, dn, fn in tqdm(
        os.walk("codebase/dataset/平台数据集"), desc="Loading KPIs"
    ):
        if dp.find("whole_metric") != -1 and dp.find("whole_metric_old") == -1:
            for f in fn:
                if merge_dict.get(f) is None:
                    merge_dict[f] = pd.read_csv(os.path.join(dp, f))
                else:
                    merge_dict[f] = (
                        pd.concat(
                            [merge_dict[f], pd.read_csv(os.path.join(dp, f))],
                            ignore_index=True,
                        )
                        .sort_values(by=["timestamp"])
                        .reset_index(drop=True)
                    )

    for f, df in merge_dict.items():  
        cmdb_id = f.split('_')[0]
        kpi_name = 'xxx'

        kpi_list.append(
            {
                "cmdb_id": cmdb_id,
                "kpi_name": kpi_name,
                "data": df.values.tolist(),
            }
        )

    return kpi_list


def align_data(log_scores, metric_values):
    """
    对齐日志得分和指标值。
    :param log_scores: 包含日志得分的数组 [N,]
    :param metric_values: 包含指标值的数组 [M,]
    :return: 对齐后的日志得分和插值后的指标值
    """

    if len(log_scores) != len(metric_values):
        min_length = min(len(log_scores), len(metric_values))
        log_scores = log_scores[:min_length]
        metric_values = metric_values[:min_length]

    return log_scores, metric_values


def calculate_score(log_scores, metric_values):
    """
    计算日志异常分数与指标之间的互信息
    :param metric_values: 指标的时间序列矩阵
    :param log_scores: 日志异常分数的时间序列
    :return: 互信息值
    """
    if len(metric_values) < 4:
        return -1e3
    metric_values = metric_values.ravel()
    log_scores = log_scores.ravel()
    score = np.corrcoef(metric_values, log_scores)[0, 1]
    return score


dataset = "platform"

if dataset == "aiops22":
    test_table = pd.read_csv("ICWS20/ICWS20/cache/aiops22/test_table.csv")
    kpi_list = get_22aiops_kpi()
elif dataset == "aiops21":
    test_table = pd.read_csv("ICWS20/ICWS20/cache/aiops21/test_table.csv")
    kpi_list = get_21aiops_kpi()
elif dataset == "gaia":
    test_table = pd.read_csv("ICWS20/ICWS20/cache/gaia/test_table.csv")
    kpi_list = get_gaia_kpi()
elif dataset == "platform":
    test_table = pd.read_csv("codebase/dataset/平台数据集/run_table.csv")
    kpi_list = get_platform_kpi()


start_time = time.time()
total = {}
top1 = {}
top3 = {}
top5 = {}

group_df = test_table.groupby(by="故障类型")
min_len = 1e6
count_dict = {}
for group_name, group_data in group_df:
    _len = len(group_data)
    if _len < min_len:
        min_len = _len
    count_dict[group_name] = 0
    total[group_name] = 0
    top1[group_name] = 0
    top3[group_name] = 0
    top5[group_name] = 0

print(count_dict)


for index,row in test_table.iterrows():
    if dataset == "aiops22":
        timestamp = row['timestamp']
        failure_type = row['故障类型']
        root_cause = row['cmdb_id']
        level = row['level']

        count_dict[failure_type] += 1
        if count_dict[failure_type] == min_len:
            continue           

        if root_cause.endswith("2-0"):
            continue

        if level == "node":
            continue
    elif dataset == "aiops21":
        timestamp = row['time'] / 1000
        root_cause = row['service']

        detectable = (
            root_cause.startswith("Tomcat")
            or root_cause.startswith("Mysql")
            or root_cause.startswith("apache")
        )
        if not detectable:
            continue

    elif dataset == "gaia":
        message = row['message']
        if message.find('|') == -1:
            continue
        times = message.split('|')[0].split(',')[0]
        timestamp = datetime.strptime(times, "%Y-%m-%d %H:%M:%S").timestamp()
        root_cause = row['service']
        if root_cause.find('service') == -1:
            continue
    
    elif dataset == "platform":
        timestamp = row['起始时间戳']
        root_cause = row['对应服务']
        if root_cause.find('(') != -1:
            start_index = root_cause.find("'") + 1
            end_index = root_cause.find("'", start_index)
            root_cause = root_cause[start_index:end_index]
    # print(root_cause)

    # if index > int(len(run_table)/5): break


    # 创建一个包含cmdb_id和对应互信息值的列表
    result_list = []
    predict_cmdbs = []

    # 计算互信息排名
    for kpi in kpi_list:
        if dataset == 'gaia':
            metric_values = np.array(
                [
                    metric[1]
                    for metric in kpi["data"]
                    if timestamp - 5 * 60 <= metric[0] <= timestamp + 5 * 60
                ]
            )
        elif dataset == "platform":
             metric_values = np.array(
                [
                    metric[3]
                    for metric in kpi["data"]
                    if timestamp - 5 * 60 <= metric[0] <= timestamp + 5 * 60
                ]
            )
        else:
            metric_values = np.array(
                [
                    metric["value"]
                    for metric in kpi["data"]
                    if timestamp - 5 * 60 <= metric["timestamp"] <= timestamp + 5 * 60
                ]
            )
        if len(metric_values) == 0:
            continue
        cmdb_id = kpi["cmdb_id"]
        kpi_name = kpi["kpi_name"]

        # metric_values = metric_values[:int(len(metric_values)/5)]

        if is_anomaly(metric_values):
            ts_file_path = f"ICWS20/ICWS20/output/{cmdb_id}_ts.npy"
            ps_file_path = f"ICWS20/ICWS20/output/{cmdb_id}_ps.npy"
            if not os.path.exists(ts_file_path) or not os.path.exists(ps_file_path):
                continue

            ts_list = np.load(ts_file_path)
            ps_list = np.load(ps_file_path)
            log_score = get_log_score(ts_list, ps_list)
            # log_score = log_score[:int(len(log_score)/5)]

            # 对齐日志异常分数与指标
            aligned_log_scores, aligned_metric_values = align_data(
                log_score, metric_values
            )

            # 计算互信息
            mutual_information = calculate_score(
                aligned_log_scores, aligned_metric_values
            )

            result_list.append(
                {
                    "cmdb_id": cmdb_id,
                    "kpi_name": kpi_name,
                    "mutual_information": mutual_information,
                }
            )
    if len(result_list) == 0:
        continue
    # 根据互信息值对结果列表进行排序
    result_list = sorted(
        result_list, key=lambda x: x["mutual_information"], reverse=True
    )

    # 输出结果
    # print("Top 5 KPIs with high MI:")
    # for i in range(5):
    #     print(
    #         "CMDB_ID: {}, KPI Name: {}, MI: {}".format(
    #             result_list[i]["cmdb_id"],
    #             result_list[i]["kpi_name"],
    #             result_list[i]["mutual_information"],
    #         )
    #     )
    # print("Top 5 CMDBs with high MI:")

    for i in range(len(result_list)):
        if result_list[i]["cmdb_id"] not in predict_cmdbs:
            predict_cmdbs.append(result_list[i]["cmdb_id"])
        if len(predict_cmdbs) == 5:
            break
    
    total[failure_type] += 1
    # print(total)
    for i in range(len(predict_cmdbs)):
        if root_cause in predict_cmdbs[i]:
            if i < 1:
                top1[failure_type] += 1
            if i < 3:
                top3[failure_type] += 1
            if i < 5:
                top5[failure_type] += 1
    # print(f"RootCause: {root_cause}, Predict: {predict_cmdbs}")

for f in count_dict.keys():
    print(f"Top1: {top1[f] / total[f]}")
    print(f"Top3: {top3[f] / total[f]}")
    print(f"Top5: {top5[f] / total[f]}")

elapsed_time = time.time() - start_time
print("elapsed_time: {:.3f}s".format(elapsed_time))
