import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import re
import logging
import warnings
from drain3.template_miner import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
import json
from multiprocessing import Pool

warnings.filterwarnings("ignore")

"""
Logger
"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not os.path.exists("logs"):
    os.makedirs("logs")
fh = logging.FileHandler(f"logs/{__name__}.log")
fh.setLevel(logging.INFO)
fh.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(fh)
"""
预处理数据
Trace:
    total invocations

    failed invocations

Metric:
    time series data

Log:
    timestamp

    service identifier

    host identifier

    severity

    log messages

    executions (execution count of the event)
"""

"""
构建因果图

时间窗口为故障发生的前30分钟

根据trace数据，从可观测的发生故障的微服务出发，利用调用失败和被调用失败的关系，构建因果图

计算 (failed invocations / total invocations) 比率

问题：
    没有数据指明最一开始发生故障的微服务是哪些？

    如何去衡量是否调用成功 or 失败

指标插值

日志转时间序列，统计error数量

节点
    service

    host
    
    metric（反应 host 的状态）
    
    fault（在 host 上发生的错误）

步骤
    1. service->service 添加失败的调用边，计算失败率

    2. service->host service 属于 host
    
    3. host -> metric, host -> fault 
"""


def standardize(ts):
    ts = np.array(ts)
    mean = ts.mean()
    std = ts.std()
    if std == 0:
        std = 0.00001
    return (ts - mean) / std


def dwt(ts1, ts2):
    ts1 = standardize(ts1)
    ts2 = standardize(ts2)

    n, m = len(ts1), len(ts2)
    dtw_matrix = np.zeros((n + 1, m + 1))
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(ts1[i - 1] - ts2[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j], dtw_matrix[i, j - 1], dtw_matrix[i - 1, j - 1]
            )

    r = dtw_matrix[n, m]
    if r == 0:
        r = 1
    else:
        r = 1 / r
    return r


def cort(ts1, ts2):
    ts1 = standardize(ts1)
    ts2 = standardize(ts2)

    tmp1 = np.array(np.array(ts1[1:]) - np.array(ts1[:-1]))
    tmp2 = np.array(np.array(ts2[1:]) - np.array(ts2[:-1]))
    sq1 = np.sqrt(np.power(tmp1, 2).sum())
    sq2 = np.sqrt(np.power(tmp2, 2).sum())
    if sq1 == 0:
        sq1 = 0.00001
    if sq2 == 0:
        sq2 = 0.00001
    r = np.dot(tmp1, tmp2) / (sq1 * sq2) + 1
    r /= 2
    if r < 0:
        r = 0
    if r > 1:
        r = 1
    return r


def detect(df, column, cnt=30):
    values = df[column].tolist()
    if len(values) == 0:
        return (False, 0)
    if len(values) <= cnt:
        cnt = len(values) - 1
    if cnt == 0:
        return (False, 0)

    refer_values = values[:cnt]
    detect_values = values[cnt:]

    mean = np.mean(refer_values)
    std = np.std(refer_values)
    if std == 0:
        std = 0.00001

    for value in detect_values:
        if abs(value - mean) > 3 * std:
            # anomaly
            score = (value - mean) / (3 * std)
            return (True, score)

    return (False, 0)


def detect_with_ref(ref_df, df, column):
    refer_values = ref_df[column].tolist()
    detect_values = df[column].tolist()

    mean = np.mean(refer_values)
    std = np.std(refer_values)
    if std == 0:
        std = 0.00001

    for value in detect_values:
        if abs(value - mean) > 3 * std:
            # anomaly
            score = (value - mean) / (3 * std)
            return (True, score)

    return (False, 0)


class Dataset:
    r"""
    the shapes of data ytrues:
    [
        "service1",
        "service2",
        ...
    ]
    the shapes of data yentrys:
    [
        "service1",
        "service2",
        ...
    ]
    the shapes of data like:
    [
        {
            "fault": {
                "host1" : {
                    "fault1": [# time series data #]
                },
                ...
            },
            "hmetric": {
                "host1": {
                    "metric1": [# time series data #]
                }
            },
            "smetric": {
                "service1": [# time series data #]
            },
            "trace": {
                "service1": {
                    "service2": fail_rate(float) // service1 -fail rate-> service2
                    ...
                }
            },
            "deploy": {
                "service1": "host1",
                ...
            },
        },
        {
            ...
        },
        ...
    ]
    """

    def __init__(self, dataset, dataset_dir, itv) -> None:
        self.dataset = dataset
        self.dataset_dir = dataset_dir
        self.itv = itv
        self.yentrys = []
        self.ytrues = []
        self.data = []

    def drain(self, df, column):
        config = TemplateMinerConfig()
        config.load("drain3/drain.ini")
        config.profiling_enabled = True
        template_miner = TemplateMiner(config=config)

        lines = df[column].tolist()
        result_json_list = []
        for line in tqdm(lines, desc=f"draining log"):
            line = str(line).strip()
            result = template_miner.add_log_message(line)
            result_json_list.append(result)

        event_id_list = []
        event_list = []
        for logdict in result_json_list:
            event_id_list.append(logdict["cluster_id"])
            event_list.append(logdict["template_mined"])
        df["event_id"] = event_id_list
        df["event"] = event_list
        return df

    def interpolation(self, ts):
        for i in range(len(ts)):
            if ts[i] == -1:
                left = -1
                for j in range(i, -1, -1):
                    if ts[j] != -1:
                        left = ts[j]
                        break
                right = -1
                for j in range(i + 1, len(ts)):
                    if ts[j] != -1:
                        right = ts[j]
                        break
                if left == -1:
                    left = right
                if right == -1:
                    right = left
                if left == -1:
                    ts[i] = 0
                else:
                    ts[i] = (left + right) / 2
        return ts

    def load(self):
        raise NotImplementedError

    def __getitem__(self, index):
        return (self.ytrues[index], self.yentrys[index], self.data[index])

    def __len__(self):
        return len(self.data)

    def save(self, file):
        with open(file, "w", encoding="utf8") as w:
            json.dump(
                {"yentrys": self.yentrys, "ytrues": self.ytrues, "data": self.data}, w
            )
        logger.info(f"save to {file}")

    def load_json(self, file):
        logger.info(f"load from {file}")
        with open(file, "r", encoding="utf8") as r:
            obj = json.load(r)
        self.yentrys = obj["yentrys"]
        self.ytrues = obj["ytrues"]
        self.data = obj["data"]


class AIops22Dataset(Dataset):
    def __init__(self, dataset, dataset_dir, itv) -> None:
        super().__init__(dataset, dataset_dir, itv)

    def load(self):
        logger.info("loading data")
        print("loading data")
        logger.info("scanning for groundtruth")
        """
        read groundtruth
        """
        gts = []
        for gt_filepath in os.listdir(os.path.join(self.dataset_dir, "groundtruth")):
            gts.append(
                pd.read_json(
                    os.path.join(self.dataset_dir, "groundtruth", gt_filepath),
                    keep_default_dates=False,
                )
            )
        gt_df = pd.concat(gts)
        gt_df = gt_df.query("level != 'node' and not cmdb_id.str.contains('2-0')")
        gt_df.loc[:, "cmdb_id"] = gt_df["cmdb_id"].apply(
            lambda x: re.sub(r"-\d", "", x)
        )
        gt_df = gt_df.drop(columns=["level", "failure_type"])
        """
        read data
        """
        logs = []
        metrics = []
        hmetrics = []
        smetrics = []
        traces = []
        for dirpath, _, filenames in os.walk(self.dataset_dir):
            logger.info(f"scanning for {dirpath}")
            for filename in filenames:
                if filename.find("-log-service") != -1:
                    logs.append(pd.read_csv(os.path.join(dirpath, filename)))
                elif filename.find("kpi_container_") != -1:
                    metrics.append(pd.read_csv(os.path.join(dirpath, filename)))
                elif filename.find("kpi_cloudbed") != -1:
                    hmetrics.append(pd.read_csv(os.path.join(dirpath, filename)))
                elif filename.find("metric_") != -1:
                    smetrics.append(pd.read_csv(os.path.join(dirpath, filename)))
                elif filename.find("trace_jaeger") != -1:
                    traces.append(pd.read_csv(os.path.join(dirpath, filename)))
        log_df = pd.concat(logs)
        metric_df = pd.concat(metrics)
        hmetric_df = pd.concat(hmetrics)
        smetric_df = pd.concat(smetrics)
        trace_df = pd.concat(traces)

        log_df = self.drain(log_df, "value")
        smetric_df["service"] = smetric_df["service"].apply(
            lambda x: re.sub(r"-.*", "", x)
        )
        trace_df["cmdb_id"] = trace_df["cmdb_id"].apply(
            lambda x: re.sub(r"\d?-\d", "", x)
        )
        trace_df["timestamp"] = trace_df["timestamp"].apply(lambda x: int(x / 1000))

        def get_level(msg):
            msg = str(msg)
            if msg.find("severity: info") != -1 or msg.find("severity: INFO") != -1:
                return "info"
            elif (
                msg.find("severity: warning") != -1
                or msg.find("severity: WARNING") != -1
            ):
                return "warning"
            else:
                return "error"

        scheduler = tqdm(total=len(gt_df), desc="processing data")

        for _, case in gt_df.iterrows():
            ed_time = case["timestamp"]
            st_time = ed_time - 30 * 60
            pre_st_time = ed_time - 60 * 60
            tmp_log_df = log_df.query(
                f"{st_time} <= timestamp & {ed_time} >= timestamp"
            )
            tmp_metric_df = metric_df.query(
                f"{st_time} <= timestamp & {ed_time} >= timestamp"
            )
            tmp_hmetric_df = hmetric_df.query(
                f"{st_time} <= timestamp & {ed_time} >= timestamp"
            )
            tmp_smetric_df = smetric_df.query(
                f"{pre_st_time} <= timestamp & {ed_time} >= timestamp"
            )
            tmp_trace_df = trace_df.query(
                f"{pre_st_time} <= timestamp & {ed_time} >= timestamp"
            )

            """
            process deploy
            """
            rels = tmp_metric_df["cmdb_id"].tolist()
            deploy = {}
            for rel in rels:
                rts = rel.split(".")
                host = rts[0]
                service = re.sub(r"\d?-\d", "", rts[1])
                if deploy.get(service, None) is None:
                    deploy[service] = host

            """
            process log
            """
            tmp_log_df["level"] = tmp_log_df["value"].apply(get_level)
            tmp_log_df["host"] = tmp_log_df["cmdb_id"].apply(
                lambda x: deploy[re.sub(r"\d?-\d", "", x)]
            )
            tmp_log_df = tmp_log_df.query("level == 'warning' or level == 'error'")

            fault = {}
            for _host, hgroup in tmp_log_df.groupby(by="host"):
                fault[_host] = {}
                for _fault, fgroup in hgroup.groupby(by="event_id"):
                    ts = []
                    for itr in range(int((ed_time - st_time) / self.itv)):
                        cst_time = st_time + itr * self.itv
                        ced_time = cst_time + self.itv
                        ts.append(
                            len(
                                fgroup.query(
                                    f"{cst_time} <= timestamp & {ced_time} >= timestamp"
                                )
                            )
                        )
                    fault[_host][_fault] = ts

            """
            process hmetric
            """
            hmetric = {}
            for _host, hgroup in tmp_hmetric_df.groupby(by="cmdb_id"):
                hmetric[_host] = {}
                for _kpi, kgroup in hgroup.groupby(by="kpi_name"):
                    ts = []
                    for itr in range(int((ed_time - st_time) / self.itv)):
                        cst_time = st_time + itr * self.itv
                        ced_time = cst_time + self.itv
                        tmp = kgroup.query(
                            f"{cst_time} <= timestamp & {ced_time} >= timestamp"
                        )
                        if len(tmp) == 0:
                            ts.append(-1)
                        else:
                            ts.append(tmp.iloc[0, :]["value"])
                    ts = self.interpolation(ts)
                    hmetric[_host][_kpi] = ts

            """
            process smetric
            """
            smetric = {}
            candidates = []
            for _service, sgroup in tmp_smetric_df.groupby(by="service"):
                ts = []
                for itr in range(int((ed_time - st_time) / self.itv)):
                    cst_time = st_time + itr * self.itv
                    ced_time = cst_time + self.itv
                    tmp = sgroup.query(
                        f"{cst_time} <= timestamp & {ced_time} >= timestamp"
                    )
                    if len(tmp) == 0:
                        ts.append(-1)
                    else:
                        ts.append(tmp.iloc[0, :]["mrt"])
                ts = self.interpolation(ts)
                smetric[_service] = ts
                result = detect(sgroup, "mrt", 30)
                if result[0]:
                    # candidates.append((_service, abs(result[1])))
                    candidates.append(_service)

            """
            process trace
            """
            trace = {}
            tmp_trace_df = tmp_trace_df.rename(columns={"parent_span": "parent_id"})
            # 父子拼接
            meta_df = tmp_trace_df[["parent_id", "cmdb_id"]].rename(
                columns={"parent_id": "span_id", "cmdb_id": "ccmdb_id"}
            )
            tmp_trace_df = pd.merge(tmp_trace_df, meta_df, on="span_id")
            for caller, caller_group in tmp_trace_df.groupby(by="cmdb_id"):
                trace[caller] = {}
                for callee, callee_group in caller_group.groupby(by="ccmdb_id"):
                    fail_cnt = 0
                    total_cnt = 0
                    ref = callee_group.query(
                        f"{pre_st_time} <= timestamp & {st_time} >= timestamp"
                    )
                    for itr in range(int((ed_time - st_time) / self.itv)):
                        cst_time = st_time + itr * self.itv
                        ced_time = cst_time + self.itv
                        tmp = callee_group.query(
                            f"{cst_time} <= timestamp & {ced_time} >= timestamp"
                        )
                        result = detect_with_ref(ref, tmp, "duration")
                        if result[0]:
                            fail_cnt += 1
                        total_cnt += 1
                    if fail_cnt == 0:
                        continue
                    trace[caller][callee] = fail_cnt / total_cnt

            node_set = []
            tmp_trace = {}
            for caller, callee_dict in trace.items():
                node_set.append(caller)
                callee_list = list(callee_dict.keys())
                if not (len(callee_list) == 1 and callee_list[0] == caller):
                    tmp_trace[caller] = callee_list
                    node_set.extend(callee_list)

            node_set = set(node_set)
            # candidates.sort(key=lambda x: x[1], reverse=True)

            new_candidates = []
            for candidate in candidates:
                if candidate in node_set:
                    new_candidates.append(candidate)

            candidates = new_candidates

            if len(candidates) != 0:
                self.yentrys.append(np.random.choice(a=candidates, size=1)[0])
                self.ytrues.append(case["cmdb_id"])
                self.data.append(
                    {
                        "fault": fault,
                        "hmetric": hmetric,
                        "smetric": smetric,
                        "trace": trace,
                        "deploy": deploy,
                    }
                )
            scheduler.update(1)

        logger.info("finish!")
        print("finish!")


class GAIADataset(Dataset):
    def __init__(self, dataset, dataset_dir, itv) -> None:
        super().__init__(dataset, dataset_dir, itv)

    def load(self): ...


class Node:
    def __init__(self, ntype, id) -> None:
        self.ntype = ntype
        self.id = id
        self.outn = []
        self.inn = []

    def __repr__(self) -> str:
        return self.ntype + "_" + self.id

    def __str__(self) -> str:
        return self.ntype + "_" + self.id

    def get_max_score(self, dst_type):
        max_score = 0
        for edge in self.outn:
            if edge.dst.ntype == dst_type and max_score < edge.score:
                max_score = edge.score

        return max_score

    def get_nodes_num(self, dst_type):
        cnt = 0
        for edge in self.outn:
            if edge.dst.ntype == dst_type:
                cnt += 1
        return cnt


class Edge:
    def __init__(self, src: Node, dst: Node, score) -> None:
        self.src = src
        self.dst = dst
        self.score = score
        src.outn.append(self)
        dst.inn.append(self)

    def __repr__(self) -> str:
        return str(self.src) + "->" + str(self.dst)


class TrinityRCL:
    def __init__(self, dataset, config) -> None:
        self.config = config
        self.dataset = dataset
        self.nodes_list = []
        self.bnodes = []
        self.bedges = []
        self.rwr_tms = []
        self.construct()

    def construct(self):
        scheduler = tqdm(total=len(self.dataset), desc="constructing graphs")
        for index, (_, yentry, data) in enumerate(self.dataset):
            self.bnodes.append({})
            self.bedges.append([])

            fault_dict = data["fault"]
            hmetric_dict = data["hmetric"]
            smetric_dict = data["smetric"]
            trace_dict = data["trace"]
            edge_dict = {}
            for src, dst_list in trace_dict.items():
                for dst in dst_list.keys():
                    if edge_dict.get(dst, None) is None:
                        edge_dict[dst] = {}
                    edge_dict[dst][src] = trace_dict[src][dst]
                    if edge_dict.get(src, None) is None:
                        edge_dict[src] = {}
                    if edge_dict[src].get(dst, None) is None:
                        edge_dict[src][dst] = trace_dict[src][dst]

            deploy_dict = data["deploy"]

            # construct graph s<->s
            _open = [yentry]
            _close = []
            while len(_open) != 0:
                _node1 = _open.pop(0)
                _close.append(_node1)
                if edge_dict.get(_node1, None) is None:
                    continue
                next_node_dict = edge_dict[_node1]
                for _node2, _ in next_node_dict.items():
                    if _node2 not in _close:
                        _open.append(_node2)

            for src, dst_list in trace_dict.items():
                if src in _close:
                    for dst in dst_list.keys():
                        if self.bnodes[index].get(src, None) is None:
                            self.bnodes[index][src] = Node("S", src)
                        if self.bnodes[index].get(dst, None) is None:
                            self.bnodes[index][dst] = Node("S", dst)
                        self.bedges[index].append(
                            Edge(
                                self.bnodes[index][src],
                                self.bnodes[index][dst],
                                trace_dict[src][dst],
                            )
                        )

            # construct graph s->h
            for src, dst in deploy_dict.items():
                if self.bnodes[index].get(src, None) is not None:
                    # add s -> h
                    if self.bnodes[index].get(dst, None) is None:
                        self.bnodes[index][dst] = Node("H", dst)
                    self.bedges[index].append(
                        Edge(self.bnodes[index][src], self.bnodes[index][dst], 0)
                    )

            # construct graph h->m
            for src, dst_dict in hmetric_dict.items():
                if self.bnodes[index].get(src, None) is not None:
                    # add h -> m
                    for dst in dst_dict.keys():
                        if self.bnodes[index].get(dst, None) is None:
                            self.bnodes[index][dst] = Node("M", dst)
                        self.bedges[index].append(
                            Edge(self.bnodes[index][src], self.bnodes[index][dst], 0)
                        )

            # construct graph h->f
            for src, dst_dict in fault_dict.items():
                if self.bnodes[index].get(src, None) is not None:
                    # add h -> f
                    for dst in dst_dict.keys():
                        if self.bnodes[index].get(dst, None) is None:
                            self.bnodes[index][dst] = Node("F", dst)
                        self.bedges[index].append(
                            Edge(self.bnodes[index][src], self.bnodes[index][dst], 0)
                        )

            """
            !Calculate the edge score
                DTW: Dynamic Time Warping

                CORT: The First Order Temporal Correlation


                1. host -> metric:
                    using DWT, metric with anomalous node Sa (ts data)

                2. host -> fault:
                    using CORT, fault with anomalous node Sa (ts data)

                3. service -> host:
                    rm = get max correlation from host -> metric

                    rf = get max correlation from host -> fault

                    r = get the ratio which is the number of failures in host j to failures in all host provisioning this service

                    score = (1 - a) * [(1 - b) * rm + b * rf] + a * r
                4. service(i) -> service(j):
                    rh = get max correlation from service(j) -> host

                    r = get failure ratio of invocations from service(i) -> service(j)

                    score = (1 - c) * rh + c * r

                
                > a, b, c are constants
                a = 0.2
                b = 0.5
                c = 0.2
            """
            alpha = self.config["alpha"]
            beta = self.config["beta"]
            gamma = self.config["gamma"]
            for edge in self.bedges[index]:
                if edge.src.ntype == "H" and edge.dst.ntype == "M":
                    edge.score = dwt(
                        smetric_dict[yentry], hmetric_dict[edge.src.id][edge.dst.id]
                    )
            for edge in self.bedges[index]:
                if edge.src.ntype == "H" and edge.dst.ntype == "F":
                    edge.score = cort(
                        smetric_dict[yentry], fault_dict[edge.src.id][edge.dst.id]
                    )
            for edge in self.bedges[index]:
                if edge.src.ntype == "S" and edge.dst.ntype == "H":
                    mrm = edge.dst.get_max_score("M")
                    mrf = edge.dst.get_max_score("F")
                    total_cnt = 0
                    for s_edge in edge.src.outn:
                        total_cnt += s_edge.dst.get_nodes_num("F")
                    if total_cnt == 0:
                        r = 0
                    else:
                        r = edge.dst.get_nodes_num("F") / total_cnt
                    edge.score = (1 - alpha) * (
                        (1 - beta) * mrm + beta * mrf
                    ) + alpha * r
            for edge in self.bedges[index]:
                if edge.src.ntype == "S" and edge.dst.ntype == "S":
                    mrh = edge.dst.get_max_score("H")
                    edge.score = (1 - gamma) * mrh + gamma * edge.score

            """
            RCL
                1. set probability:
                    P(ij) = score
                    
                    e(ji) belong to E, e(ij) not belong to E, P(ji) = ro * score(ij)

                    P(ii) = max(0, max(go to i score) - max(leave i score))

                    > ro is contants
            """

            self.nodes_list.append(list(self.bnodes[index].keys()))

            rho = self.config["rho"]
            r = len(self.bnodes[index].keys())
            tm = np.zeros((r, r))
            for edge in self.bedges[index]:
                i = self.nodes_list[index].index(edge.src.id)
                j = self.nodes_list[index].index(edge.dst.id)
                tm[i][j] = edge.score
            for i in range(r):
                for j in range(r):
                    if tm[j][i] == 0:
                        tm[j][i] = rho * tm[i][j]

            for i in range(r):
                tm[i][i] = max(0, np.max(tm[:, i]) - np.max(tm[:, i]))

            self.rwr_tms.append(tm)
            scheduler.update(1)

    def rwr(self, restart_p):
        logger.info("start rwr")
        ypreds = []
        scheduler = tqdm(total=len(self.rwr_tms), desc="rwring")
        for tm, nodes, nodes_list, entry in zip(
            self.rwr_tms, self.bnodes, self.nodes_list, self.dataset.yentrys
        ):
            """
            2. RWR
                total times = 1,000,000, per 100 stop

            3. rank by visited times
            """
            cnts = {node: 0 for node in nodes_list}
            node2idx = {node: index for index, node in enumerate(nodes_list)}

            total_times = self.config["total_times"]
            per_times = self.config["per_times"]
            p = tm / np.sum(tm, axis=1, keepdims=True)
            pool = Pool(20)
            tasks = []
            for _ in range(int(total_times / per_times)):
                curr = node2idx[entry]
                task = pool.apply_async(
                    _rwr_, (curr, nodes_list, p, per_times, restart_p)
                )
                tasks.append(task)
            pool.close()
            for task in tasks:
                _cnts = task.get()
                for key, val in _cnts.items():
                    cnts[key] += val

            cnts = [(node, cnt) for node, cnt in cnts.items()]
            cnts.sort(key=lambda x: x[1], reverse=True)
            ypred = []
            for node, cnt in cnts:
                if len(ypred) >= 5:
                    break
                if nodes[node].ntype == "S":
                    ypred.append(node)
            while len(ypred) < 5:
                ypred.append(ypred[-1])
            ypreds.append(ypred)
            scheduler.update(1)
        logger.info("finish")
        return ypreds


def _rwr_(entry, nodes_list, p, per_times, restart_p):
    curr = entry
    cnts = {node: 0 for node in nodes_list}
    for _ in range(per_times):
        # count the enter
        cnts[nodes_list[curr]] += 1
        if np.random.rand() < restart_p:
            # restart
            curr = entry
            continue
        # probability
        curr_p = p[curr, :]
        # random choice
        curr = np.random.choice(a=range(len(curr_p)), size=1, p=curr_p)[0]
    return cnts


def get_topk(ytrues, ypreds):
    topk = [0, 0, 0, 0, 0]
    for ytrue, ypred in zip(ytrues, ypreds):
        for index, pred in enumerate(ypred):
            if pred == ytrue:
                for i in range(index, 5):
                    topk[i] += 1
                break
    topk = np.array(topk)
    return topk / len(ytrues)


if __name__ == "__main__":
    logger.info("start new experiment")
    import yaml

    with open("aiops22.yaml", "r", encoding="utf8") as r:
        config = yaml.load(r, yaml.Loader)
    logger.info(f"{str(config)}")

    np.random.seed(config["seed"])

    print(config)
    dataset = AIops22Dataset(config["dataset"], config["dataset_dir"], config["itv"])
    if os.path.exists("aiops22.json"):
        dataset.load_json("aiops22.json")
    else:
        dataset.load()
        dataset.save("aiops22.json")
    rcl = TrinityRCL(dataset, config)
    ypred = rcl.rwr(config["restart_p"])
    topk = get_topk(dataset.ytrues, ypred)
    print(topk)
    logger.info(str(topk))
    logger.info("end experiment")
