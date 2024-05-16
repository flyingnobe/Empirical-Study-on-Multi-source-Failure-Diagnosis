import json
from datetime import datetime
import pytz
import time
import numpy as np
from multiprocessing import Process
import os
import pandas as pd

aiops22_full_list = os.listdir('/Users/fengxiaoyu/Desktop/PDiagnose/22aiops/metric')
aiops22_metric_list = []
for metric in aiops22_full_list:
    if metric[-4:] == '.csv':
        aiops22_metric_list.append(metric)
        print(metric)

platform_full_list = os.listdir('/Users/fengxiaoyu/Desktop/PDiagnose/平台数据集/metric_1')
platform_metric_list = []
for metric in platform_full_list:
    if metric[-4:] == '.csv':
        platform_metric_list.append(metric)
        print(metric)


tz = pytz.timezone('Asia/Shanghai')
def ts_to_date(timestamp):
    return datetime.fromtimestamp(timestamp//1000, tz).strftime('%Y-%m-%d')

def time_to_ts(ctime):
    try:
        # 先尝试解析包含小时和分钟的格式
        timeArray = time.strptime(ctime, '%Y/%m/%d %H:%M')
    except ValueError:
        try:
            # 如果失败，尝试只有日期的格式
            timeArray = time.strptime(ctime, '%Y/%m/%d')
        except ValueError:
            # 如果仍然失败，记录错误或返回一个默认值
            print(f"Failed to parse date: {ctime}")
            return None
    return int(time.mktime(timeArray))


# 核密度估计 + 加权滑动平均 + 突变检测
from sklearn.neighbors import KernelDensity
from sklearn import preprocessing

class PDedection:
    def __init__(self, config=None):
        self.config = config
        if config is None:
            self.config = {}
            self.config['w1'] = 0.5
            self.config['w2'] = 0.2
            self.config['w3'] = 0.3
            self.config['threshold'] = 0.4
    
    def ked(self, win_list, test_list):
        win_list = np.array(win_list)
        test_list = np.array(test_list)
        res = []
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(win_list[:, np.newaxis])
        res = 1 - np.exp(kde.score_samples(test_list[:, np.newaxis]))[0] # 样本的p值
        return np.array(res)

    def wma(self, win_list, test_list):
        # 待检测点包含在滑动窗口内
        win_list, test_list = list(win_list), list(test_list)
        res = []
        T = len(win_list)
        for i in range(len(test_list)):
            del win_list[0]
            win_list.append(test_list[i])
            wma = 2*sum([t*win_list[t-1] for t in range(1, T+1)])/(T*(T+1))
            res.append(np.abs(test_list[i]-wma))
        return np.array(res)

    def deviation(self, win_list, test_list):
        first_diff = np.diff(np.append(win_list[-1], test_list)) # 一阶差分
        res = []
        for i in range(len(test_list)):
            if first_diff[i] == 0:
                res.append(0)
            elif test_list[i] == 0:
                res.append(0)
            else:
                res.append(np.abs(first_diff[i]/test_list[i]))
        return np.array(res)

    # 需要手动设置阈值
    # config
    def detection(self, win_list, test_list):  
        min_max_scaler = preprocessing.MinMaxScaler()
        X_minMax = min_max_scaler.fit_transform(np.append(win_list, test_list)[:, np.newaxis]).flatten()
        win_list, test_list = X_minMax[: len(win_list)], X_minMax[len(win_list): ]
        scores = self.config['w1']*self.ked(
            win_list, test_list)+self.config['w2']*self.wma(
            win_list, test_list)+self.config['w3']*self.deviation(
            win_list, test_list)
        alarms = [i for i in range(len(scores)) if scores[i] > self.config['threshold']]
        result = {}
        result['alarm'] = alarms
        result['score'] = scores
        return result

# 指标异常信息提取
# 检测时间范围：故障时间段，
# 返回结果：异常KPI队列，包含全部的点 （时间戳，微服务，指标名称，异常分数）
class MetricAnomaly(Process):
    def __init__(self, cases, metric_path, data_dir, pid, load, dataset='gaia', config=None):
        super().__init__()
        self.dataset = dataset
        self.id = pid
        self.cases = cases.iloc[pid*load: (pid+1)*load]
        self.periods = ['2021-07-01_2021-07-15', '2021-07-15_2021-07-31']
        det_config = {}
        if self.dataset == 'gaia':
            self.metrics = self.get_all_metric_names(pd.read_csv(metric_path))
            det_config['w1'] = 0.5
            det_config['w2'] = 0.2
            det_config['w3'] = 0.3
            det_config['threshold'] = 0.4
        elif self.dataset == '21aiops':
            self.metrics = chosen_metrics
            det_config['w1'] = 0.7
            det_config['w2'] = 0.2
            det_config['w3'] = 0.1
            det_config['threshold'] = 1
        elif self.dataset == '22aiops':
            self.metrics = aiops22_metric_list
            det_config['w1'] = 0.6
            det_config['w2'] = 0.2
            det_config['w3'] = 0.5
            det_config['threshold'] = 0.4
        elif self.dataset == 'platform':
            self.metrics = platform_metric_list
            det_config['w1'] = 0.6
            det_config['w2'] = 0.2
            det_config['w3'] = 0.5
            det_config['threshold'] = 0.4
        else:
            raise Exception('Unknow Dataset!')
        self.data_dir = data_dir
        self.detector = PDedection(det_config)
        if config is None:
            config = {}
            config['minute'] = 60000
            config['MIN_TRAIN_LENGTH'] = 5
            config['MIN_TEST_LENGTH'] = 1
        self.config = config
        self.res = dict(zip(list(self.cases.index), [[] for _ in range(len(self.cases))]))
        self.time_used = 0
    
    # 读取指标文件
    def read(self, metric):
        data = pd.read_csv(os.path.join(self.data_dir, metric))
        data.index = [ts_to_date(ts) for ts in data['timestamp']]
        return data
    
    # 获取所有的指标信息，筛去zookeeper、system、redis
    def get_all_metric_names(self, metrics_info):
        metric_names = []
        for index, row in metrics_info.iterrows():
            if row['name'].split('_')[0] in ['system', 'zookeeper', 'redis']: # 这三类略过
                continue
            for period in self.periods:
                metric_names.append('_'.join([row['name'], period+'.csv']))
        return metric_names
    
    def get_metric_events(self):
        print(self.detector.config)
        # 减少文件读取次数
        for metric in self.metrics:
            if not os.path.exists(f'{self.data_dir}/{metric}'):
                continue
            metric_data = self.read(metric)
            start_time = time.time()
            for case_id, case in self.cases.iterrows():
                # 异常开始和故障结束后两分钟
                if self.dataset == 'platform':
                    start_ts = int(case['st_time'])
                    end_ts = int(case['ed_time'])+2*self.config['minute']
                else:
                    start_ts = time_to_ts(case['start'])
                    end_ts = time_to_ts(case['end'])+2*self.config['minute']
                win_start_ts = start_ts - 30*self.config['minute']
                win_data = metric_data[(metric_data['timestamp']>=win_start_ts)&
                                       (metric_data['timestamp']<start_ts)]
                test_data = metric_data[(metric_data['timestamp']>=start_ts)&
                                       (metric_data['timestamp']<end_ts)]
                if len(test_data) < self.config[
                    'MIN_TEST_LENGTH'] or len(win_data) < self.config[
                    'MIN_TRAIN_LENGTH']:
                    continue
                res = self.detector.detection(win_data['value'].values, test_data['value'].values)
                test_ts = test_data['timestamp'].values
                if self.dataset == 'gaia':
                    for i in range(len(res['alarm'])):
                        # （时间戳，微服务，指标名称，异常分数）
                        metric_splits = metric.split('_')
                        service = metric_splits[0]
                        name = '_'.join(metric_splits[2: -2])
                        self.res[case_id].append((int(test_ts[res['alarm'][i]]), service, name, 
                                                  res['score'][res['alarm'][i]]))
                elif self.dataset == '21aiops':
                    for i in range(len(res['alarm'])):
                        # （时间戳，微服务，指标名称，异常分数）
                        metric_splits = metric.split('+')
                        service = metric_splits[0]
                        name = metric_splits[1].split('.csv')[0]
                        self.res[case_id].append((int(test_ts[res['alarm'][i]]), service, name, 
                                                  res['score'][res['alarm'][i]]))
                elif self.dataset == '22aiops':
                    for i in range(len(res['alarm'])):
                        # （时间戳，微服务，指标名称，异常分数）
                        metric_splits = metric.split('+')
                        service = metric_splits[0]
                        name = metric_splits[1][:-4]
                        self.res[case_id].append((int(test_ts[res['alarm'][i]]), service, name, 
                                                  res['score'][res['alarm'][i]]))
                elif self.dataset == 'platform':
                    for i in range(len(res['alarm'])):
                        # （时间戳，微服务，指标名称，异常分数）
                        metric_splits = metric.split('+')
                        service = metric_splits[0]
                        name = metric_splits[1][:-4]
                        self.res[case_id].append((int(test_ts[res['alarm'][i]]), service, name, 
                                                  res['score'][res['alarm'][i]]))

            end_time = time.time()
            self.time_used += (end_time - start_time)
    
    def save_res(self, savepath):                   
        with open(savepath, 'w') as f:
            json.dump(self.res, f)
        print(f'{self.id} Time used: ', self.time_used)
        print('Save successfully!')
    
    def run(self):
        self.get_metric_events()
        self.save_res(f'{self.dataset}/metric/{self.dataset}_metric_{self.id}.json')
        with open(f'{self.dataset}/metric/time_used_{self.id}', 'w') as f:
            f.write(f'{self.time_used}')
    