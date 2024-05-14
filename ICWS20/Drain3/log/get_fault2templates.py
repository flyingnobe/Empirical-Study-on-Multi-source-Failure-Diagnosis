import time
import os
import pandas as pd

def get_fault2templates(fault_file, mid_dir):
    print(f'[get_fault2templates] fault_file: {fault_file}')
    faults = pd.read_excel(fault_file)
    
    inputs = os.walk(mid_dir)
    paths = []
    for root, dirs, files in inputs:
        for file in files:
            if root.endswith('drain_result') and 'structured' in file:
                path = os.path.join(root, file)
                paths.append(path)

    for path in paths:
        cur_csv = pd.read_csv(path)
        
        if cur_csv.iloc[0]['timestamp'] == 'timestamp':
            cur_csv = cur_csv.drop([0])
        cur_csv[['timestamp']] = cur_csv[['timestamp']].astype(int)
        cur_csv.sort_values('timestamp',inplace=True)
        
        if cur_csv.shape[0] < 1:
            continue
    
        csv_st_time = int(cur_csv.iloc[0]['timestamp'])
        csv_ed_time = int(cur_csv.iloc[-1]['timestamp'])
        
        for idx, row in faults.iterrows():
            _id = row['id']
            execute_time_str = str(row['执行时间']).split('.')[0]
            execute_time = int(time.mktime(time.strptime(execute_time_str, "%Y-%m-%d %H:%M:%S")))
            duration = int(row['持续时间（s）'])
            st_time = execute_time - duration
            ed_time = execute_time + duration * 2
            
            if ed_time < csv_st_time or st_time > csv_ed_time:
                continue
            
            fault_file_dir = os.path.dirname(fault_file)
            fault2template_file = f'{mid_dir}fault2template/fault_{execute_time}_{_id}.csv'
            print(fault2template_file)
            if not os.path.exists(f'{mid_dir}fault2template/'):
                os.makedirs(f'{mid_dir}fault2template/')
            selected_rows = cur_csv[(st_time <= cur_csv['timestamp']) & (cur_csv['timestamp'] <= ed_time)]
            selected_rows.to_csv(fault2template_file, mode='a', index=False, header=False)
            print(f'[get_fault2templates] execute_time: {execute_time} \
            st_time: {st_time} ed_time: {ed_time} fault2template_file: {fault2template_file}')
    print(f'[get_fault2templates done] fault_file: {fault_file}')