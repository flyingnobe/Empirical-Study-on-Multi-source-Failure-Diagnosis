#!/usr/bin/env python
import os
import pandas as pd

def split_file(input_dir, file_size=128*1024*1024):
    if not os.path.exists(input_dir):
        raise Exception(f'no such input_dir: {input_dir}')
    inputs = os.walk(input_dir)
    for root, dirs, files in inputs:
        for file in files:
            mid_dir = f'mid/{root[5:]}/{file[:-4]}/'
            if root.endswith('logs') and file.endswith('csv'):
                size = os.path.getsize(os.path.join(root, file))
                path = os.path.join(root, file)
                if size > file_size:
                    num = int((size + file_size - 1) / file_size)
                    split_file_dir = f'{mid_dir}splits/'
                    print(f'[split] src: {path} dest: {split_file_dir} num: {num}')
                    if not os.path.exists(split_file_dir):
                        os.makedirs(split_file_dir)

                    has_handled = True
                    for i in range(0, num):
                        split_file_name = f'{file[:-4]}_{i}.csv'
                        split_file_path = os.path.join(split_file_dir, split_file_name)
                        if not os.path.exists(split_file_path):
                            has_handled = False
                    if has_handled:
                        print(f'[has split] {file}')
                        continue
                    
                    df = pd.read_csv(path)
                    avg_line_num = int((df.shape[0] + num - 1) / num)
                    for i in range(0, num):
                        split_file = df.iloc[i * avg_line_num + 1 : (i + 1) * avg_line_num + 1]
                        split_file_name = f'{file[:-4]}_{i}.csv'
                        split_file_path = os.path.join(split_file_dir, split_file_name)
                        split_file.to_csv(split_file_path, index=False)
                        print(i, split_file_path)
    