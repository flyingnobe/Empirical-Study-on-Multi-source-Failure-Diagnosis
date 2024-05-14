import os
from log.logparser.logparser import Drain

def drain_work(input_root_dir, file_size=128*1024*1024):
    if not os.path.exists(input_root_dir):
        raise Exception(f'no such root_dir: {input_root_dir}')

    log_format = '<log_id>,<timestamp>,<cmdb_id>,<log_name>,<Content>'
    regex      = [
        r'blk_(|-)[0-9]+' , # block id
        r'(/|)([0-9]+\.){3}[0-9]+(:[0-9]+|)(:|)', # IP
        r'(?<=[^A-Za-z0-9])(\-?\+?\d+)(?=[^A-Za-z0-9])|[0-9]+$', # Numbers
    ]
    st         = 0.5  # Similarity threshold
    depth      = 4  # Depth of all leaf nodes  4

    inputs = os.walk(input_root_dir)

    for root, dirs, files in inputs:
        for file in files:
            if root.endswith('logs') and file.endswith('csv'):
                size = os.path.getsize(os.path.join(root, file))
                path = os.path.join(root, file)
                input_dir = root if size <= file_size else f'mid/{root[5:]}/{file[:-4]}/splits'
                output_dir = f'mid/{root[5:]}/drain_result/'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                exists = set(os.listdir(output_dir))

                parser = Drain.LogParser(log_format,
                    indir=input_dir,
                    outdir=output_dir,
                    depth=depth, st=st, rex=regex)
                if size <= file_size:
                    log_file = file
                    if f'{log_file}_structured.csv' in exists and f'{log_file}_templates.csv' in exists:
                        print(f'[skip exist]{log_file}')
                        continue
                    print(f'[parse]{log_file}')
                    print(f'[parse] input_dir: {input_dir}  output_dir: {output_dir}')
                    parser.parse(log_file)
                    print(f'[parse done]{log_file}')
                else:
                    files = os.listdir(input_dir)
                    for log_file in files:
                        if f'{log_file}_structured.csv' in exists and f'{log_file}_templates.csv' in exists:
                            print(f'[skip exist]{log_file}')
                            continue
                        print(f'[parse]{log_file}')
                        print(f'[parse] input_dir: {input_dir}  output_dir: {output_dir}')
                        parser.parse(log_file)
                        print(f'[parse done]{log_file}')