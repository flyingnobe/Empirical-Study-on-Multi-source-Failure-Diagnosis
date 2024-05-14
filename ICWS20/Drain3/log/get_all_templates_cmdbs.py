import os
import pandas as pd

def get_all_templates_cmdbs(drain_result_dir, output_file):
    print('[get_all_templates_cmdbs]')
    drain_result = []
    for root, dirs, files in os.walk(drain_result_dir):
        for file in files:
            if root.endswith('drain_result') and 'structured' in file:
                drain_result.append(os.path.join(root, file))

    df = pd.DataFrame(columns=['templates_cmdbs'])
    for file in drain_result:
        print(file)
        t = pd.read_csv(file)
        if t.iloc[0]['timestamp'] == 'timestamp':
            t = t.drop([0])
        tdf = pd.DataFrame((t['EventTemplate'] + '_' + t['cmdb_id']).unique(), columns=['templates_cmdbs'])
        df = df.append(tdf, ignore_index=True)

    ans = pd.DataFrame(df['templates_cmdbs'].unique(), columns=['templates_cmdbs'])
    ans.to_csv(output_file)
    print(f'[get_all_templates_cmdbs done!] output: {output_file}')