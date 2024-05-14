import os
import pandas as pd

def concat_template(input_dir, output_file):
    print(f'[concat_template] input_dir:{input_dir} output_file:{output_file}')
    inputs = os.walk(input_dir)

    if os.path.exists(output_file):
        os.remove(output_file)

    for root, dirs, files in inputs:
        for file in files:
            if 'templates' in file:
                src = os.path.join(root, file)
                df = pd.read_csv(src, header=None)
                df.to_csv(output_file, mode='a', index=False, header=False)
                print(f'[concat_template process] {src}')

    df = pd.read_csv(output_file)
    df = df[(df['EventTemplate'] != 'value') & (df['EventId'] != 'EventId')]
    df[["Occurrences"]] = df[["Occurrences"]].astype(int)
    grouped_res = df.groupby(['EventId', 'EventTemplate'])['Occurrences'].sum().reset_index(name="Occurrences")
    grouped_res.to_csv(output_file, index=False)
    print(f'[concat_template done] the result is stored in {output_file}')