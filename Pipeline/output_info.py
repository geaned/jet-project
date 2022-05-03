from typing import List
import pandas as pd
import os

COLUMN_TO_NUMBER = {
    'Name': 0,
    'Good': 1,
    'Verdict': 2
}

def make_base_dataframe_for_paths(paths: List[str]) -> pd.DataFrame:
    dataframe_keys = list(COLUMN_TO_NUMBER.keys())

    return pd.DataFrame.from_dict(
        {
            'Name': [os.path.basename(path) for path in paths],
            'Good': [True]*len(paths),
            'Verdict': ['OK']*len(paths),
        },
    )

def set_value_in_dataframe(df: pd.DataFrame, name: str, column: str, value):
    row_idx = df[df['Name'] == name].index[0]
    column_idx = COLUMN_TO_NUMBER[column]
    df.iloc[row_idx, column_idx] = value

def set_negative_entry(df: pd.DataFrame, name: str, reason: str):
    set_value_in_dataframe(df, name, 'Good', False)
    set_value_in_dataframe(df, name, 'Verdict', reason)

def write_dataframe_sorted_by_name(df: pd.DataFrame, file_path: str):
    df.sort_values(by=['Name']).to_csv(file_path, index=False)

def get_full_stats_from_dataframe(df: pd.DataFrame):
    print(f'{"Total":<26}{len(df.index):>5}')
    
    verdict_counts = []
    for verdict in df['Verdict'].unique():
        verdict_counts.append((verdict, len(df[df['Verdict'] == verdict])))
    
    for verdict, count in sorted(verdict_counts, key=lambda vc: vc[1], reverse=True):
        print(f'{verdict:<26}{count:>5}')
