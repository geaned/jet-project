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
            dataframe_keys[0]: [os.path.basename(path) for path in paths],
            dataframe_keys[1]: [True]*len(paths),
            dataframe_keys[2]: ['OK']*len(paths),
        },
    )

def set_value_in_dataframe(df: pd.DataFrame, name: str, column: str, value):
    row_idx = df[df['Name'] == name].index[0]
    column_idx = COLUMN_TO_NUMBER[column]
    df.iloc[row_idx, column_idx] = value

def set_negative_entry(df: pd.DataFrame, name: str, reason: str):
    set_value_in_dataframe(df, name, 'Good', False)
    set_value_in_dataframe(df, name, 'Verdict', reason)
