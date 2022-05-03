from typing import List
import pandas as pd
import os

COLUMN_TO_NUMBER = {
    'Name': 0,
    'Good': 1,
    'Verdict': 2
}

IMAGE_REASON_ORDER = [
    'OK',
    'Image is too bright',
    'Image is too blurry',
    'Image has no rods',
    'Image has too many rods',
    'The rods are on the image periphery',
]

CROP_REASON_ORDER = [
    'OK',
    'Crop image is too bright',
    'Crop image is too blurry',
    'Crop area is too thin',
    'Crop square is too small',
    'Text area not found',
    'Digits not found',
]

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

def get_full_stats_from_dataframe(df: pd.DataFrame, type: str):
    if type == 'image':
        verdict_order = IMAGE_REASON_ORDER
    elif type == 'crop':
        verdict_order = CROP_REASON_ORDER
    else:
        return

    print(f'{"Total":<38}{len(df.index):>5}')
    for verdict in verdict_order:
        print(f'{verdict:<38}{len(df[df["Verdict"] == verdict]):>5}')
