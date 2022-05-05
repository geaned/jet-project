import pandas as pd
import os
import shutil

SOURCE_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'crops')
DEST_FOLDER = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'filtered')
FILTER_REASON = 'Text area not found'
DATAFRAME_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, 'crop_quality.csv')

quality_df = pd.read_csv(DATAFRAME_PATH)
filtered_quality_df = quality_df[quality_df['Verdict'] == FILTER_REASON]
filtered_crop_names = filtered_quality_df['Name']

try:
    os.makedirs(DEST_FOLDER)
except:
    pass

for crop_name in filtered_crop_names:
    shutil.copy(os.path.join(SOURCE_FOLDER, crop_name), os.path.join(DEST_FOLDER, crop_name))
