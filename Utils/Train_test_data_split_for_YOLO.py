import os
import random
from sklearn.model_selection import train_test_split

import shutil


def copy_images(prefixes: str, is_test: bool):
    folder_dest = 'valid' if is_test else 'train'
    for prefix in prefixes:
        shutil.copy(os.path.join(DATA_TO_SPLIT, IMG_FOLDER, prefix+'.png'), os.path.join(DATA_TO_POPULATE, IMG_FOLDER, folder_dest, prefix+'.png'))

def copy_labels(prefixes: str, is_test: bool):
    folder_dest = 'valid' if is_test else 'train'
    for prefix in prefixes:
        shutil.copy(os.path.join(DATA_TO_SPLIT, LABEL_FOLDER, prefix+'.txt'), os.path.join(DATA_TO_POPULATE, LABEL_FOLDER, folder_dest, prefix+'.txt'))


random.seed(42)

TEST_SIZE = 0.2

DATA_TO_SPLIT = os.path.join(os.path.dirname(os.path.realpath("__file__")), os.pardir, 'data_all')
DATA_TO_POPULATE = os.path.join(os.path.dirname(os.path.realpath("__file__")), os.pardir, 'data')

# these two should be created in DATA_TO_POPULATE with 'train' and 'valid' subfolders
IMG_FOLDER = 'img'
LABEL_FOLDER = 'labels'

found_prefixes = list(name.replace('.png', '') for name in os.listdir(os.path.join(DATA_TO_SPLIT, IMG_FOLDER)))
random.shuffle(found_prefixes)

test_prefixes = found_prefixes[:int(TEST_SIZE * len(found_prefixes))]
train_prefixes = found_prefixes[int(TEST_SIZE * len(found_prefixes)):]

copy_images(train_prefixes, is_test=False)
copy_images(test_prefixes, is_test=True)
copy_labels(train_prefixes, is_test=False)
copy_labels(test_prefixes, is_test=True)
