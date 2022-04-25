import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import segmentation_models_pytorch as smp

import gdown

from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

def optimal_angle(mask, clusterization=False):
    def dispersion(angle):
        proj = np.sin(angle)*x_proj + np.cos(angle)*y_proj
        return (proj**2).mean()-(proj.mean())**2
    
    change_size = int(np.sqrt(mask.sum()/10000))
    if change_size > 1:
        mask = mask[::change_size,::change_size]
    
    indexes = np.nonzero(mask)
    
    if clusterization:
        indexes = np.array(indexes).T
        clustered = DBSCAN(eps=3, min_samples=2).fit(indexes)
        classes = np.array(clustered.labels_)
        most_frequent_class = np.argmax(np.bincount(classes))
        indexes = indexes[np.where(classes==most_frequent_class)]
        indexes = indexes.T
        indexes = (list(indexes[0]), list(indexes[1]))
    
    x_proj = np.repeat([np.arange(mask.shape[1])], mask.shape[0], axis=0)[indexes]
    y_proj = np.repeat([np.arange(mask.shape[0])], mask.shape[1], axis=0).T[indexes]
    answer = minimize(dispersion, x0=[1], bounds=[[-10,10]]).x[0]
    answer = answer/np.pi*180
    answer += 720
    answer %= 180
    if answer > 90:
        answer -= 180
    return answer

def detect_text(images_by_file_path, model_path):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        model = torch.load(model_path, map_location=torch.device(DEVICE))
        print('Found text segmentation weights!')
    except FileNotFoundError:
        print('Could not find text segmentation weights, downloading...')
        gdown.download('https://drive.google.com/uc?id=1ziaSS_upk7VHgS6jE1QnN_USSIUzCZNK', model_path, quiet=False)
        model = torch.load(model_path, map_location=torch.device(DEVICE))

    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    # get filenames and output progress
    masks_by_file_path = {}
    for file_path in images_by_file_path:
        print(f'Evaluating text mask on {file_path}...')
        img = images_by_file_path[file_path]
        image = cv2.resize(preprocessing_fn(img), (640, 640))
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        x_tensor = x_tensor.permute(0, 3, 1,2)
        x_tensor = x_tensor.to(torch.float32)

        pr_mask = model.predict(x_tensor)
        pr_mask = (pr_mask > 0.5).squeeze().cpu()
        pr_mask = np.array(pr_mask, dtype=float)
        pr_mask = cv2.resize(pr_mask, (img.shape[0], img.shape[1]))
        masks_by_file_path[file_path] = pr_mask
    return masks_by_file_path

def rotate_to_horizontal(file_paths, model_path):
    print('Parsing crops...')
    images_by_file_path = {}
    for file_path in file_paths:
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images_by_file_path[file_path] = image

    print('Looking for text masks on crops...')
    masks_by_file_path = detect_text(images_by_file_path, model_path)
    
    for file_path in file_paths:
        file_name = file_path.split('/')[-1]
        print(f'Rotating and saving crop to {file_path}... ', end='')

        if masks_by_file_path[file_path].sum() == 0:
            print(f'Failed! Text area not found')
            continue
        print()

        angle = optimal_angle(masks_by_file_path[file_path], clusterization=True)
        cv2.imwrite(
            os.path.join('rotation_results', file_name),
            imutils.rotate_bound(images_by_file_path[file_path], angle=angle),
        )
