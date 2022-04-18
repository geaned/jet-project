import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import segmentation_models_pytorch as smp
import albumentations as albu

from scipy.optimize import minimize
from sklearn.cluster import DBSCAN

import Mask2Base64

def optimal_angle(mask, clasterization=False):
    def dispersion(angle):
        proj = np.sin(angle)*x_proj + np.cos(angle)*y_proj
        return (proj**2).mean()-(proj.mean())**2
    
    indexes = np.nonzero(mask)
    
    if clasterization:
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

def detect_text(img):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_PATH = './td_model/_ocr_model.pth'
    model = torch.load(MODEL_PATH, map_location=torch.device(DEVICE))
    
    ENCODER = 'resnet34'
    ENCODER_WEIGHTS = 'imagenet'
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    image = cv2.resize(preprocessing_fn(img), (640, 640))
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    x_tensor = x_tensor.permute(0, 3, 1,2)
    x_tensor = x_tensor.to(torch.float32)

    pr_mask = model.predict(x_tensor)
    pr_mask = (pr_mask > 0.5).squeeze().cpu()
    pr_mask = np.array(pr_mask, dtype=float)
    pr_mask = cv2.resize(pr_mask, (img.shape[0], img.shape[1]))
    return pr_mask

def rotate_to_horizontal(filename):
    path = os.path.join('./crop_results', filename)
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = detect_text(image)
    angle = optimal_angle(mask, clasterization=True)
    
    out_path = os.path.join('./rotation_results', filename)
    cv2.imwrite(out_path, imutils.rotate_bound(image, angle=angle))

rotate_to_horizontal('IMG_1748_ROD_901072579.png')
