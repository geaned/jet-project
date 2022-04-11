import json, os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import Mask2Base64

def insert_matrix(arr, x_begin, y_begin, new_shape):
    answer = np.copy(arr)
    if y_begin > 0:
        answer = np.vstack((np.zeros((y_begin-1,answer.shape[1])), answer))
    if x_begin > 0:
        answer = np.hstack((np.zeros((answer.shape[0],x_begin-1)), answer))
    if answer.shape[0] < new_shape[0]:
        answer = np.vstack((answer, np.zeros((new_shape[0]-answer.shape[0],answer.shape[1]))))
    if answer.shape[1] < new_shape[1]:
        answer = np.hstack((answer, np.zeros((answer.shape[0], new_shape[1]-answer.shape[1]))))
    return answer

def make_masks(data_dir, res_folder, ann_folder='ann', img_folder='img', echo=False):
    ann_files = os.listdir(os.path.join(data_dir, ann_folder))
    for fname in ann_files:
        if echo:
            print("\n", fname)
        with open(os.path.join(data_dir, ann_folder, fname), 'r') as json_file:
            img_name = fname.split('.')[0]
            image = cv2.imread(os.path.join(data_dir, img_folder, img_name + '.png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            data = json.load(json_file)
            if data['objects']:
                result = np.zeros((image.shape[0], image.shape[1], len(data['objects'])))
                for n, subject in enumerate(data['objects']):
                    if echo:
                        print(subject['id'])
                    mask = np.array(Mask2Base64.base64_2_mask(subject['bitmap']['data']),dtype=int)
                    x_min = subject['bitmap']['origin'][0]
                    y_min = subject['bitmap']['origin'][1]
                    mask = insert_matrix(mask, x_min, y_min, image.shape)
                    result[:,:,n] = mask
                    
                file_name = os.path.join(res_folder, img_name + '_mask' + '.png')
                cv2.imwrite(file_name, result)

def crop_blocks(data_dir, res_folder, ann_folder='ann', img_folder='img', echo=False):
    ann_files = os.listdir(os.path.join(data_dir, ann_folder))
    for fname in ann_files:
        if echo:
            print("\n", fname)
        img_name = fname.split('.')[0]
        image = cv2.imread(os.path.join(data_dir, img_folder, img_name + '.png'))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with open(os.path.join(data_dir, ann_folder, fname), 'r') as json_file:
            data = json.load(json_file)
            if data['objects']:
                for rod in data['objects']:
                    if echo:
                        print(rod['id'])
                    mask = np.array(Mask2Base64.base64_2_mask(rod['bitmap']['data']),dtype=int)
                    x_min = rod['bitmap']['origin'][0]
                    y_min = rod['bitmap']['origin'][1]
                    x_max = x_min + mask.shape[1]
                    y_max = y_min + mask.shape[0]
                    
                    cropped_image = image[y_min:y_max,x_min:x_max]
                    file_name = os.path.join(res_folder, img_name + '_ROD_' + str(rod['id']) + '.png')
                    cv2.imwrite(file_name, cropped_image)
