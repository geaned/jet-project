import os
import argparse
from typing import List, Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model


class Options:
    def __init__(self, saved_model_path: str):
        self.num_gpu = torch.cuda.device_count()
        self.workers = 4
        self.saved_model = saved_model_path
        self.batch_max_length = 25 # maximum-label-length
        self.batch_size = 192
        self.imgH = 32
        self.imgW = 100
        self.character = '0123456789abcdefghijklmnopqrstuvwxyz'
        self.num_class = 37
        self.sensitive = False
        self.PAD = False
        self.Transformation = 'None'
        self.FeatureExtraction = 'ResNet'
        self.SequenceModeling = 'None'
        self.Prediction = 'CTC'
        self.rgb = False
        self.input_channel = 1 # the number of input channel of Feature extractor
        self.output_channel = 512 # the number of output channel of Feature extractor


def get_rosetta_predictions_and_confs(saved_model_path: str, image_folder: str) -> List[Tuple[str, str, float]]:
    predictions_array = []

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cudnn.benchmark = True
    cudnn.deterministic = True

    # load model
    opt = Options(saved_model_path)
    model = Model(opt)
    model = torch.nn.DataParallel(model).to(DEVICE)
    model.load_state_dict(torch.load(opt.saved_model, map_location=DEVICE))

    # prepare data
    align_collate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    data = RawDataset(root=image_folder, opt=opt)  # use RawDataset
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=opt.batch_size, shuffle=False,
        num_workers=int(opt.workers), collate_fn=align_collate, pin_memory=True,
    )

    converter = CTCLabelConverter(opt.character)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in data_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(DEVICE)
            # For max length prediction
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(DEVICE)

            preds = model(image, text_for_pred)

            # Select max probability (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds[:, :, :10].max(2) # because we want to predict only digits
            preds_str = converter.decode(preds_index, preds_size)

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]
                predictions_array.append((img_name, pred, confidence_score))

    return predictions_array

def create_folder_if_necessary(folder_name: str):
    try:
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created...')
    except FileExistsError:
        print(f'Folder "{folder_name}" exists...')

def write_strings_to_text_files(destination_folder: str, rosetta_predictions: List[Tuple[str, str, float]]):
    create_folder_if_necessary(destination_folder)

    print('Writing found strings to text files...')
    for img_name, predicted_number, confidence in rosetta_predictions:
        whole_img_name = os.path.basename(img_name).replace('.png', '')[:-2]
        label_name = os.path.join(destination_folder, whole_img_name + '.txt')
        with open(label_name, 'a') as text_results:
            text_results.write(f'{predicted_number}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--saved_model', required=True, help='path to recognition saved model')
    parser.add_argument('--string_result_folder', required=True, help='path to result folder with predictions of serial numbers')
    opt = parser.parse_args()

    rosetta_preds = get_rosetta_predictions_and_confs(opt.saved_model, opt.image_folder)
    write_strings_to_text_files(opt.string_result_folder, rosetta_preds)
