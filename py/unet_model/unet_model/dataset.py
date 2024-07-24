import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


def count_lines_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        return len(lines)


class VocDataSet(Dataset):
    def __init__(self, path='D:\\Code\\pytest\\pythonProject\\data\\VOCdevkit\\VOC2012'):
        with open(path + '\\ImageSets\\Segmentation\\train.txt', 'r', encoding='utf-8') as file:
            self.data_list = file.readlines()
            print(f'total image : {len(self.data_list)}')
            self.path = path
            self.transform = transforms.ToTensor()

    def __getitem__(self, index):
        x = Image.open(self.path + '\\JPEGImages\\' + self.data_list[index][:-1] + '.jpg')
        y = Image.open(self.path + '\\SegmentationClass\\' + self.data_list[index][:-1] + '.png')
        # x = cv2.imread(self.path + '\\JPEGImages\\' + self.data_list[index][:-1] + '.jpg', cv2.IMREAD_COLOR)
        # y = cv2.imread(self.path + '\\SegmentationClass\\' + self.data_list[index][:-1] + '.png', cv2.IMREAD_GRAYSCALE)
        resize = transforms.Resize((320, 480))
        pil_transforms = transforms.PILToTensor()
        return resize(self.transform(x)), torch.squeeze(resize(pil_transforms(y)), dim=0)

    def __len__(self):
        return len(self.data_list)
