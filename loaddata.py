from torch.utils.data import Dataset
import numpy as np
import torch
from skimage import io, transform
import os

from PIL import Image
from skimage import io, transform
import os


class kittidata(Dataset):
    def __init__(self, data_path, label_path, split_ratio, transform=None):
        self.class_name = ('sky', 'building', 'road', 'sidewalk', 'fence',
                           'vegetation', 'pole', 'car', 'sign', 'pedestrian'
                                                                'cyclist', 'ignore')
        self.class_color = ((128, 128, 128), (128, 0, 0), (128, 64, 128), (0, 0, 192), (64, 64, 128),
                            (128, 128, 0), (192, 192, 128), (64, 0, 128), (192, 128, 128), (64, 64, 0),
                            (0, 128, 192), (0, 0, 0))
        self.class_n = 12

        self.data_path = data_path
        self.label_path = label_path
        self.phase = 'train'
        self.transform = transform

        self.data = os.listdir(data_path)  # only store path rather than image, to save space
        self.label = os.listdir(label_path)
        self.split_ratio = split_ratio

        self.total_size = len(self.data)
        self.train_size = int(self.total_size * split_ratio)
        self.val_size = self.total_size - self.train_size

        self.train = self.data[:self.train_size]
        self.val = self.data[self.val_size:]
        self.train_label = self.label[:self.train_size]
        self.val_label = self.label[self.val_size:]

    def setphase(self, phase):
        self.phase = phase
        return self

    def __len__(self):
        if self.phase == 'train':
            return self.train_size
        else:
            return self.val_size

    def __getitem__(self, idx):
        if self.phase == 'train':
            img = io.imread(os.path.join(self.data_path, self.train[idx]))
            label = io.imread(os.path.join(self.label_path, self.train_label[idx]))
        else:
            img = io.imread(os.path.join(self.data_path, self.val[idx]))
            label = io.imread(os.path.join(self.label_path, self.val_label[idx]))

        h, w, c = label.shape
        h = int(h // 32 * 32)
        w = int(w // 32 * 32)
        img = transform.resize(img, (h, w), mode='constant', preserve_range=True).astype('uint8')
        label = transform.resize(label, (h, w), mode='constant', preserve_range=True).astype('uint8')

        if self.transform:
            img = self.transform(img)

        num_label = torch.zeros(h, w).view(-1).short()
        for i, v in enumerate(label.reshape(-1, c)):
            try:
                num_label[i] = self.class_color.index(tuple(v[:3]))
            except:  # some pixel values not follow the defined labels above.
                # print(tuple(v[:3])) # too much inaccuracy, some due to resize, the rest come from the image reader
                num_label[i] = 0  # it is not good yet
        num_label = num_label.view(h, w)
        target = torch.zeros(self.class_n, h, w)
        for c in range(self.class_n):
            target[c, num_label == c] = 1

        return img, target, num_label

    def visualize(self, label):
        h, w = label.shape
        temp_label = np.zeros((h, w, 3), dtype='uint8')
        for i in range(h):  # how to write more elegantly
            for j in range(w):
                temp_label[i, j] = self.class_color[int(label[i, j])]
        return temp_label