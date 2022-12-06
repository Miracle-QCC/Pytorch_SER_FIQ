import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import cv2
from torchvision import transforms

data_transforms = {
    'train':
        transforms.Compose([
        transforms.RandomRotation(45),#随机旋转，-45到45度之间随机选
        transforms.RandomHorizontalFlip(p=0.5),#随机水平翻转 选择一个概率概率
        transforms.RandomVerticalFlip(p=0.5),#随机垂直翻转
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5019, 0.5019, 0.5019])#均值，标准差
    ]),
    'valid':
        transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5019, 0.5019, 0.5019])  #均值，标准差
     ]),
}

class Face_Quality_Dataset(Dataset):

    def __init__(self, root_dir, ann_file, transform=None):
        super(Face_Quality_Dataset, self).__init__()
        self.ann_file = ann_file
        self.root_dir = root_dir
        self.img_label = self.load_annotations()
        self.img = [os.path.join(self.root_dir, img) for img in list(self.img_label.keys())]
        self.label = [label for label in list(self.img_label.values())]
        self.transform = transform

    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        image = Image.open(self.img[idx])
        label = self.label[idx]
        if self.transform:
            image = self.transform(image)
        weight = torch.from_numpy(np.array(label))
        return image, weight

    def load_annotations(self):
        data_infos = {}
        with open(self.ann_file) as f:
            samples = [x.strip().split(' ') for x in f.readlines()]
            for filename, gt_label in samples:
                data_infos[filename] = np.array(gt_label, dtype=np.int64)
        return data_infos

if __name__ == '__main__':
    pass