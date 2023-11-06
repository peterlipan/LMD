import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ISICDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(ISICDataset, self).__init__()
        file = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.images = file['image'].values
        self.labels = np.argmax(file.iloc[:, 1:].values.astype(int), 1)
        self.transform = transform
        self.n_class = len(np.unique(self.labels))
        self.class_names = file.columns[1:]

        print('Total # images:{}, labels:{}, number of classes: {}'.format(len(self.images),len(self.labels), self.n_class))

    def __getitem__(self, index):
        try:
            image_name = os.path.join(self.root_dir, self.images[index]+'.jpg')
            image = Image.open(image_name).convert('RGB')
        # the file extension of ISIC Archive dataset is JPG
        except FileNotFoundError:
            image_name = os.path.join(self.root_dir, self.images[index]+'.JPG')
            image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images)

    def get_labels(self):
        return self.labels


class KvasirDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(KvasirDataset, self).__init__()
        self.csv = pd.read_csv(csv_file)

        self.root_dir = root_dir
        self.labels = self.csv['labels'].values
        self.transform = transform
        self.n_class = len(np.unique(self.labels))
        self.class_names = self.csv.groupby('Finding').max().sort_values('labels', ascending=True).index.tolist()

        print('Total # images:{}, labels:{}, number of classes: {}'.format(self.csv.shape[0],
                                                                       len(self.labels), self.n_class))

    def __getitem__(self, index):
        row = self.csv.iloc[index, :]
        organ = row['Organ']
        finding = row['Finding']
        classification = row['Classification']
        filename = row['Video file']
        organ2folder = {'Lower GI': 'lower-gi-tract', 'Upper GI': 'upper-gi-tract'}
        if organ in organ2folder:
            organ = organ2folder[organ]
        img_path = os.path.join(self.root_dir, organ, classification, finding, filename+'.jpg')
        image = Image.open(img_path).convert('RGB')

        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.labels)
    
    def get_labels(self):
        return self.labels
