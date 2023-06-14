import os
import numpy as np
import torch
import h5py
from torch.utils.data import DataLoader, Dataset


class matrixDataset2(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y = self.y[index]
        if self.transform is not None:
            X = self.transform(X)
            Y = self.transform(Y)
        return X, Y

class matrixDataset_class(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X
#
class matrixDataset_end_to_end(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.LongTensor(y)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class matrixDataset_DRR(Dataset):
    def __init__(self, x, y1, y2, transform=None):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y1 = torch.tensor(y1, dtype=torch.float)
        self.y2 = torch.tensor(y2, dtype=torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y1 = self.y1[index]
        Y2 = self.y2[index]
        return X, Y1, Y2

class matrixDataset_DRR_IRM(Dataset):
    def __init__(self, x, y1, y2, y3, transform=None):
        self.x = torch.tensor(x,   dtype = torch.float)
        self.y1 = torch.tensor(y1, dtype = torch.float)
        self.y2 = torch.tensor(y2, dtype = torch.float)
        self.y3 = torch.tensor(y3, dtype = torch.float)
        self.transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        X = self.x[index]
        Y1 = self.y1[index]
        Y2 = self.y2[index]
        Y3 = self.y3[index]
        return X, Y1, Y2, Y3

class matrixDataset_TWO_GET_DRR(Dataset):
    def __init__(self, x, y1, transform=None):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y1 = torch.tensor(y1, dtype=torch.float)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        Y1 = self.y1[index]
        return X, Y1
# batch_size =16
# train_set = matrixDataset(train_x, train_y)
# test_set = matrixDataset(test_x, test_y)
# train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
