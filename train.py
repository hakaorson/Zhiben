from sklearn import metrics
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from Preprocess import preprocess
from Model import basemodel
import torch
from torch import nn

if __name__ == '__main__':
    dataset = preprocess.main('Data/bbbp_test.csv')
    basemodel.train(dataset.train_data)
    pass
