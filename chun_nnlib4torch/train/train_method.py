import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import abc
import copy
import time
from torch.utils.tensorboard import SummaryWriter
from evaluate.confusion_matrix import createConfusionMatrix


class TrainMethod(abc.ABC):
    # abstract Class
    def __init__(self, model, dataloader, criterion, optimizer, evaluation = None, epochs = 25):
        self._model = model
        self._dataloader = dataloader
        self._epochs = epochs
        self._optimizer = optimizer
        self._criterion = criterion 
        self._evaluation = evaluation

    @abc.abstractclassmethod
    def fit(self):
        return NotImplemented

