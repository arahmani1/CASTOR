import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pandas import Index, RangeIndex
from collections.abc import Iterable


def identity(x):
    return x

def leaky_relu(arr):
    alpha = 0.1
    return np.maximum(alpha*arr, arr)

def relu(arr):
    return np.maximum(0, arr)


def train_pi(num_epochs, model, data, gamma):
            model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            soft = nn.Softmax()
            for _ in range(num_epochs):
                        y_pre = model(data)  
                        loss = criterion(y_pre, gamma)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()   
            return soft(y_pre),loss.item(),model

def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss

class Tensor(np.ndarray):

    def __new__(cls, object=None, index=None, columns=None):

        if object is None:
            raise TypeError("Tensor() missing required argument 'object' (pos 0)")
        elif isinstance(object, list):
            object = np.array(object)
        elif isinstance(object, pd.DataFrame):
            index = object.index
            columns = object.columns
            object = object.values
        elif isinstance(object, (np.ndarray, cls)):
            pass
        else:
            raise TypeError(
                "Type of the required argument 'object' must be array-like."
            )
        if index is None:
            index = range(object.shape[0])
        if columns is None:
            columns = range(object.shape[1])
        obj = np.asarray(object).view(cls)
        obj.index = index
        obj.columns = columns

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        if self.ndim == 0: return
        elif self.ndim == 1:
            self.columns = RangeIndex(0, 1, step=1, dtype=int)
        else:
            self.columns = RangeIndex(0, self.shape[1], step=1, dtype=int)
        self.index = RangeIndex(0, self.shape[0], step=1, dtype=int)

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        assert isinstance(value, Iterable)
        if len(list(value)) != self.shape[0]:
            raise ValueError("Size of value is not equal to the shape[0].")
        self._index = Index(value)

    @property
    def columns(self):
        return self._columns

    @columns.setter
    def columns(self, value):
        assert isinstance(value, Iterable)
        if (self.ndim > 1 and len(list(value)) != self.shape[1]):
            raise ValueError("Size of value is not equal to the shape[1].")
        self._columns = Index(value)