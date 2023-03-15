import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from keras.datasets import mnist

from model.cnn_model import TestConvNet
from torchsummary import summary
from torch.utils.data import DataLoader
from train.classifier_train import ClassifierTrainWithSummaryWriter
import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim

if __name__ == '__main__':
    #dwt_transform = DiscreteWaveletTransform(times=3)
    #data = yaml.load(parameter_yml,Loader=yaml.CLoader)
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255

    featuresTrain = torch.from_numpy(X_train)
    targetsTrain = torch.from_numpy(Y_train) # data type is long

    featuresTrain = featuresTrain.unsqueeze(1)

    featuresTest = torch.from_numpy(X_test)
    targetsTest = torch.from_numpy(Y_test) # data type is long
    featuresTest = featuresTest.unsqueeze(1)

    train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
    test = torch.utils.data.TensorDataset(featuresTest,targetsTest)

    dataloader = {'train':DataLoader(train,batch_size=100,shuffle=True),'val':DataLoader(test,batch_size=100,shuffle=True)}

    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    dwt_cnn = TestConvNet().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(dwt_cnn.parameters(),lr=0.001,momentum=0.9)

    confusion_matrix_name_tuple = ('0','1','2','3','4','5','6','7','8','9')

    dwt_train = ClassifierTrainWithSummaryWriter(dwt_cnn,dataloader,criterion,optimizer_ft,confusion_matrix_name_tuple,device=device,epochs=10)

    dwt_train.fit()
