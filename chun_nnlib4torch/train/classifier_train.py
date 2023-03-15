import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import abc
import copy
import time
from .train_method import TrainMethod
from torch.utils.tensorboard import SummaryWriter
from evaluate.confusion_matrix import createConfusionMatrix
from visual.visual_outcome import visual_train_val
from .outcome.classifier_outcome import store_classifier_outcome
import matplotlib.pyplot as plt
import pandas as pd

class ClassifierTrainWithSummaryWriter(TrainMethod):

    def __init__(self, model, dataloader, criterion, optimizer,confusion_matrix_name_tuple,device = None, evaluation = None, epochs = 25):

        super().__init__( model, dataloader, criterion, optimizer, evaluation, epochs)
        self._confusion_matrix_name_tuple = confusion_matrix_name_tuple
        if device==None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

    def fit(self):
        since = time.time() # 記錄開始時間
        timestr = time.strftime("%Y%m%d-%H%M%S")
        train_store_path = 'runs/train_'+type(self._model).__name__+'_'+timestr
        val_store_path = 'runs/val_'+type(self._model).__name__+'_'+timestr

        writer_train = SummaryWriter(train_store_path)
        writer_val = SummaryWriter(val_store_path)
        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0

        for epoch in range(self._epochs):
            print('Epoch {}/{}'.format(epoch, self._epochs - 1))
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._model.train()  # 將模型設定為訓練模式
                else:
                    self._model.eval()   # 將模型設定為驗證模式
                running_loss = 0.0
                running_corrects = 0
                n_sample = 0
                n_val_sample = 0
            # 以 DataLoader 載入 batch 資料
                for inputs, labels in self._dataloader[phase]:
                    # 將資料放置於 GPU 或 CPU
                    inputs = inputs.to(self._device,dtype=torch.float)
                    labels = labels.to(self._device)

                    # 重設參數梯度（gradient）
                    self._optimizer.zero_grad()
                   # 只在訓練模式計算參數梯度

                    with torch.set_grad_enabled(phase == 'train'):
                        # 正向傳播（forward）
                        outputs = self._model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self._criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()  # 反向傳播（backward）
                            self._optimizer.step() # 更新參數
                            n_sample +=labels.size(0) 
                        elif phase == 'val':
                            n_val_sample +=labels.size(0)

                    # 計算統計值
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    

                if phase == 'train':
                    # 更新 scheduler
                    #scheduler.step()
                    epoch_loss = running_loss / n_sample
                    epoch_acc = running_corrects.double() / n_sample
                    writer_train.add_scalar('Acc',epoch_acc,epoch)
                    writer_train.add_scalar('Loss',epoch_loss,epoch)

                elif phase == 'val':
                    epoch_loss = running_loss / n_val_sample
                    epoch_acc = running_corrects.double() / n_val_sample
                    writer_val.add_scalar('Acc',epoch_acc,epoch)
                    writer_val.add_scalar('Loss',epoch_loss,epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 記錄最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self._model.state_dict())
    # 輸出最佳準確度
        print('Best val Acc: {:4f}'.format(best_acc))

        # 載入最佳模型參數
        self._model.load_state_dict(best_model_wts)
        heatmap,cf_matrix = createConfusionMatrix(self._model,self._dataloader['val'],self._confusion_matrix_name_tuple)
        writer_val.add_figure("Confusion Matrix",heatmap)
        writer_val.close()
        writer_train.close()
        return self._model

class ClassifierTrainWithMatplotlib(TrainMethod):

    def __init__(self, model, dataloader, criterion, optimizer,confusion_matrix_name_tuple,device=None, evaluation = None, epochs = 25,store_outcomes=False):

        super().__init__( model, dataloader, criterion, optimizer, evaluation, epochs)
        self._confusion_matrix_name_tuple = confusion_matrix_name_tuple

        self._store_outcomes = store_outcomes
        if device==None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device


    def fit(self):
        since = time.time() # 記錄開始時間
        timestr = time.strftime("%Y%m%d-%H%M%S")

        self.outcomes = {'epoch':[],'train_loss':[],'train_acc':[],'val_loss':[],'val_acc':[]}
        best_model_wts = copy.deepcopy(self._model.state_dict())
        best_acc = 0.0

        for epoch in range(self._epochs):
            print('Epoch {}/{}'.format(epoch, self._epochs - 1))
            print('-' * 10)
            self.outcomes['epoch'].append(epoch)
            for phase in ['train', 'val']:
                if phase == 'train':
                    self._model.train()  # 將模型設定為訓練模式
                else:
                    self._model.eval()   # 將模型設定為驗證模式
                running_loss = 0.0
                running_corrects = 0
                n_sample = 0
                n_val_sample = 0
            # 以 DataLoader 載入 batch 資料
                with torch.no_grad():
                    for inputs, labels in self._dataloader[phase]:
                        # 將資料放置於 GPU 或 CPU
                        inputs = inputs.to(self._device,dtype=torch.float)
                        labels = labels.to(self._device)

                        # 重設參數梯度（gradient）
                        self._optimizer.zero_grad()
                       # 只在訓練模式計算參數梯度

                        with torch.set_grad_enabled(phase == 'train'):
                            # 正向傳播（forward）
                            outputs = self._model(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = self._criterion(outputs, labels)

                            if phase == 'train':
                                loss.backward()  # 反向傳播（backward）
                                self._optimizer.step() # 更新參數
                                n_sample +=labels.size(0) 
                                torch.cuda.empty_cache()
                            elif phase == 'val':
                                n_val_sample +=labels.size(0)

                        # 計算統計值
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    

                if phase == 'train':
                    # 更新 scheduler
                    #scheduler.step()
                    epoch_loss = running_loss / n_sample
                    epoch_acc = running_corrects.double() / n_sample

                    self.outcomes['train_loss'].append(epoch_loss)
                    self.outcomes['train_acc'].append(torch.detach(epoch_acc.data.cpu()).item())
                    

                elif phase == 'val':
                    epoch_loss = running_loss / n_val_sample
                    epoch_acc = running_corrects.double() / n_val_sample

                    self.outcomes['val_loss'].append(epoch_loss)
                    self.outcomes['val_acc'].append(torch.detach(epoch_acc.data.cpu()).item())

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

                # 記錄最佳模型
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self._model.state_dict())
            
    # 輸出最佳準確度
        print('Best val Acc: {:4f}'.format(best_acc))

        # 載入最佳模型參數
        self._model.load_state_dict(best_model_wts)
        self.outcomes = pd.DataFrame.from_dict(self.outcomes)
        
        visual_train_val(self.outcomes)

        heatmap,cf_matrix = createConfusionMatrix(self._model,self._dataloader['val'],self._confusion_matrix_name_tuple)
        if self._store_outcomes:
            store_classifier_outcome(self.outcomes,cf_matrix)
            
        return self._model
