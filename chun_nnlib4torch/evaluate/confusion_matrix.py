from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np

def createConfusionMatrix(model, loader,classes_name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    y_pred = []
    y_true = []
    for i,(inputs,labels) in enumerate(loader):
        if torch.cuda.is_available():
            inputs, labels = inputs.to(device,dtype=torch.float), labels.to(device)
        output = model(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.cpu().numpy()
        y_true.extend(labels)  # save ground truth

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix, index=[i for i in classes_name],
                         columns=[i for i in classes_name])

    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True,fmt='d').get_figure(),df_cm
