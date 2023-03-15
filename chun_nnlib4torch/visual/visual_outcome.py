import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import re

def visual_train_val_loss(df):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(df['epoch'],df['train_loss'],label='Train Loss')
    ax.plot(df['epoch'],df['val_loss'],label='Val Loss')
    ax.legend()
    ax.plot()

def visual_train_val_acc(df):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(df['epoch'],df['train_acc'],label='Train Acc')
    ax.plot(df['epoch'],df['val_acc'],label='Val Acc')
    ax.legend()
    ax.plot()

def visual_train_val(df):
    visual_train_val_acc(df)
    visual_train_val_loss(df)
    
def visual_cf_matrix(df):
    sn.heatmap(df, annot=True,fmt='d').get_figure()
    
def load_train_csv_revise(file_name):
    train_outcome = pd.read_csv('./run/train_outcome_20230216-194908.csv')
    train_acc = train_outcome['train_acc']
    val_acc = train_outcome['val_acc']
    search_accuracy = re.compile(r'\d\.[0-9._]+')
    for i in range(len(train_acc)):
        train_acc_num = search_accuracy.search(train_acc.iloc[i])
        val_acc_num = search_accuracy.search(val_acc.iloc[i])
        train_acc.iloc[i] = float(train_acc_num.group())
        val_acc.iloc[i] = float(val_acc_num.group())
    return train_outcome