import pandas as pd
import matplotlib.pyplot as plt

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
    

    
