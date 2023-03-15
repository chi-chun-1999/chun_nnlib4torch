#%%
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from visual.visual_outcome import visual_train_val
import pandas as pd

# %%
train_outcome = pd.read_csv('./run/train_outcome_20230216-194908.csv')


train_acc = train_outcome['train_acc']
val_acc = train_outcome['val_acc']
#%%
search_accuracy = re.compile(r'\d\.[0-9]+')
for i in range(len(train_acc)):
    train_acc_num = search_accuracy.search(train_acc.iloc[i])
    val_acc_num = search_accuracy.search(val_acc.iloc[i])
    train_acc.iloc[i] = float(train_acc_num.group())
    val_acc.iloc[i] = float(val_acc_num.group())

print(train_outcome)
# %%
train_outcome.to_csv('test.csv',index=False)
# %%

train_outcome = pd.read_csv('./run/train_outcome_20230315-093713.csv')
visual_train_val(train_outcome)
# %%
