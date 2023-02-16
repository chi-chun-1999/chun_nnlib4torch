import pandas as pd
import time
import os

def store_classifier_outcome(train_val_df, cf_matrix_df,store_dir='run/', store_with_time_name=True):
    
    train_val_str = None
    cf_matrix_str = None
    
    if not os.path.exists(store_dir):
        os.mkdir(store_dir)
    if store_with_time_name:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        train_val_str = store_dir+"train_outcome_"+timestr+".csv"
        cf_matrix_str = store_dir+"cf_matrix_outcome_"+timestr+".csv"
    else:
        train_val_str = store_dir+"train_outcome.csv"
        cf_matrix_str = store_dir+"cf_matrix_outcome.csv"
        
    train_val_df.to_csv(train_val_str,index=False)
    cf_matrix_df.to_csv(cf_matrix_str,index=False)

