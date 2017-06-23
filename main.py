import numpy as np
import pandas as pd
import xgboost as xgb
import gc

print ('Loading data...')

train = pd.read_csv('data/train_2016_v2.csv')
props = pd.read_csv('data/properties_2016.csv')
sample = pd.read_csv('data/sample_submission.csv')

print ('Binding to float32...')

for c, dtype in zip(props.columns, props.dtypes):
    if dtype == np.float32:
        props[c] = props[c].astype(np.float32)

print ('Create training set...')

