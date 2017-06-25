import numpy as np
import pandas as pd
import xgboost as xgb
import gc
import matplotlib.pyplot as plt

def showPlot(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=10)
    plt.show()


print ('Loading data...')

train = pd.read_csv('data/train_2016_v2.csv')
props = pd.read_csv('data/properties_2016.csv')
sample = pd.read_csv('data/sample_submission.csv')

print ('Binding to float32...')

drop_cols = ['parcelid', 'logerror', 'transactiondate']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid']

for c, dtype in zip(props.columns, props.dtypes):
#    print c, dtype
#    s = props[c]
#    print s.head()
#    print s.describe()

    if dtype == np.float32:
        props[c] = props[c].astype(np.float32)

print ('Create training set...')

df_train = train.merge(props, how='left', on='parcelid')

x_train = df_train.drop(drop_cols, axis=1)
y_train = df_train['logerror'].values
print (x_train.shape, y_train.shape)

train_cols = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print ('Building DMatrix...')
