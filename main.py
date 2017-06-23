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

drop_cols = ['parcelid']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid']

for c, dtype in zip(props.columns, props.dtypes):
    print c, dtype
    s = props[c]
    print s.head()
    print s.describe()

    if dtype == np.float32:
        props[c] = props[c].astype(np.float32)

print ('Create training set...')



