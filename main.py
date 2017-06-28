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


def one_hot_encoding(cols, train):
    for col in cols:
        # Perform encoding on training data
        one_hot = pd.get_dummies(train[col], prefix=col)
        if col <> "playerid":
            train = train.drop(col, axis=1)
        train = train.join(one_hot)

    return train


print ('Loading data...')

train = pd.read_csv('data/train_2016_v2.csv')
props = pd.read_csv('data/properties_2016.csv')
sample = pd.read_csv('data/sample_submission.csv')

print ('Binding to float32...')

drop_cols = ['parcelid', 'logerror', 'transactiondate', 'propertyzoningdesc', 'propertycountylandusecode']
one_hot_encode_cols = ['airconditioningtypeid', 'architecturalstyletypeid', 'buildingclasstypeid',
                       'heatingorsystemtypeid', 'storytypeid']

for c, dtype in zip(props.columns, props.dtypes):
#    print c, dtype
#    s = props[c]
#    print s.head()
#    print s.describe()

    if dtype == np.float32:
        props[c] = props[c].astype(np.float32)

print ('One hot encode categorical columns...')

props = one_hot_encoding(one_hot_encode_cols, props)

print ('Create training set...')

df_train = train.merge(props, how='left', on='parcelid')

x_train = df_train.drop(drop_cols, axis=1)
y_train = df_train['logerror'].values

print (x_train)
print (x_train.shape, y_train.shape)

train_cols = x_train.columns

for c in x_train.dtypes[x_train.dtypes == object].index.values:
    x_train[c] = (x_train[c] == True)

del df_train; gc.collect()

split = 80000

x_train, y_train, x_valid, y_valid = x_train[:split], y_train[:split], x_train[split:], y_train[split:]

print ('Building DMatrix...')

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)

del x_train, x_valid; gc.collect()

print ('Training...')

params = {}
params['eta'] = 0.02
params['objective'] = 'reg:linear'
params['eval_metric'] = 'mae'
params['max_depth'] = 4
params['silent'] = 1

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
clf = xgb.train(params, d_train, 10000, watchlist, early_stopping_rounds = 100, verbose_eval=10)

del d_train, d_valid; gc.collect()

print ('Building test set...')

sample['parcelid'] = sample['ParcelId']

df_test = sample.merge(props, how='left', on='parcelid')

del props; gc.collect()

x_test = df_test[train_cols]

for c in x_test.dtypes[x_test.dtypes == object].index.values:
    x_test[c] = (x_test[c] == True)

del df_test, sample; gc.collect()

d_test = xgb.DMatrix(x_test)

del x_test; gc.collect()

print ('Predicting on test...')

p_test = clf.predict(d_test)

del d_test; gc.collect()

sub = pd.read_csv('data/sample_submission.csv')

for c in sub.columns[sub.columns != 'ParcelId']:
    sub[c] = p_test

print ('Writing to CSV...')
sub.to_csv('data/xgb_starter.csv', index=False, float_format='%.4f')

print (sub.shape)
print (sub.iloc[407820])
print (sub.iloc[407821])
print (sub.iloc[407822])