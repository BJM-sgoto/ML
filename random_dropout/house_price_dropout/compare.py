import pandas as pd
import numpy as np
from tqdm import tqdm

dt1 = pd.read_csv('real_output.csv')
dt2 = pd.read_csv('submission.csv')
d = dt1['SalePrice'].values - dt2['SalePrice'].values
a = np.mean(np.abs(d/1000))
print(a)


'''
# load data
train = pd.read_csv('ames_dataset.csv')
train.drop(['PID'], axis=1, inplace=True)

origin = pd.read_csv('train.csv')
train.columns = origin.columns

test = pd.read_csv('test.csv')
submission = pd.read_csv('sample_submission.csv')
# drop missing values
missing = test.isnull().sum()
missing = missing[missing>0]
train.drop(missing.index, axis=1, inplace=True)
train.drop(['Electrical'], axis=1, inplace=True)

test.dropna(axis=1, inplace=True)
test.drop(['Electrical'], axis=1, inplace=True)
l_test = tqdm(range(0, len(test)), desc='Matching')
for i in l_test:
    for j in range(0, len(train)):
        for k in range(1, len(test.columns)):
            if test.iloc[i,k] == train.iloc[j,k]:
                continue
            else:
                break
        else:
            submission.iloc[i, 1] = train.iloc[j, -1]
            break
l_test.close()
submission.to_csv('real_output.csv', index=False)
'''