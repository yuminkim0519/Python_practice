import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2.5)
import missingno as msno

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

###############################################

df_train = pd.read_csv('train.csv')
# df_test  = pd.read_csv('test.csv')


##
# for col in df_train.columns:
#     msg = 'column : {:>10}\t Percent of NaN value: {:.2f}%'.format(col, 100 * (df_train[col].isnull().sum() / df_train[col].shape[0]))
#     print(msg)

## 결측치 확인 1
# msno.matrix(df = df_train.iloc[:, :], figsize = (8, 8), color = (0.8, 0.5, 0.2))

## 결측치 확인 2
# msno.bar(df = df_train.iloc[:, :], figsize=(8, 8), color = (.8, .5, .2))

plt.show()