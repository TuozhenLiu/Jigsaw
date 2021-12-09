# -*- coding:utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------------
# Author: Tuozhen
# Date: 2021/12/9
# Description:
# ----------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy as np


train_df = pd.read_csv("input/validation_data.csv")
k_fold_indexes = np.array_split(np.random.permutation(np.arange(train_df.shape[0])), 10)
fold = 0
train = train_df[~train_df.index.isin(k_fold_indexes[fold])].reset_index(drop=True)
print(train.shape)
