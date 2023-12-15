# 1. Consider the Iris Dataset
# 2. Design and implement an ML-based procedure for answering thi question:
#       what's the hardest variable to be predicted in the dataset?

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import set_config

# Setting to allow all types of Transformers to return a pandas Dataframe instead of a numpy 2D array
set_config(transform_output='pandas')

df = pd.read_csv('./datasets/iris.csv')
x = df.drop(columns=['Id', 'SepalLengthCm'])
y = df[['SepalLengthCm']]

# If sparse_output is False the OneHotEncoder returns a list instead of a sparse matrix
enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
enc.fit(x[['Species']])

encoded = enc.transform(x[['Species']])
print(encoded)
x = pd.concat([x, encoded], axis=1)
x.drop(columns=['Species'], inplace=True, axis=1)
print(x.to_string())
print(y.to_string())

