# 1. Consider the Iris Dataset
# 2. Design and implement an ML-based procedure for answering thi question:
#       what's the hardest variable to be predicted in the dataset?

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error


# Create 4 dataset one for each prediction variable
def prepare_data():
    df = pd.read_csv('./datasets/iris.csv')
    df.info()
    df.drop(columns=['Id'], inplace=True)
    enc = LabelEncoder()
    scaler = StandardScaler()
    # Get all column that are not object, to scale them
    scaling_columns = list(df.select_dtypes(exclude=['object']).columns)
    # Get all column that are object(Categorical value) to encod them in integers
    encoding_columns = list(df.select_dtypes(include=['object']).columns)
    df[scaling_columns] = scaler.fit_transform(df[scaling_columns])

    # !!!IMPORTANT!!! without using apply(), the resulted dataframe will still have the column as object type
    # which can cause the sklearn models to launch an error during the fitting phase (I don't know why)
    df[encoding_columns] = df[encoding_columns].apply(enc.fit_transform)
    df.info()
    data = []

    for col in df.keys().values:
        x = df.drop(columns=col)
        y = df[[col]]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
        data.append({
            'x_train': x_train,
            'x_test': x_test,
            'y_train': y_train,
            'y_test': y_test
        })
    return data


def regression_model(data):
    model = DecisionTreeRegressor()
    model.fit(data['x_train'], data['y_train'])
    y_pred = model.predict(data['x_test'])
    error = mean_absolute_error(data['y_test'], y_pred)
    deep = model.get_depth()
    return error, deep


def classification_model(data):
    model = DecisionTreeClassifier()
    model.fit(data['x_train'], data['y_train'])
    y_pred = model.predict(data['x_test'])
    score = accuracy_score(data['y_test'], y_pred)
    deep = model.get_depth()
    return score, deep


if __name__ == '__main__':
    list_dataset = prepare_data()
    result = []
    for data in list_dataset[:4]:
        result.append(regression_model(data))
    result.append(classification_model(list_dataset[-1]))
    print(result)
