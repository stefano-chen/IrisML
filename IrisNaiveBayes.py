import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv('./datasets/iris.csv')
x = df.drop(columns=['Id', 'Species']).to_numpy()
y = df['Species'].to_numpy()

encoder = LabelEncoder()
scaler = StandardScaler()
y = encoder.fit_transform(y)
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

score = cross_validate(GaussianNB(), x, y, cv=10, scoring='roc_auc_ovr')
for key in score:
    print(f'{key}: {score[key]}\n')

cls = GaussianNB()
cls.fit(x_train, y_train)
y_pred = cls.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100}%')
cmatrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cmatrix).plot(colorbar=False)
plt.show()

