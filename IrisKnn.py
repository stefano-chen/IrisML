import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('./datasets/iris.csv')
x = df.drop(columns=['Id', 'Species']).to_numpy()
y = df['Species'].to_numpy()

encoder = LabelEncoder()
scaler = StandardScaler()
y = encoder.fit_transform(y)
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)
params = {
    'n_neighbors': [3, 4, 5, 7, 10],
    'p': [1, 2, 3]
}

cls = GridSearchCV(KNeighborsClassifier(), param_grid=params, scoring='accuracy', cv=10)
cls.fit(x_train, y_train)
print(f'Best parameters: {cls.best_params_}\n')
print(f'Best score: {cls.best_score_*100:4.2f}%\n')

predictor = cls.best_estimator_
y_pred = predictor.predict(x_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)*100:4.2f}%\n')

sns.set_style('dark')
cmatrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cmatrix).plot(colorbar=False)
plt.show()
