import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Obtaining Data
df = pd.read_csv('./datasets/iris.csv')
df.info()
# Create Observation set and Class set
x = df.drop(columns=['Id', 'Species'])
y = df['Species']

# Normalization and LabelEncoding
encoder = LabelEncoder()
scaler = StandardScaler()
y = encoder.fit_transform(y)
x = scaler.fit_transform(x)

# Splitting Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# GridSearch with CrossValidation to determine the best parameters
params = {
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
cls = GridSearchCV(SVC(), param_grid=params, scoring='accuracy', cv=10)
cls.fit(x_train, y_train)
print(f'\nBest parameters: {cls.best_params_}\n')
print(f'Accuracy on Train set: {cls.best_score_*100:4.2f}%\n')
predictor = cls.best_estimator_

# Testing the best Classifier
y_pred = predictor.predict(x_test)

# Calculate Accuracy on Test set
print(f'Accuracy on Test set: {accuracy_score(y_test, y_pred)*100:4.2f}%\n')

# Create the Plot
sns.set_style('dark')
fig, ax = plt.subplots()
ax.set_title('Confusion Matrix')
plt.gcf().canvas.manager.set_window_title('Support Vector Machine')

# Calculate Confusion Matrix
cmatrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cmatrix).plot(ax=ax).im_.colorbar.remove()
plt.show()


