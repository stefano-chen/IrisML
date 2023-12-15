import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import roc_curve, RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Loading Data
df = pd.read_csv('./datasets/iris.csv')
x = df.drop(columns=['Id', 'Species']).to_numpy()
y = df['Species'].to_numpy()

# Preprocessing
encoder = LabelEncoder()
scaler = StandardScaler()
y = encoder.fit_transform(y)
x = scaler.fit_transform(x)

# Splitting Dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

# GridSearch with CrossValidation
params = {
    'n_estimators': [50, 75, 100, 125, 150],
    'criterion': ['gini', 'entropy', 'log_loss']
}
cls = GridSearchCV(RandomForestClassifier(), params, scoring='roc_auc_ovo', cv=10)
cls.fit(x_train, y_train)
print(f'Best parameters: {cls.best_params_}\n')
print(f'Best score: {cls.best_score_*100:4.2f}%\n')
predictor = cls.best_estimator_

# Testing the predictor
y_pred = predictor.predict(x_test)
print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100:4.2f}%\n')

# Calculate Confusion Matrix
sns.set_style('dark')
fig, ax = plt.subplots()
ax.set_title('Confusion Matrix')
plt.gcf().canvas.manager.set_window_title('Random Forest')
cmatrix = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cmatrix).plot(ax=ax, colorbar=False)
plt.show()



