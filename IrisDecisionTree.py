import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Reading and Splitting the dataset
df = pd.read_csv('./datasets/iris.csv')
x = df.values[:, 1:5]
y = df.values[:, 5]

# Define a LabelEncoder and a StandardScale to normalize the data
encoder = LabelEncoder()
scaler = StandardScaler()

# Calculate the Label Class
encoder.fit(y)
# Calculate the mean and variance to normalize the data
scaler.fit(x)

# Normalization
x = scaler.transform(x)
# Label Encoding
y = encoder.transform(y)

# Splitting the data in learning set and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y , test_size=0.2, random_state=100, shuffle=True)

# Doing a GridSearch with CrossValidation to determine the best parameters
# Create a dictionary with a subset of the DecisionTree's parameters
params = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [5, 3, 2, 7],
    'min_samples_split': [2, 3, 4]
}
# Creating a GridSearch on a DecisionTreeClassifier, using roc-auc-ovr as score and with 10 folds
cls = GridSearchCV(tree.DecisionTreeClassifier(), param_grid=params, scoring='roc_auc_ovr', cv=10)
# Apply the GridSearch
cls.fit(x_train, y_train)
# Extracting the best classifier
predictor = cls.best_estimator_

# Testing using the test set
y_pred = predictor.predict(x_test)

# Display Confusion Matrix
sns.set_style('dark')
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].set_title('Confusion Matrix')
axes[1].set_title('Tree Plot')
plt.gcf().canvas.manager.set_window_title('Support Vector Machine')

cm = confusion_matrix(y_test, y_pred, labels=predictor.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=predictor.classes_).plot(ax=axes[0]).im_.colorbar.remove()
tree.plot_tree(predictor, ax=axes[1], fontsize=12)

# Calculate the Accuracy of the predictor
str_accuracy = f'Accuracy : {accuracy_score(y_test, y_pred)*100:4.2f}%'
plt.text(0.05, 0.05, s=str_accuracy, fontsize=12, transform=plt.gcf().transFigure)

# Using the predictor with a new data
data = [[6.3, 2.5, 5.0, 2.0,]]
data = scaler.transform(data)
y_pred = predictor.predict(data)
y_pred_prob = predictor.predict_proba(data)[0]
print(data)
print(encoder.inverse_transform(y_pred)[0] + f' with probability: {max(y_pred_prob)}')

plt.show()




