import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Reading the dataset from local computer
df = pd.read_csv('./datasets/iris.csv')
print(df.to_string())
df.info()


# Extracting the Features Column Name
col_names = list(df.keys()[1:5])

# Set the plot style
sns.set_style('dark')

# Create 4 subplots
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))

# Converting the matrix of Axes(Cartesian Planes) in a 1D numpy array
axes = np.array(axes).flatten()

# For Loop with two index on two different iterables
for col, ax in zip(col_names, axes):
    # Drawing the 4 BoxPlot one for each feature
    sns.boxplot(data=df, y=col, hue='Species', ax=ax)

# Create a pair scatter plot
sns.pairplot(df.drop('Id', axis=1), hue='Species', height=2)

# Shows all the plots
plt.show()
