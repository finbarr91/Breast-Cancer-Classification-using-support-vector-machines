import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

cancer = load_breast_cancer()
print('\nThe breast cancer dataset:\n',cancer)

print('\nDataset Keys:\n',cancer.keys())
print('\n Description of the Cancer:\n',cancer['DESCR'])

print('\n Target of the dataset:\n ',cancer['target'])

print('\n Target names of the dataset:\n',cancer['target_names'])
print('\n Feature nanes of the dataset:\n',cancer['feature_names'])
print('\n Shape of the dataset:\n',cancer['data'].shape)

# CREATING A DATAFRAME SO THAT THE MANIPULATION OF THE DATASET WILL BE EASIER

df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns = np.append(cancer['feature_names'],['target']))

print(df_cancer.head())
print(df_cancer.tail())

# VISUALIZING THE DATASET
sns.pairplot(df_cancer, hue= 'target', vars= ['mean radius', 'mean texture', 'mean area', 'mean perimeter', 'mean smoothness'])
plt.show()

sns.countplot(df_cancer['target'])
plt.show()

sns.scatterplot(x='mean area', y= 'mean smoothness', hue = 'target', data = df_cancer)
plt.show()

plt.figure(figsize=(30,20))
sns.heatmap(df_cancer.corr(), annot= True)
plt.tight_layout()
plt.show()

# MODEL TRAINING (FINDING A PROBLEM SOLUTION)
X = df_cancer.drop(['target'], axis=1)
y = df_cancer['target']
print(X.shape)
print(y.shape)
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size= 0.3, random_state= 42)
svc_model = SVC()
svc_model.fit(X_train,y_train)
accuracy = svc_model.score(X_test,y_test)
print('\n Accuracy:\n', accuracy)

# EVALUATING THE MODEL
y_predict = svc_model.predict(X_test)


cm= confusion_matrix(y_test,y_predict) # to evaluate the performance of the model
sns.heatmap(cm,annot=True)
plt.show()

# IMPROVING THE MODEL
min_train = X_train.min()
range_train = (X_train-min_train).max()
X_train_scaled = (X_train - min_train)/range_train
sns.scatterplot(x=X_train['mean area'], y= X_train['mean smoothness'], hue = y_train)
plt.show()

sns.scatterplot(x=X_train_scaled['mean area'], y= X_train_scaled['mean smoothness'], hue = y_train)
plt.show()

min_test = X_test.min()
range_test = (X_test - min_test).max()
X_test_scaled = (X_test - min_test)/range_test

svc_model.fit(X_train_scaled,y_train)
Acc = svc_model.score(X_test_scaled,y_test)
print(Acc)
y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_predict)
sns.heatmap(cm, annot=True)
plt.show()

print(classification_report(y_test,y_predict))

# IMPROVING THE MODEL - TO IMPROVE THE VALUE OF C AND GAMMA
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [1,0.1,0.01,0.001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit= True, verbose=4)
grid.fit(X_train_scaled,y_train)
print(grid.best_params_)

grid_predictions = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test , grid_predictions)
sns.heatmap(cm, annot= True)

print(classification_report(y_test,y_predict))

