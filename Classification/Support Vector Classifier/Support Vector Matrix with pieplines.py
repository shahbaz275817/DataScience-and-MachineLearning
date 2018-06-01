from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

df = pd.read_csv('data.csv')
df=df.drop(columns=['capital-gain','capital-loss'])
df = pd.get_dummies(df,drop_first=True)
print(df.shape)
df[df == ' ?'] = np.nan
df = df.dropna(0,how='any')
print(df.shape)

X_train = df.iloc[:,:98].values
y_train = df.iloc[:,98:].values
y_train = np.reshape(y_train,(32561,))

df1 = pd.read_csv('test.csv')
df1=df1.drop(columns=['capital-gain','capital-loss'])
df1 = pd.get_dummies(df1,drop_first=True)
df1 = df1.drop(columns=['income_ <=50K.'])
print(df1.shape)
df1[df1 == ' ?'] = np.nan
df1 = df1.dropna(0,how='any')
print(df1.shape)
X_test = df1.iloc[:,:98].values
y_test = df1.iloc[:,98:].values
y_test = np.reshape(y_test,(16282,))

imp = Imputer(missing_values='NaN',strategy='most_frequent',axis=0)
# imp.fit(X_test)
# X_test = imp.transform(X_test)
from xgboost import XGBClassifier
clf = SVC()

from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.5, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(clf, param_grid= parameters, scoring='accuracy', cv= 5, n_jobs= -1)
grid_search.fit(X_train,y_train)
# print("SVC best params: {}".format(grid_search.best_params_))
# print("SVC best score: {}".format(grid_search.best_score_))

steps = [('imputer',imp),('cv',grid_search),('SVM',clf)]

pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
y_predict = pipeline.predict(X_test)
print(pipeline.score(X_test,y_test))
print(y_predict)
print(classification_report(y_test,y_predict))
print("SVC best params: {}".format(pipeline.best_params_))
print("SVC best score: {}".format(pipeline.best_score_))