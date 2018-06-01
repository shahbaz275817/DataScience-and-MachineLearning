import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13].values
y = df.iloc[:, 13].values

from sklearn.preprocessing import OneHotEncoder, LabelEncoder

labelencoder1 = LabelEncoder()
labelencoder2 = LabelEncoder()
X[:, 1] = labelencoder1.fit_transform(X[:, 1])
X[:, 2] = labelencoder2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

model = Sequential()
model.add(Dense(12, kernel_initializer='uniform', activation='relu', input_dim=11))
model.add(Dropout(p=0.1))
model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
model.add(Dropout(p=0.1))
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=10)

y_predict = model.predict(X_test)
y_predict = (y_predict > 0.5)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predict)

new_prediction = model.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Cross-Validation and parameter tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


def build_classifier(optimizer='adam'):
    model = Sequential()
    model.add(Dense(12, kernel_initializer='uniform', activation='relu', input_dim=11))
    model.add(Dense(12, kernel_initializer='uniform', activation='relu'))
    model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


classifier = KerasClassifier(build_fn=build_classifier, epochs=100, batch_size=10)
accuracies = cross_val_score(classifier, X_train, y_train, cv=10)
print(accuracies)

# Parameter tuning with Grid Search CV
from sklearn.model_selection import GridSearchCV

params = {'epochs': [50, 100, 500, 1000],
          'batch_size': [25, 32],
          'optimizer': ['adam', 'rmsprop']}
grid_cv = GridSearchCV(estimator=classifier, param_grid=params, scoring=['accuracy'], cv=10, refit = False)
grid_cv = grid_cv.fit(X_train, y_train)
best_params = grid_cv.best_params_
best_accuracy = grid_cv.best_score_
