import numpy as np
import pandas as pd

dataset = pd.read_csv('data.csv',header=None)
X = dataset.iloc[:,1:]
y = dataset.iloc[:,0:1]

from sklearn.model_selection  import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components = 2 )
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print(explained_variance)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state = 42)
clf.fit(X_train,y_train)

y_predict = clf.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_predict)
print(cm)
print(accuracy_score(y_test,y_predict))
