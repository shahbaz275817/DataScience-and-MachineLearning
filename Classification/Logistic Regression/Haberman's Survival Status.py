import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('data.csv')

target_dict = {2: 0, 1: 1}
df['target'] = df['survived'].map(target_dict)   #Required for roc_curve as the function assumes 0 as false and 1 as true

X = df.iloc[:,:3]
Y = df.loc[:,'target']

c_space = np.logspace(-5,8,15)
param_grid = {'C': c_space}

logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=10)

logreg.fit(X_train,y_train)
logreg_cv.fit(X_train,y_train)

y_predict = logreg.predict(X_test)

print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))

print(logreg.score(X_test,y_test))
print("Tuned LogistricRegression Paramameters: {}".format(logreg_cv.best_params_))
print("Best score is: {}".format(logreg_cv.best_score_))

y_predict_prob = logreg.predict_proba(X_test)[:,1]
fpr, tpr, threshold = roc_curve(y_test,y_predict_prob)

cv_scores = cross_val_score(logreg,X,Y,cv=5,scoring='roc_auc')
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_scores))


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()