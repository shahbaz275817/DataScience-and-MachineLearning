import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

df = pd.read_csv('iris1.csv')

target_dict = {'Iris-setosa': 0,'Iris-versicolor': 1,'Iris-virginica': 2}
df['target'] = df['Species'].map(target_dict)

x = df.iloc[:,1:5].values
y = df.iloc[:,6:].values
# dummies=pd.get_dummies(df,drop_first=True)
# print(dummies)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)
knn = KNeighborsClassifier(n_neighbors=4)
#knn = LogisticRegression()
knn.fit(x_train,y_train)

prediction = knn.predict(x_test)
print("Prediction: {}".format(prediction))

neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)
    #knn = LogisticRegression()

    # Fit the classifier to the training data
    knn.fit(x_train, y_train)

    # Compute accuracy on the training set
    train_accuracy[i] = knn.score(x_train, y_train)

    # Compute accuracy on the testing set
    test_accuracy[i] = knn.score(x_test, y_test)


# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


############# Classification reports and confusion matrix
print(confusion_matrix(y_test,prediction))
print(classification_report(y_test,prediction))


accuracy_train = knn.score(x_train,y_train)
accuracy_test = knn.score(x_test,y_test)
print(accuracy_train)
print(accuracy_test)

param_grid = {'n_neighbors':np.arange(1, 25)}
knn = KNeighborsClassifier()
knn_cv =  GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)
print("KNN best params: {}".format(knn_cv.best_params_))
print("KNN best score: {}".format(knn_cv.best_score_))
