import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv')
print(df.head())

# df = pd.get_dummies(df)
# print(df.head())
X = df.iloc[:,2:8]
y = df.iloc[:,8:]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)

ridge = Ridge(alpha=0.5,normalize=True).fit(X_train,y_train)
ridge_cv = cross_val_score(ridge,X,y,cv=5)
print(ridge_cv)

print(ridge.score(X_test,y_test))

# df.boxplot('PRP','CACH',rot=60)
# plt.show()
m, b = np.polyfit(X_train['CACH'],y_train['PRP'],deg=1)
plt.plot(X_train['CACH'],m*X_train['CACH']+b)
plt.scatter(X_train['CACH'],y_train['PRP'],color='blue')
plt.show()

lasso = Lasso(alpha=0.4,normalize=True)
lasso.fit(X_train,y_train)
print(lasso.score(X_test,y_test))
print(lasso.coef_)
