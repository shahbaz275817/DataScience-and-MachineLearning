from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV


train = pd.read_csv('train.txt', sep='\t', dtype=str, header=None)
test =  pd.read_csv('test.txt', sep='\t', dtype=str, header=None)

X_train = train.iloc[:,1:]
y_train = train.iloc[:,0:1]

X_test = test.iloc[:,1:]
y_test = test.iloc[:,0:1]


TOKENS_ALPHANUMERIC = '[A-Za-z]+(?=\\s+)'

steps = [('vectorizer',HashingVectorizer(token_pattern = TOKENS_ALPHANUMERIC,
                                                    norm=None, binary=False, lowercase=False,
                                                    ngram_range=(1,2))),
        ('scale', MaxAbsScaler()),
        ('clf', OneVsRestClassifier(LogisticRegression()))]

pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
accuracy = pipeline.score(X_test,y_test)
print(accuracy)

