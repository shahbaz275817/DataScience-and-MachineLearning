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
from sklearn.cross_validation import train_test_split

train = pd.read_csv('train.txt', sep='\t', dtype=str, header=None)
test = pd.read_csv('test.txt', sep='\t', dtype=str, header=None)

X_train = train.iloc[:,1:].values.tolist()
X_train = [item for sublist in X_train for item in sublist]
y_train = train.iloc[:,0:1]
#
X_test = test.iloc[:,1:].values.tolist()
X_test =[item for sublist in X_test for item in sublist]
y_test = test.iloc[:,0:1]

TOKENS_ALPHANUMERIC = '[A-Za-z]+(?=\\s+)'

steps = [('vectorizer',HashingVectorizer(token_pattern = TOKENS_ALPHANUMERIC, norm=None, binary=False, lowercase=False,
                                                     ngram_range=(1,2))),
         ('scale', MaxAbsScaler()),
         ('clf', OneVsRestClassifier(LogisticRegression()))]

pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
accuracy = pipeline.score(X_test,y_test)
print(accuracy)

#### Another Method by manual text cleaning #########

# Text cleaning
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,11293):
    news = re.sub('[^a-zA-z]', ' ', train[1][i])
    #news = news.lower()
    news = news.split()
    ps = PorterStemmer()
    news = [ps.stem(word) for word in news if not word in set(stopwords.words('english'))]
    news = ' '.join(news)
    corpus.append(news)


cv = CountVectorizer(token_pattern= TOKENS_ALPHANUMERIC,stop_words='english')
X = cv.fit_transform(corpus).toarray()
y = train.iloc[:,0:1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print(cm)
acc = accuracy_score(y_test,y_pred)
print(acc)