# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the datasetsiop_ml_train_participant.cs
dataset = pd.read_csv('ficheiro.tsv', delimiter="\t")

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
#download de uma package que tem palavras que nao sao significantes
#para o estudo como 'thi, that, a, or'
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = [] #coleçao de texto do mesmo tipo
# 1088 numero de linhas a serem examinadas no dataset
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['coluna_independente'][i]) #estamos a substituir 
    review = review.lower() #lowercase
    review = review.split() #mete em lista cada palavra
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    # com o stem estamos a meter as palavras no infinitivo, ou seja, por exemplo
    # a palavra 'loved' passa para 'love' e tambem outro tipo de mudanças
    # que podem ser feitas para melhor compreensão do texto.
    review = ' '.join(review) #retorna o novo texto com as mudanças
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500) #limpar texto, numero de palavras mais usadas
X = cv.fit_transform(corpus).toarray() #palavras mais utilizadas (ja retiramos as que nao interessam)
y = dataset.iloc[:, 1].values #coluna dos valores dependentes com todas as linhas

#######################################
# algoritmo GaussianNB

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


########################################
# algoritmo random_forest_classification


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25,random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 45, criterion = 'entropy', random_state = 0, warm_start=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
