# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 07:59:59 2021

@author: Saquib Quddus
"""
import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
import pickle
from sklearn.feature_extraction.text import CountVectorizer

dataset=pd.read_csv("spam.csv",encoding='latin-1') 

dataset=dataset[["v1","v2"]].rename(columns={"v1":"label","v2":"message"},inplace=False)

X=dataset["message"]
y=pd.get_dummies(dataset["label"],drop_first=True)



corpus=[]
ps=PorterStemmer()
for row in range(0,len(X)):
    message=re.sub("[^a-zA-Z]"," ",X[row])
    message=message.lower()
    words=message.split()
    words=[word for word in words if word not in set(stopwords.words('english'))]
    words=[ps.stem(word) for word in words]
    message=" ".join(words)
    corpus.append(message)
    
cv=CountVectorizer()     
X_final=cv.fit_transform(corpus)
pickle.dump(cv,open('Count_Vectorizer.pkl','wb'))
X_train,X_test,y_train,y_test=train_test_split(X_final,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()
classifier.fit(X_train,y_train)
prediction=classifier.predict(X_test)   

pickle.dump(classifier,open('classifier.pkl','wb'))

from sklearn.metrics import accuracy_score,confusion_matrix
print(accuracy_score(y_test,prediction))    
print(confusion_matrix(y_test,prediction))



 

    