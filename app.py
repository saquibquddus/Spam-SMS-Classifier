# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 07:54:53 2021

@author: Saquib Quddus
"""
import pickle    
from flask import Flask,render_template,request

classifier=pickle.load(open('classifier.pkl','rb'))
cv=pickle.load(open('Count_Vectorizer.pkl','rb'))
app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method=="POST":
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=classifier.predict(vect)
        return render_template('result.html',prediction=my_prediction)
    
if __name__ == '__main__':
	app.run(debug=False)
    

    

