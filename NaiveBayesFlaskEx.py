#!flask/bin/python
from flask import Flask
from flask import request
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np

#assigning predictor and target variables
x= np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
y = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4]) 

#Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets 
model.fit(x, y)

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/prediction', methods=['GET'])
def get_prediction():
    f1 = float(request.args.get('f1'))
    f2 = float(request.args.get('f2'))
    prediction = model.predict([[f1,f2]])
    return str(prediction)
   
if __name__ == '__main__':
    app.run(port=5000,host='0.0.0.0')        
    # app.run(debug=True)