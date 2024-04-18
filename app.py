import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import joblib
app = Flask(__name__)




@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    test_data = list(request.form.values())
    test_data = [[float(s) for s in test_data]]
    test_data = pd.DataFrame(test_data)
    file = open("model.joblib","rb")
    model = joblib.load(file)
    output = model.predict(test_data)
    output = output[0]
    return render_template('index.html', prediction_text= "The Credit Score is {}".format(output))
if __name__ == "__main__":
    app.run(debug=True)