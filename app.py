import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open("model_svm.pkl", "rb"))

@app.route("/")
def home() :
    return render_template('index.html')

@app.route("/predict", methods=["POST"])
def predict() :
    mka = request.form['mka']
    sks = request.form['sks']
    ipk = request.form['ipk']
    float_features = [float(x) for x in request.form.values()]
    feature = [np.array(float_features)]
    prediction = model.predict(feature)
    return render_template("index.html", prediction_text="{}".format(prediction), mka=mka, sks=sks, ipk=ipk)

if __name__ == "__main__" :
    app.run(debug=True)