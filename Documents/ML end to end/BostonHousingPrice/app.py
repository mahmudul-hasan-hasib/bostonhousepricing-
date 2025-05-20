import pickle
import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

## Load model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))  # Load the LinearRegression model
scaler = pickle.load(open('scaling.pkl', 'rb'))     # Load the StandardScaler

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print(data)
        # Transform the input data using the scaler
        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
        # Make prediction using the regression model
        output = regmodel.predict(new_data)
        print(output[0])
        return jsonify(float(output[0]))
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The house price prediction is {}".format(output))

if __name__ == "__main__":
    app.run(debug=True)