from flask import Flask, request, jsonify
import joblib
import statsmodels.api as sm
import pandas as pd

app = Flask(__name__)

# Load the GLM model
model = joblib.load('glm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    df = pd.DataFrame([data])

    # Add constant for GLM model
    df = sm.add_constant(df)

    # Make prediction
    prediction = model.predict(df)

    return jsonify({'premium_prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
