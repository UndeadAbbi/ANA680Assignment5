from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
        data = request.get_json(force=True)
        features = np.array(feature_values).reshape(1, -1)
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)
        return jsonify({'quality': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
