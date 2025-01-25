from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("best_decision_tree_model.pkl")

# Define a prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the POST request
    data = request.get_json()

    # Convert input data into numpy array
    features = np.array(data['features']).reshape(1, -1)

    # Get model prediction
    prediction = model.predict(features)

    # Return prediction as a JSON response
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
