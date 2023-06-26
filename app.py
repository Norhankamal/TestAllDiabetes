from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import LabelEncoder
import pickle

app = Flask(__name__)

# Render the HTML page
@app.route('/')
def home():
    return 'Welcome To My Application'

# Load the XGBoost model
xgb_model = pickle.load(open('xgb_model.pkl','rb'))

# Load the scaler model
scaler = pickle.load(open('scaler.pkl','rb'))

# Load the label encoder model
label_encoder = LabelEncoder()
label_encoder = pickle.load(open('label_encoder.pkl','rb'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs from the request
        user_inputs = request.json['user_inputs']

        # Prepare user inputs for prediction
        user_inputs = [float(value) for value in user_inputs]

        # Perform any required preprocessing steps on user inputs
        user_inputs_scaled = scaler.transform([user_inputs])

        # Predict diabetes for user inputs
        prediction = xgb_model.predict(user_inputs_scaled)
        predicted_class = label_encoder.inverse_transform(prediction)

        return jsonify({'predicted_class': predicted_class[0]})

    except FileNotFoundError:
        return jsonify({'error': 'File not found.'})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()