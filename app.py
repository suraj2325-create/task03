from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        transformed = scaler.transform([data])
        prediction = model.predict(transformed)
        return render_template('index.html', prediction_text=f"Predicted Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [float(x) for x in request.form.values()]
        transformed = scaler.transform([data])
        prediction = model.predict(transformed)
        return render_template('index.html', prediction_text=f"Predicted Price: ₹{prediction[0]:,.2f}")
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
