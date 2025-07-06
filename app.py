from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and scaler
model = joblib.load("Personalitys.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected input columns
columns = ['Time_spent_Alone', 'Stage_fear', 'Social_event_attendance',
           'Going_outside', 'Drained_after_socializing', 'Friends_circle_size', 'Post_frequency']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        values = [float(request.form.get(col)) for col in columns]
        df = pd.DataFrame([values], columns=columns)

        # Scale input
        scaled = scaler.transform(df)

        # Predict
        pred = model.predict(scaled)[0]
        label = "Introvert" if pred == 1 else "Extrovert"

        return render_template('index.html', prediction=label)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
