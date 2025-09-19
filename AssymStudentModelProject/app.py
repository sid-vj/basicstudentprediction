from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)


model = pickle.load(open("student_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    name = request.form['name']
    roll = request.form['roll']
    student_class = request.form['class']
    department = request.form['department']
    year = request.form['year']
    quiz = float(request.form['quiz'])
    midterm = float(request.form['midterm'])
    model_exam = float(request.form['model_exam'])

    # Prepare data for model (must match training feature order)
    features = np.array([[quiz, midterm, model_exam]])

    # Get prediction from model
    prediction = model.predict(features)[0]

    return render_template(
        'index.html',
        prediction_text=f"Prediction for {name} (Roll {roll}): {prediction}"
    )

if __name__ == "__main__":
    app.run(debug=True)
