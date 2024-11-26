from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict import CustomData, PredictPipeline

app = Flask('score')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method =='GET':
        return render_template('form.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )
        student_df = data.convert_data_to_dataframe()
        print(student_df)

        print("Starting prediction")
        predict_pipeline = PredictPipeline()

        print("Processing prediction")
        results = predict_pipeline.predict(student_df)

        print("Completing prediction")
        return render_template('form.html', results=f"{results[0]:.2f}")


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
