# Machine Learning Zoomcamp 2024: Midterm Project

## Student Exam Performance Prediction
This project presents an end-to-end machine learning solution for predicting student exam performance. 

## Problem Description 
The objective of this project is to build a regression model for predicting the United States high school students' mathematics examination scores (column: 'math score') to identify academically strong students prior to the examination.

If we can reach 80% accuracy at predicting students math exam scores during the proof of concept, the school will use this information to invite students to enroll in Advanced Placement (AP) courses.

## Dataset
**Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)

**Description**:
This dataset describes the high school students from the United States (gender, race/ethnicity, parental background), the influence of other variables on students performance (lunch type, test preparation course), and their marks in various subjects (math score, reading score, and writing score).

## Project Structure

```bash
├── artifacts
│   ├── data.csv
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── test.csv
│   ├── train.csv
├── notebook
│   ├── data
│   │   ├── StudentsPerformance.csv
│   ├── eda.ipynb
│   ├── model_training.ipynb
├── src
│   ├── components
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── predict.py
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   ├── utils.py
├── templates
│   ├── form.html
│   ├── index.html
├── .gitignore
├── Dockerfile
├── README.md
├── app.py
├── requirements.txt
└── setup.py
```

## How to Run This Project (locally)
**1. Prerequisites**

You need to have Docker installed.

**2. Download the project files**

Clone this repository:
```bash
git clone https://github.com/wanyingng/student-performance-prediction.git
```
Or, select "Download ZIP"

**3. Set up the environment**

Navigate to the project root directory:
```bash
cd student-exam-performance-prediction
```

Create a virtual environment and activate it:
```bash
python3.11 -m venv .venv
.\.venv\Scripts\activate
```

Install the project dependencies:
```bash
pip install -r requirements.txt
```

Re-build the models (Optional if `model.pkl` and `preprocessor.pkl` already exist in `/artifacts` folder):
```bash
python src/components/data_ingestion.py
```

**4. Build the Dockerfile**

```bash
docker build -t score-service .
```

**5. Run the Docker image**

```bash
docker run -it --rm -p 9696:9696 score-service
```

**6. Access the Flask app through your web browser**

Enter `http://127.0.0.1:9696/predict` or `http://localhost:9696/predict` as the URL.









