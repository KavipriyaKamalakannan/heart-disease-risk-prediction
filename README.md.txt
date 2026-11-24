# Heart Disease Risk Prediction

## Project Overview
This project develops a machine learning-based application to predict the presence of heart disease using clinical parameters.

The application uses a trained Random Forest classification model and Streamlit to provide an interactive prediction interface.

## Objectives
- Analyse clinical parameters associated with heart disease.
- Develop a machine learning classification model.
- Predict the presence or absence of heart disease.
- Provide an interactive web-based interface.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib
- Google Colab

## Dataset
UCI Cleveland Heart Disease Dataset.

Dataset source: https://archive.ics.uci.edu/dataset/45/heart+disease

## Model
Random Forest Classifier trained using the Cleveland Heart Disease dataset.

## Application Features
- Accepts 13 clinical input parameters.
- Predicts the model's classification.
- Displays an estimated prediction probability.

## How to Run the Application

1. Install the required packages:

   `python -m pip install -r requirements.txt`

2. Run the application:

   `python -m streamlit run app.py`

3. Open the local URL displayed in the terminal.

## Disclaimer
This project is developed for educational and academic purposes only. It is not intended for clinical diagnosis or medical decision-making. Predictions require appropriate clinical validation before any real-world use.