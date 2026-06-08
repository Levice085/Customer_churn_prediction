#  Telco Customer Churn Prediction

An interactive machine learning web application built with Streamlit that predicts the likelihood of a telecommunications customer churning (leaving the provider). 

## Project Overview
This project leverages the **Telco Customer Churn dataset** from Kaggle. The predictive model is powered by a **Gradient Boosting Classifier**, trained to identify high-risk customers based on their demographics, account information, and subscribed services.

The project features a complete end-to-end pipeline:
1. **Exploratory Data Analysis (EDA) & Model Training:** Handled in `customer_churn_prediction.ipynb`.
2. **Data Preprocessing Pipeline:** Uses exact feature mapping from training, custom label encoding, and standard scaling to ensure flawless real-time processing.
3. **Interactive Web Interface:** Built with Streamlit (`web_app.py`) for an intuitive user experience.

## Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Machine Learning:** Scikit-Learn (Gradient Boosting)
* **Data Manipulation:** Pandas, NumPy
* **Model Serialization:** Joblib

## Repository Structure
* `web_app.py`: The main Streamlit application script containing the UI and prediction logic.
* `customer_churn_prediction.ipynb`: Jupyter notebook containing EDA, feature engineering, and the training loop for the Gradient Boosting model.
* `model.pkl`: The serialized, tuned Gradient Boosting classification model.
* `scaler.pkl`: The fitted Standard Scaler used to normalize inputs and enforce strict column ordering.
* `encoders_dict.pkl`: A saved dictionary of unique Label Encoders required to process categorical text inputs into numeric data safely.
* `requirements.txt`: The list of Python package dependencies required for deployment.

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd <your-repository-directory>
