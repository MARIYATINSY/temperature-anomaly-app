# Temperature Anomaly Detection

## About the Project

Temperature Anomaly Detection is a machine learning project that analyzes historical temperature data and identifies unusual temperature conditions.

The main aim of this project is to understand how temperature changes over time and predict temperature anomalies using different machine learning and deep learning models.

The project also includes a Streamlit dashboard where the results can be easily viewed and explored.

## Objectives

* Analyze historical temperature data
* Calculate temperature anomalies
* Study temperature trends and seasonal patterns
* Create useful time-series features
* Train different machine learning and deep learning models
* Compare the performance of the models
* Predict temperature anomalies
* Identify different temperature conditions
* Display the results using a Streamlit dashboard

## Dataset

The dataset contains historical temperature data from **2010 to 2025**.

The data is divided into:

* Training data: 2010–2020
* Testing data: 2021–2025

Some of the important features used in the project are:

* Temperature anomaly
* Previous day anomaly
* 3-day lag anomaly
* 7-day lag anomaly
* 7-day rolling mean
* 7-day rolling standard deviation
* Month
* Day of Year
* Temperature range

## What is Temperature Anomaly?

Temperature anomaly is the difference between the observed temperature and the normal or reference temperature.

For example:

```text
Temperature Anomaly = Observed Temperature - Reference Temperature
```

A positive value means the temperature is higher than normal, while a negative value means it is lower than normal.

## Models Used

The following models were implemented and compared:

1. Random Forest
2. XGBoost
3. Vanilla RNN
4. CNN
5. Hybrid Model

The hybrid approach combines predictions from machine learning models to improve the prediction performance.

## Model Performance

The current results obtained from the project are:

| Model         | R² Score |
| ------------- | -------: |
| XGBoost       |   95.09% |
| Hybrid Model  |   95.09% |
| Random Forest |   92.85% |
| Vanilla RNN   |   89.35% |
| CNN           |   83.64% |

Among the tested models, **XGBoost and the Hybrid Model** gave the best performance with an R² score of about **95.09%**.

## Temperature Categories

The predicted temperature anomaly is classified into five categories:

* Above +1°C → Extreme Heat
* +0.5°C to +1°C → Warm
* -0.5°C to +0.5°C → Normal
* -1°C to -0.5°C → Cool
* Below -1°C → Extreme Cold

This makes the prediction easier to understand.

## Streamlit Dashboard

A Streamlit dashboard was developed for this project.

The dashboard contains:

* Home
* Exploratory Data Analysis
* Model Comparison
* Prediction
* Regional Analysis

The dashboard allows users to view the data, compare model performance, and make temperature anomaly predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* TensorFlow
* Keras
* Streamlit
* Jupyter Notebook
* VS Code

## Project Workflow

```text
Temperature Data
      ↓
Data Preprocessing
      ↓
Temperature Anomaly Calculation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Temperature Anomaly Prediction
      ↓
Streamlit Dashboard
```

## How to Run

First clone the repository:

```bash
git clone https://github.com/your-username/Temperature-Anomaly-Detection.git
```

Go to the project folder:

```bash
cd Temperature-Anomaly-Detection
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Future Improvements

In the future, this project can be improved by:

* Adding more weather and climate variables
* Improving the hybrid model
* Using advanced deep learning models
* Adding real-time weather data
* Adding more locations
* Adding automatic alerts for extreme temperatures
* Deploying the Streamlit dashboard online

## Author

**Mariyatinsy**

MSc Data Science

This project was developed as an academic project to study temperature anomalies using machine learning and deep learning.
