# ============================================================
# TEMPERATURE ANOMALY PREDICTION STREAMLIT DASHBOARD
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import random
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Ridge
from tensorflow.keras.models import load_model

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(page_title="Temperature Anomaly Prediction", layout="wide")

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

page = st.sidebar.selectbox(
    "Navigation",
    ["Home","EDA","Model Comparison","Prediction","Regional Analysis"]
)

# ============================================================
# RANDOM SEED
# ============================================================

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_excel("Temperature_dataset.xlsx")

    df.columns = df.columns.str.strip().str.lower()

    df.rename(columns={
        'date':'Date',
        't2m_max':'Tmax',
        't2m_min':'Tmin',
        'doy':'DOY'
    }, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'])

    df = df.sort_values('Date').reset_index(drop=True)

    df = df.dropna(subset=['Tmax','Tmin'])

    return df

df = load_data()

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df['Tmean'] = (df['Tmax'] + df['Tmin'])/2
df['temp_range'] = df['Tmax'] - df['Tmin']

df['Tmean'] = df['Tmean'].rolling(7).mean()

train_mask = (df['Date'].dt.year >= 2010) & (df['Date'].dt.year <= 2020)

climatology = df[train_mask].groupby('DOY')['Tmean'].mean()

df['climatology_mean'] = df['DOY'].map(climatology)

df['temp_anomaly'] = df['Tmean'] - df['climatology_mean']

df['anomaly_lag1'] = df['temp_anomaly'].shift(1)
df['anomaly_lag3'] = df['temp_anomaly'].shift(3)
df['anomaly_lag7'] = df['temp_anomaly'].shift(7)

df['rolling_mean_7'] = df['temp_anomaly'].rolling(7).mean()
df['rolling_std_7'] = df['temp_anomaly'].rolling(7).std()

df['month'] = df['Date'].dt.month

df_model = df.dropna().reset_index(drop=True)

# ============================================================
# REGION MAP
# ============================================================

region_map = {

"chennai":"Coastal",
"kanchipuram":"Coastal",
"thoothukudi":"Coastal",
"ramanathapuram":"Coastal",
"puducherry":"Coastal",
"thiruvarur":"Coastal",
"kanyakumari":"Coastal",

"madurai":"Plain",
"trichy":"Plain",
"karur":"Plain",
"perambalur":"Plain",
"viruthunagar":"Plain",

"coimbatore":"Plateau",
"dharmapuri":"Plateau",
"vellore":"Plateau",
"tiruppattur":"Plateau",

"salem":"Hilly",
"theni":"Hilly",
"dindigul":"Hilly"

}

if "place" in df_model.columns:

    df_model["place"] = df_model["place"].str.lower()

    df_model["region"] = df_model["place"].map(region_map).fillna("Unknown")

# ============================================================
# FEATURES
# ============================================================

features = [
'anomaly_lag1',
'anomaly_lag3',
'anomaly_lag7',
'rolling_mean_7',
'rolling_std_7',
'temp_range',
'month',
'DOY'
]

X = df_model[features]
y = df_model['temp_anomaly']

train_mask = (df_model['Date'].dt.year >= 2010) & (df_model['Date'].dt.year <= 2020)
test_mask = (df_model['Date'].dt.year >= 2021) & (df_model['Date'].dt.year <= 2025)

X_train = X[train_mask]
y_train = y[train_mask]

X_test = X[test_mask]
y_test = y[test_mask]

# ============================================================
# SCALER
# ============================================================

@st.cache_resource
def get_scaler(X_train):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return scaler, X_train_scaled, X_test_scaled

scaler, X_train_scaled, X_test_scaled = get_scaler(X_train)

# ============================================================
# LOAD MACHINE LEARNING MODELS
# ============================================================

@st.cache_resource
def load_ml_models():

    xgb = joblib.load("xgb_model.pkl")
    rf = joblib.load("rf_model.pkl")

    xgb_pred = xgb.predict(X_test_scaled)
    rf_pred = rf.predict(X_test_scaled)

    return xgb, rf, xgb_pred, rf_pred

xgb, rf, xgb_pred, rf_pred = load_ml_models()

# ============================================================
# LOAD DEEP LEARNING MODELS
# ============================================================

@st.cache_resource
def load_dl_models():

    def create_sequences(X,y,steps=3):

        Xs=[]
        ys=[]

        for i in range(len(X)-steps):
            Xs.append(X.iloc[i:i+steps].values)
            ys.append(y.iloc[i+steps])

        return np.array(Xs),np.array(ys)

    X_seq,y_seq = create_sequences(X,y,3)

    split = int(0.8*len(X_seq))

    X_train_s,X_test_s = X_seq[:split],X_seq[split:]
    y_train_s,y_test_s = y_seq[:split],y_seq[split:]

    test_dates_seq = df_model['Date'].iloc[split + 3:].reset_index(drop=True)

    rnn_model = load_model("rnn_model.keras")
    cnn_model = load_model("cnn_model.keras")

    rnn_pred = rnn_model.predict(X_test_s).flatten()
    cnn_pred = cnn_model.predict(X_test_s).flatten()

    return rnn_pred, cnn_pred, y_test_s, test_dates_seq

rnn_pred, cnn_pred, y_test_s, test_dates_seq = load_dl_models()

# ============================================================
# HYBRID MODEL
# ============================================================

@st.cache_resource
def train_hybrid():

    min_len = min(len(xgb_pred), len(rf_pred))

    xgb_aligned = xgb_pred[-min_len:]
    rf_aligned = rf_pred[-min_len:]
    y_aligned = y_test.values[-min_len:]

    stack_features = np.column_stack((xgb_aligned,rf_aligned))

    meta_model = Ridge(alpha=1.0)
    meta_model.fit(stack_features,y_aligned)

    hybrid_pred = meta_model.predict(stack_features)

    return hybrid_pred, xgb_aligned, rf_aligned, y_aligned, min_len, meta_model

hybrid_pred, xgb_aligned, rf_aligned, y_aligned, min_len, meta_model = train_hybrid()

# ============================================================
# HOME PAGE
# ============================================================

if page == "Home":

    st.title("Temperature Anomaly Prediction")

    st.write("""
This project predicts **temperature anomalies** using multiple machine learning
and deep learning models.

### Models Used
- XGBoost
- Random Forest
- Vanilla RNN
- CNN
- Hybrid Model (XGBoost + Random Forest)

The Hybrid model combines XGBoost and RandomForest predictions using
a stacking method to improve accuracy.
""")

    st.subheader("Dataset Head")

    st.dataframe(df.head())

# ============================================================
# EDA
# ============================================================

elif page == "EDA":

    st.title("Exploratory Data Analysis")

    fig = plt.figure(figsize=(12,4))
    plt.plot(df_model['Date'],df_model['temp_anomaly'])
    plt.axhline(0,linestyle="--")
    plt.title("Temperature Anomaly Time Series")
    st.pyplot(fig)

    fig = plt.figure(figsize=(10,6))
    sns.heatmap(df_model.corr(numeric_only=True),cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    st.pyplot(fig)

    corr = df_model.corr(numeric_only=True)

    fig = plt.figure(figsize=(8,6))
    target_corr = corr['temp_anomaly'].abs().sort_values(ascending=False)[1:11]
    target_corr.plot(kind='bar', color='lightblue', edgecolor='black', linewidth=1)
    plt.title("Top 10 Features Most Correlated with Temperature Anomaly")
    plt.ylabel("Correlation Value")
    plt.xlabel("Features")
    st.pyplot(fig)

    for col in ['Tmean','temp_anomaly','rolling_mean_7','rolling_std_7']:

        fig = plt.figure()
        sns.histplot(df_model[col],kde=True)
        plt.title(f"Distribution of {col}")
        st.pyplot(fig)

    fig = plt.figure(figsize=(12,5))
    plt.plot(df_model['Date'], df_model['Tmean'])
    plt.title("Time Series of Mean Temperature")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    st.pyplot(fig)

# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("Model Performance Comparison")

    results=[]

    def evaluate(name,y_true,y_pred):

        rmse=np.sqrt(mean_squared_error(y_true,y_pred))
        mse=mean_squared_error(y_true,y_pred)
        mae=mean_absolute_error(y_true,y_pred)
        r2=r2_score(y_true,y_pred)

        results.append([name,rmse,mse,mae,r2])

    evaluate("XGBoost",y_test,xgb_pred)
    evaluate("RandomForest",y_test,rf_pred)
    evaluate("Vanilla RNN",y_test_s,rnn_pred)
    evaluate("CNN",y_test_s,cnn_pred)
    evaluate("Hybrid",y_aligned,hybrid_pred)

    comparison_df=pd.DataFrame(results,columns=["Model","RMSE","MSE","MAE","R2"])

    st.dataframe(comparison_df)

    metrics = ["RMSE", "MAE", "R2"]

    for metric in metrics:
      fig = plt.figure(figsize=(8,4))
      plt.barh(
          comparison_df["Model"],
          comparison_df[metric]
          )
      plt.title(f"Model Comparison - {metric}")
      plt.xlabel(metric)
      plt.ylabel("Model")
      plt.grid(alpha=0.3)
      st.pyplot(fig)

    x = np.arange(len(comparison_df["Model"])) # Move x and width inside the block for clarity
    width = 0.25

    fig_overall = plt.figure(figsize=(10,6))
    plt.bar(x - width, comparison_df["RMSE"], width, label="RMSE", color="blue")
    plt.bar(x, comparison_df["R2"], width, label="R²", color="green")
    plt.bar(x + width, comparison_df["MSE"], width, label="MSE", color="red")

    plt.xticks(x, comparison_df["Model"])
    plt.xlabel("Models")
    plt.ylabel("Metric Values")
    plt.title("Overall Model Performance Comparison")

    plt.legend()
    plt.grid(axis='y', linestyle="--", alpha=0.6)

    st.pyplot(fig_overall)

    st.subheader("Actual vs Predicted - XGBoost")

    dates=df_model[test_mask]['Date'].iloc[-min_len:]

    fig=plt.figure(figsize=(12,5))

    plt.plot(dates,y_aligned,label="Actual")
    plt.plot(dates,xgb_aligned,label="XGBoost Prediction")

    plt.legend()
    plt.title("XGBoost Actual vs Predicted")

    st.pyplot(fig)


    st.subheader("Actual vs Predicted - Random Forest")

    fig=plt.figure(figsize=(12,5))

    plt.plot(dates,y_aligned,label="Actual")
    plt.plot(dates,rf_aligned,label="RandomForest Prediction")

    plt.legend()
    plt.title("RandomForest Actual vs Predicted")

    st.pyplot(fig)


    st.subheader("Actual vs Predicted - Hybrid Model")

    fig=plt.figure(figsize=(12,5))

    plt.plot(dates,y_aligned,label="Actual")
    plt.plot(dates,hybrid_pred,label="Hybrid Prediction")

    plt.legend()
    plt.title("Hybrid Model Actual vs Predicted")

    st.pyplot(fig)


    st.subheader("Actual vs Predicted - Vanilla RNN")

    fig=plt.figure(figsize=(12,5))

    plt.plot(test_dates_seq,y_test_s,label="Actual")
    plt.plot(test_dates_seq,rnn_pred,label="RNN Prediction")

    plt.legend()
    plt.title("Vanilla RNN Actual vs Predicted")

    st.pyplot(fig)


    st.subheader("Actual vs Predicted - CNN")

    fig=plt.figure(figsize=(12,5))

    plt.plot(test_dates_seq,y_test_s,label="Actual")
    plt.plot(test_dates_seq,cnn_pred,label="CNN Prediction")

    plt.legend()
    plt.title("CNN Actual vs Predicted")

    st.pyplot(fig)

# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.title("Hybrid Model Prediction")

    future_date=st.date_input("Select Date")

    if st.button("Predict"):

        doy=pd.to_datetime(future_date).dayofyear
        month=pd.to_datetime(future_date).month

        last=df_model.iloc[-1]

        input_data=np.array([[

            last["temp_anomaly"],
            df_model.iloc[-3]["temp_anomaly"],
            df_model.iloc[-7]["temp_anomaly"],
            df_model["temp_anomaly"].tail(7).mean(),
            df_model["temp_anomaly"].tail(7).std(),
            last["temp_range"],
            month,
            doy

        ]])

        input_scaled=scaler.transform(input_data)

        xgb_future=xgb.predict(input_scaled)
        rf_future=rf.predict(input_scaled)

        stack_future=np.column_stack((xgb_future,rf_future))

        hybrid_future=meta_model.predict(stack_future)

        st.success(f"Predicted anomaly: {hybrid_future[0]:.2f} °C")

# ============================================================
# REGIONAL ANALYSIS
# ============================================================

elif page == "Regional Analysis":

    st.title("Regional Temperature Anomaly Analysis")

    def classify(v):

        if v>1:
            return "Extreme Heat"
        elif v>0.5:
            return "Warm"
        elif v>=-0.5:
            return "Normal"
        elif v>=-1:
            return "Cool"
        else:
            return "Extreme Cold"

    df_model["Anomaly_Class"]=df_model["temp_anomaly"].apply(classify)

    fig=plt.figure(figsize=(10,5))
    sns.countplot(data=df_model,x="Anomaly_Class",hue="region")
    st.pyplot(fig)

    st.subheader("Anomaly Values")

    st.dataframe(df_model[["Date","place","region","temp_anomaly","Anomaly_Class"]])

    st.subheader("Average Regional Anomaly")

    avg_table=df_model.groupby("region")["temp_anomaly"].mean().reset_index()

    st.dataframe(avg_table)
