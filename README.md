🪙 Gold Price Prediction using LSTM

A deep learning project that predicts future gold prices using a Long Short-Term Memory (LSTM) neural network.
The model is trained on historical price data and visualizes both actual and predicted trends.

🧠 Overview

Gold prices are influenced by multiple dynamic factors, making prediction a challenging time-series problem.
This project applies Recurrent Neural Networks (RNNs) — specifically LSTM layers — to learn temporal dependencies in price data and forecast future values.

Goal:

Train an LSTM model on gold price data

Evaluate performance with metrics such as RMSE

Visualize real vs. predicted values

📊 Dataset

The dataset typically contains historical gold prices with columns such as:

Date

Open

High

Low

Close

Volume (optional)

You can replace the dataset with any other time series data with a similar format.

⚙️ Project Structure
📦 Gold-Price-Prediction-LSTM
 ┣ 📂 data/                 # Historical gold price dataset (CSV)
 ┣ 📂 models/               # Trained model weights (.h5)
 ┣ 📜 gold_price_lstm.py    # Main script for training & prediction
 ┣ 📜 utils.py              # Helper functions for data preprocessing
 ┣ 📜 requirements.txt      # Dependencies
 ┗ 📜 README.md             # Documentation

🧩 Key Steps
1️⃣ Data Preprocessing

Load CSV data

Normalize using MinMaxScaler

Split into train/test sets

Convert into supervised sequences for LSTM

2️⃣ Model Architecture

Sequential model

One or more LSTM layers

Dense output layer for regression

Example:

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

3️⃣ Training & Evaluation
model.fit(X_train, y_train, epochs=50, batch_size=32)
predictions = model.predict(X_test)

4️⃣ Visualization

Matplotlib is used to plot:

Actual vs Predicted prices

Training loss curve

🚀 How to Run
🧰 Requirements

Install dependencies:

pip install -r requirements.txt

▶️ Run the script
python gold_price_lstm.py

💾 Output

Trained model saved as .h5 file

RMSE printed in console

Plot showing predicted vs actual prices

📈 Example Result
<p align="center"> <img src="docs/gold_prediction_plot.png" width="600"> </p>
🧰 Technologies Used
Category	Library
Deep Learning	TensorFlow / Keras
Data Handling	pandas, numpy
Visualization	matplotlib
Scaling	sklearn.preprocessing
Environment	Python 3.x
📄 License

Developed by Mohammad Mottaghi
as part of an academic project on Time Series Forecasting using LSTM Networks.
© 2024 Mohammad Mottaghi
