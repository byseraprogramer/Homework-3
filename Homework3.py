import concurrent
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import ta
from bs4 import BeautifulSoup
from flask import Flask, abort, jsonify
from flask_cors import CORS
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense, LSTM, Dropout
from keras.src.models import Sequential
from keras.src.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import StochasticOscillator
from ta.trend import CCIIndicator
app = Flask(__name__)
CORS(app)

def fetch_issuer_codes(base_url):
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    options = soup.select(".panel .form-control option")
    issuer_codes = [
        option["value"].strip()
        for option in options if option["value"].strip() and not any(char.isdigit() for char in option["value"])
    ]
    return issuer_codes


def extract_issuer_data(valid_code):

    def fetch_yearly_data(year):
        data_for_year = []
        try:
            response = requests.post(
                f"https://www.mse.mk/mk/stats/symbolhistory/{valid_code}",
                data={"FromDate": f"01.01.{year}", "ToDate": f"31.12.{year}"}
            )
            soup = BeautifulSoup(response.text, 'html.parser')
            rows = soup.select("tbody tr")
            for row in rows:
                cols = row.select("td")
                if len(cols) >= 9:
                    data_for_year.append({
                        "NAME": valid_code,
                        "DATE": cols[0].text.strip(),
                        "PRICE OF LAST TRANSACTION IN mkd": cols[1].text.strip(),
                        "MAX": cols[2].text.strip(),
                        "MIN": cols[3].text.strip(),
                        "AVERAGE PRICE": cols[4].text.strip(),
                        "%CHANGE": cols[5].text.strip(),
                        "QUANTITY": cols[6].text.strip(),
                        "Turnover in BEST in mkd": cols[7].text.strip(),
                        "TOTAL TURNOVER in mkd": cols[8].text.strip()
                    })
        except Exception as e:
            print(f"Error fetching data for {valid_code} in year {year}: {e}")
        return data_for_year

    matrix = []
    today = datetime.today()
    current_year = today.year
    years = [current_year - i for i in range(10)]  # Last 10 years

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        results = executor.map(fetch_yearly_data, years)
        for yearly_data in results:
            matrix.extend(yearly_data)

    if not matrix:
        return pd.DataFrame()

    def safe_convert_to_float(column):
        if column.dtype == 'O':
            return column.str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        return column

    data = pd.DataFrame(matrix)
    data['DATE'] = pd.to_datetime(data['DATE'], format='%d.%m.%Y')
    data.sort_values('DATE', inplace=True)
    data['PRICE OF LAST TRANSACTION IN mkd'] = (
         data['PRICE OF LAST TRANSACTION IN mkd'].replace('', pd.NA)
          .fillna(method='ffill').fillna(method='bfill').fillna(0)
    )
    data['MIN'] = data['MIN'].replace('', pd.NA)
    data['MAX'] = data['MAX'].replace('', pd.NA)
    data['MIN'] = data['MIN'].fillna(data['PRICE OF LAST TRANSACTION IN mkd'])
    data['MAX'] = data['MAX'].fillna(data['PRICE OF LAST TRANSACTION IN mkd'])

    data['AVERAGE PRICE'] = data['AVERAGE PRICE'].replace('', pd.NA)
    data['QUANTITY'] = data['QUANTITY'].replace('', pd.NA)
    data['%CHANGE'] = data['%CHANGE'].replace('', pd.NA)
    data['Turnover in BEST in mkd'] = data['Turnover in BEST in mkd'].replace('', pd.NA)
    data['TOTAL TURNOVER in mkd'] = data['TOTAL TURNOVER in mkd'].replace('', pd.NA)

    data['AVERAGE PRICE'] = data['AVERAGE PRICE'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    data['QUANTITY'] = data['QUANTITY'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    data['%CHANGE'] = data['%CHANGE'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    data['Turnover in BEST in mkd'] = data['Turnover in BEST in mkd'].fillna(method='ffill').fillna(method='bfill').fillna(0)
    data['TOTAL TURNOVER in mkd'] = data['TOTAL TURNOVER in mkd'].fillna(method='ffill').fillna(method='bfill').fillna(0)

    data['AVERAGE PRICE'] = safe_convert_to_float(data['AVERAGE PRICE'])
    data['PRICE OF LAST TRANSACTION IN mkd'] = safe_convert_to_float(data['PRICE OF LAST TRANSACTION IN mkd'])
    data['MIN'] = safe_convert_to_float(data['MIN'])
    data['MAX'] = safe_convert_to_float(data['MAX'])
    data['QUANTITY'] = safe_convert_to_float(data['QUANTITY'])
    data['%CHANGE'] = safe_convert_to_float(data['%CHANGE'])
    data['Turnover in BEST in mkd'] = safe_convert_to_float(data['Turnover in BEST in mkd'])
    data['TOTAL TURNOVER in mkd'] = safe_convert_to_float(data['TOTAL TURNOVER in mkd'])

    data['SMA_10'] = data['PRICE OF LAST TRANSACTION IN mkd'].rolling(window=10).mean().fillna(method='ffill').fillna(method='bfill').fillna(0)
    data['EMA_10'] = data['PRICE OF LAST TRANSACTION IN mkd'].ewm(span=10, adjust=False).mean().fillna(method='ffill').fillna(method='bfill').fillna(0)
    # Simple Moving Average (SMA) with a window of 20
    data['SMA_20'] = data['PRICE OF LAST TRANSACTION IN mkd'].rolling(window=20).mean().fillna(method='ffill').fillna(
        method='bfill').fillna(0)

    # Exponential Moving Average (EMA) with a span of 20
    data['EMA_20'] = data['PRICE OF LAST TRANSACTION IN mkd'].ewm(span=20, adjust=False).mean().fillna(
        method='ffill').fillna(method='bfill').fillna(0)

    # Weighted Moving Average (WMA) with a window of 10
    weights = np.arange(1, 11)  # Create weights from 1 to 10
    data['WMA_10'] = data['PRICE OF LAST TRANSACTION IN mkd'].rolling(window=10).apply(
        lambda prices: np.dot(prices, weights) / weights.sum(), raw=True).fillna(method='ffill').fillna(
        method='bfill').fillna(0)

    data['RSI'] = ta.momentum.RSIIndicator(data['PRICE OF LAST TRANSACTION IN mkd'], window=14).rsi().fillna(method='ffill').fillna(method='bfill').fillna(0)
    cci = CCIIndicator(high=data['MAX'], low=data['MIN'], close=data['PRICE OF LAST TRANSACTION IN mkd'], window=20)
    data['CCI'] = cci.cci().fillna(method='ffill').fillna(method='bfill').fillna(0)
    stoch = StochasticOscillator(high=data['MAX'], low=data['MIN'], close=data['PRICE OF LAST TRANSACTION IN mkd'], window=14, smooth_window=3)
    data['%K'] = stoch.stoch().fillna(method='ffill').fillna(method='bfill').fillna(0)
    data['%D'] = stoch.stoch_signal().fillna(method='ffill').fillna(method='bfill').fillna(0)

    data['Signal'] = data['RSI'].apply(lambda x: 'Buy' if x < 30 else 'Sell' if x > 70 else 'Hold')
    data['CCI_Signal'] = data['CCI'].apply(lambda x: 'Buy' if x < -100 else 'Sell' if x > 100 else 'Hold')
    data['Stochastic_Signal'] = data['%K'].apply(lambda x: 'Buy' if x < 20 else 'Sell' if x > 80 else 'Hold')

    data['Signal'].dropna(inplace=True)
    data['CCI_Signal'].dropna(inplace=True)
    data['Stochastic_Signal'].dropna(inplace=True)

    return data


def prepare_lstm_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['PRICE OF LAST TRANSACTION IN mkd'].values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.7)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    def create_sequences(data, sequence_length=5):
        x, y = [], []
        for i in range(len(data) - sequence_length):
            x.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(x), np.array(y)

    x_train, y_train = create_sequences(train_data)
    x_test, y_test = create_sequences(test_data)

    return x_train, y_train, x_test, y_test, scaler


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(30, return_sequences=False),
        Dropout(0.2),
        Dense(10, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model


# def plot_predictions(y_test, predictions, scaler, stock_symbol):
#     y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
#     predictions_unscaled = scaler.inverse_transform(predictions)
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(y_test_unscaled, color='blue', label='Actual Prices')
#     plt.plot(predictions_unscaled, color='red', label='Predicted Prices')
#     plt.title(f'Stock Price Prediction for {stock_symbol}')
#     plt.xlabel('Time')
#     plt.ylabel('Price')
#     plt.legend()
#     plt.savefig(f"data/lstm_plot_{stock_symbol}.png")
#     plt.close()


@app.route("/api/data/<string:issuer_code>", methods=["GET"])
def get_issuer_data(issuer_code):
    to_path = f"demo/src/main/resources/technical_analysis_{issuer_code}.csv"
    file_path = f"data/technical_analysis_{issuer_code}.csv"
    if not os.path.exists(file_path):
        return abort(404, description=f"Data for issuer code {issuer_code} not found.")

    df = pd.read_csv(file_path)
    df.to_csv(to_path, index=False)
    return df.to_json(orient="records")

@app.route("/api/companies", methods=["GET"])
def get_company_codes():
    codes = fetch_issuer_codes("https://www.mse.mk/mk/stats/symbolhistory/ADIN")
    return jsonify(codes)

@app.route("/api/predict/<string:issuer_code>", methods=["GET"])
def get_prediction(issuer_code):
    to_path = f"demo/src/main/resources/lstm_predictions_{issuer_code}.csv"
    file_path = f"data/technical_analysis_{issuer_code}.csv"
    data = pd.read_csv(file_path)
    x_train, y_train, x_test, y_test, scaler = prepare_lstm_data(data)
    model = build_lstm_model((x_train.shape[1], 1))
    model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1,
              callbacks=[EarlyStopping(monitor='loss', patience=5)])
    predictions = model.predict(x_test)
    predictions_unscaled = scaler.inverse_transform(predictions)
    results = pd.DataFrame({
    'Actual Prices': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
    'Predicted Prices': predictions_unscaled.flatten()
    })
    results.to_csv(f"data/lstm_predictions_{issuer_code}.csv", index=False)
    df = pd.read_csv(f"data/lstm_predictions_{issuer_code}.csv")
    df.to_csv(to_path, index=False)
    return df.to_json(orient="records")

def main():

    base_url = "https://www.mse.mk/mk/stats/symbolhistory/ADIN"

    if not os.path.exists("data"):
        os.mkdir("data")

    print("Fetching issuer codes...")
    issuer_codes = fetch_issuer_codes(base_url)
    print(f"Found {len(issuer_codes)} issuer codes.")

    for valid_code in issuer_codes:
        print(f"Processing data for: {valid_code}...")
        file_path = f"data/technical_analysis_{valid_code}.csv"

        # If data doesn't exist, extract and save it
        if not os.path.exists(file_path):
            stock_data = extract_issuer_data(valid_code)
            if stock_data.empty:
                print(f"No data available for {valid_code}. Skipping.")
                continue
            stock_data.to_csv(file_path, index=False)
            print(f"Data saved for {valid_code}.\n")

        # Train and predict using LSTM
        # print(f"Training LSTM model for: {valid_code}...")
        # data = pd.read_csv(file_path)
        # get_prediction(valid_code)
        # try:
            # x_train, y_train, x_test, y_test, scaler = prepare_lstm_data(data)
            # model = build_lstm_model((x_train.shape[1], 1))
            # model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1,
            #           callbacks=[EarlyStopping(monitor='loss', patience=5)])
            # predictions = model.predict(x_test)
            #
            # # Save results and plot
            # predictions_unscaled = scaler.inverse_transform(predictions)
            # results = pd.DataFrame({
            #     'Actual Prices': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
            #     'Predicted Prices': predictions_unscaled.flatten()
            # })
            # results.to_csv(f"data/lstm_predictions_{valid_code}.csv", index=False)
        #     plot_predictions(y_test, predictions, scaler, valid_code)
        #     print(f"Prediction results saved for {valid_code}.\n")
        # except Exception as e:
        #     print(f"Error processing {valid_code}: {e}")
        #     continue
    print("Processing completed for all issuer codes.")
    for code in issuer_codes:
        get_issuer_data(code)

    # for code in issuer_codes:
    #     get_prediction(code)

if __name__ == "__main__":
    main()
    app.run(host='0.0.0.0', port=5000)