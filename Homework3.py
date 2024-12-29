import os
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import matplotlib.pyplot as plt
from keras.src.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from keras.src.models import Sequential
from keras.src.layers import Dense, LSTM, Dropout
from keras.src.callbacks import EarlyStopping
import ta
from ta.momentum import StochasticOscillator
from ta.trend import CCIIndicator

base_url = "https://www.mse.mk/mk/stats/symbolhistory/ADIN"


def process_technical_analysis(valid_code):
    matrix = []
    today = datetime.today()
    current_year = today.year
    for i in range(11):
        year = current_year - i
        response = requests.post(
            f"https://www.mse.mk/mk/stats/symbolhistory/{valid_code}",
            data={"FromDate": f"01.01.{year}", "ToDate": f"31.12.{year}"}
        )
        soup = BeautifulSoup(response.text, 'html.parser')
        rows = soup.select("tbody tr")
        for row in rows:
            cols = row.select("td")
            if len(cols) >= 9:
                matrix.append({
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

    data = pd.DataFrame(matrix)
    data['DATE'] = pd.to_datetime(data['DATE'], format='%d.%m.%Y')
    data.sort_values('DATE', inplace=True)

    data['PRICE OF LAST TRANSACTION IN mkd'] = data['PRICE OF LAST TRANSACTION IN mkd'].replace('', pd.NA).fillna(method='ffill').fillna(method = 'bfill').fillna(0)
    data['MIN'] = data['MIN'].replace('', pd.NA).fillna(data['PRICE OF LAST TRANSACTION IN mkd'])
    data['MAX'] = data['MAX'].replace('', pd.NA).fillna(data['PRICE OF LAST TRANSACTION IN mkd'])

    def safe_convert_to_float(column):
        if column.dtype == 'O':
            return column.str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
        return column

    data['PRICE OF LAST TRANSACTION IN mkd'] = safe_convert_to_float(data['PRICE OF LAST TRANSACTION IN mkd'])
    data['MIN'] = safe_convert_to_float(data['MIN'])
    data['MAX'] = safe_convert_to_float(data['MAX'])

    data['SMA_10'] = data['PRICE OF LAST TRANSACTION IN mkd'].rolling(window=10).mean()
    data['EMA_10'] = data['PRICE OF LAST TRANSACTION IN mkd'].ewm(span=10, adjust=False).mean()
    data['RSI'] = ta.momentum.RSIIndicator(data['PRICE OF LAST TRANSACTION IN mkd'], window=14).rsi()
    cci = CCIIndicator(high=data['MAX'], low=data['MIN'], close=data['PRICE OF LAST TRANSACTION IN mkd'], window=20)
    data['CCI'] = cci.cci()
    stoch = StochasticOscillator(high=data['MAX'], low=data['MIN'], close=data['PRICE OF LAST TRANSACTION IN mkd'],
                                 window=14, smooth_window=3)
    data['%K'] = stoch.stoch()
    data['%D'] = stoch.stoch_signal()

    data['Signal'] = data['RSI'].apply(lambda x: 'Buy' if x < 30 else 'Sell' if x > 70 else 'Hold')
    data['CCI_Signal'] = data['CCI'].apply(lambda x: 'Buy' if x < -100 else 'Sell' if x > 100 else 'Hold')
    data['Stochastic_Signal'] = data['%K'].apply(lambda x: 'Buy' if x < 20 else 'Sell' if x > 80 else 'Hold')

    return data


def prepare_data(file_path):

    data = pd.read_csv(file_path)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['PRICE OF LAST TRANSACTION IN mkd'].values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.7)  # 70% for training
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

    return x_train, y_train, x_test, y_test, scaler, data


def build_optimized_lstm_model(input_shape):

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

def plot_predictions(y_test, predictions, scaler, stock_symbol):

    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_unscaled = scaler.inverse_transform(predictions)

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_unscaled, color='blue', label='Actual Prices')
    plt.plot(predictions_unscaled, color='red', label='Predicted Prices')
    plt.title(f'Stock Price Prediction for {stock_symbol}')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.savefig(f"data/lstm_plot_{stock_symbol}.png")
    plt.close()


def main():
    if not os.path.exists("data"):
        os.mkdir("data")

    if not os.listdir("data"):
        print("The 'data' folder is empty. Performing technical analysis...")

        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        options = soup.select(".panel .form-control option")

        for option in options:
            valid_code = option["value"].strip()
            if valid_code and not any(char.isdigit() for char in valid_code):
                print(f"Processing data for: {valid_code}...")

                stock_data = process_technical_analysis(valid_code)
                stock_data.to_csv(f"data/technical_analysis_{valid_code}.csv", index=False)
                print(f"Technical analysis completed for {valid_code}.\n")

    else:
        print("The 'data' folder is not empty. Proceeding with LSTM model training and predictions...")

        response = requests.get(base_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        options = soup.select(".panel .form-control option")

        for option in options:
            valid_code = option["value"].strip()
            if valid_code and not any(char.isdigit() for char in valid_code):
                print(f"Training and predicting LSTM model for: {valid_code}...")

                file_path = f"data/technical_analysis_{valid_code}.csv"
                x_train, y_train, x_test, y_test, scaler, _ = prepare_data(file_path)


                model = build_optimized_lstm_model((x_train.shape[1], 1))
                model.fit(x_train, y_train, batch_size=64, epochs=50, verbose=1,
                          callbacks=[EarlyStopping(monitor='loss', patience=5)])

                predictions = model.predict(x_test)


                predictions_unscaled = scaler.inverse_transform(predictions)
                results = pd.DataFrame({
                    'Actual Prices': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                    'Predicted Prices': predictions_unscaled.flatten()
                })
                results.to_csv(f"data/lstm_predictions_{valid_code}.csv", index=False)
                print(f"Predictions saved for {valid_code}.\n")

                plot_predictions(y_test, predictions, scaler, valid_code)

if __name__ == "__main__":
    main()