import requests
import pandas as pd
import time
import numpy as np


def get_klines(symbol, interval, limit):
    url = f'https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data)
    df.columns = ['Open time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close time', 'Quote asset volume',
                  'Number of trades', 'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore']
    df['Open time'] = pd.to_datetime(df['Open time'], unit='ms')
    df['Close time'] = pd.to_datetime(df['Close time'], unit='ms')
    df.set_index('Close time', inplace=True)
    df.drop(columns=['Open time', 'Ignore'], inplace=True)
    df = df.astype(float)
    return df


def get_eth_price():
    url = 'https://fapi.binance.com/fapi/v1/ticker/price?symbol=ETHUSDT'
    response = requests.get(url)
    data = response.json()
    return float(data['price'])


# Загрузка исторических данных по ценам BTCUSDT и ETHUSDT
btc_data = get_klines('BTCUSDT', '1m', 60)
eth_data = get_klines('ETHUSDT', '1m', 60)

# Выборка последних 60 минут цен ETHUSDT и BTCUSDT
eth_prices = eth_data['Close'].tail(60).values
btc_prices = btc_data['Close'].tail(60).values

# Формирование матрицы признаков X и вектора целевой переменной Y
X = btc_prices.reshape(-1, 1)
Y = eth_prices.reshape(-1, 1)

# Выбор оптимальных параметров модели регрессии с помощью МНК
X = np.hstack((X, np.ones((X.shape[0], 1))))
coeffs = np.linalg.lstsq(X, Y, rcond=None)[0]
regressor = lambda x: np.dot(np.hstack((x, np.ones((x.shape[0], 1)))), coeffs)

# Бесконечный цикл, который следит за текущей ценой фьючерса ETHUSDT
previous_predicted_eth_price = None
while True:
    eth_price = get_eth_price()

    # Вычисление предсказанной цены ETHUSDT без учета влияния BTCUSDT
    predicted_eth_price = regressor(np.array([[eth_price]]))[0][0]

    # Вычисление движения цены ETH
    eth_movement = (predicted_eth_price - previous_predicted_eth_price) / previous_predicted_eth_price if previous_predicted_eth_price else 0
    previous_predicted_eth_price = predicted_eth_price

    # Если движение превышает 1%, выводим сообщение в консоль
    if abs(eth_movement) > 0.01:
        print(f'ETH movement: {eth_movement * 100:.2f}%')

    # Ждем 5 секунд перед повторным запросом
    time.sleep(5)
