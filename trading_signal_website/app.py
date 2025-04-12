
from flask import Flask, render_template, jsonify
import ccxt
import pandas as pd
import time
import numpy as np
import gc

app = Flask(__name__)

exchange = ccxt.binance({ 'enableRateLimit': True })

def fetch_ohlcv(symbol, timeframe, since=None, limit=1000):
    try:
        return exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
    except Exception as e:
        print(f"An error occurred while fetching OHLCV: {e}")
        return []

def get_signals(symbol, timeframe='1m', num_bars=1000, trend_windows=None):
    if trend_windows is None:
        trend_windows = [100, 150]

    all_data = []
    limit = 1000
    total_points = num_bars
    end_time = exchange.milliseconds()

    tf_map = {'m': 60 * 1000, 'h': 60 * 60 * 1000, 'd': 24 * 60 * 60 * 1000}
    candle_ms = tf_map.get(timeframe[-1], 60 * 1000) * int(timeframe[:-1])

    while len(all_data) < total_points:
        since_time = end_time - limit * candle_ms
        data = fetch_ohlcv(symbol, timeframe, since=since_time, limit=limit)
        if not data:
            break
        all_data = data + all_data
        end_time = data[0][0]
        time.sleep(exchange.rateLimit / 1000.0)

    all_data = all_data[-total_points:]
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df['median_price'] = df['close'].rolling(window=3, center=True).median().bfill().ffill()

    df['EMA_very_fast'] = df['median_price'].ewm(span=3, adjust=False).mean()
    df['EMA_fast'] = df['median_price'].ewm(span=8, adjust=False).mean()
    df['EMA_slow'] = df['median_price'].ewm(span=22, adjust=False).mean()
    df['MA_slow'] = df['median_price'].rolling(window=22).mean()

    for window in trend_windows:
        df[f'Trend_{window}'] = df['close'].rolling(window=window).mean()

    df['EMA_fast_prev'] = df['EMA_fast'].shift(1)
    df['MA_slow_prev'] = df['MA_slow'].shift(1)
    df['signal'] = 0
    buy_signals, sell_signals = [], []
    pending_signal, pending_trends = None, None

    for i in range(1, len(df)):
        if pd.isna(df['EMA_fast_prev'].iloc[i]) or pd.isna(df['MA_slow_prev'].iloc[i]):
            continue

        trends = [1 if df[f'Trend_{w}'].iloc[i] > df[f'Trend_{w}'].iloc[i - 1] else -1 for w in trend_windows if not pd.isna(df[f'Trend_{w}'].iloc[i])]
        if len(trends) != len(trend_windows):
            continue

        buy = df['EMA_fast_prev'].iloc[i] <= df['MA_slow_prev'].iloc[i] and df['EMA_fast'].iloc[i] > df['MA_slow'].iloc[i]
        sell = df['EMA_fast_prev'].iloc[i] >= df['MA_slow_prev'].iloc[i] and df['EMA_fast'].iloc[i] < df['MA_slow'].iloc[i]
        intended = 1 if buy else -1 if sell else None

        if intended:
            if all(t == intended for t in trends):
                df.iat[i, df.columns.get_loc('signal')] = intended
                (buy_signals if intended == 1 else sell_signals).append((df.index[i], df['low' if intended == 1 else 'high'].iloc[i]))
                pending_signal, pending_trends = None, None
            else:
                pending_signal, pending_trends = intended, trends

        if pending_signal and all(t == pending_signal for t in trends) and any(pt != pending_signal for pt in pending_trends):
            df.iat[i, df.columns.get_loc('signal')] = pending_signal
            (buy_signals if pending_signal == 1 else sell_signals).append((df.index[i], df['low' if pending_signal == 1 else 'high'].iloc[i]))
            pending_signal, pending_trends = None, None

    result = {
        "ohlc": [[int(idx.timestamp() * 1000), row['open'], row['high'], row['low'], row['close']] for idx, row in df.iterrows()],
        "buy_signals": [[int(ts.timestamp() * 1000), price] for ts, price in buy_signals],
        "sell_signals": [[int(ts.timestamp() * 1000), price] for ts, price in sell_signals],
        "ema_very_fast": [[int(ts.timestamp() * 1000), val] for ts, val in df['EMA_very_fast'].dropna().items()],
        "ema_fast": [[int(ts.timestamp() * 1000), val] for ts, val in df['EMA_fast'].dropna().items()],
        "ema_slow": [[int(ts.timestamp() * 1000), val] for ts, val in df['EMA_slow'].dropna().items()],
        "ma_slow": [[int(ts.timestamp() * 1000), val] for ts, val in df['MA_slow'].dropna().items()],
        "trend_100": [[int(ts.timestamp() * 1000), val] for ts, val in df['Trend_100'].dropna().items()],
        "trend_150": [[int(ts.timestamp() * 1000), val] for ts, val in df['Trend_150'].dropna().items()]
    }

    return result

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/data_json")
def data_json():
    symbol = "BTC/USDT"
    chart_data = get_signals(symbol, timeframe="1m", num_bars=500, trend_windows=[100, 150])
    return jsonify(chart_data)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
    gc.collect()
