"""Fetch daily OHLCV from TradingView for every universe ticker and add USD-converted open/close.

Tickers whose file already exists are skipped, so a rerun only retries the ones in failed.txt.
"""
import logging
import os

import pandas as pd
from tvDatafeed import Interval, TvDatafeed

from config import FX_CSV, RAW_DIR, UNIVERSE_CSV

N_BARS = 20000
# yfinance exchange code -> FX column; every other exchange (US, crypto, commodities) already quotes in USD.
EXCHANGE_FX = {'IST': 'USDTRY', 'SHH': 'USDCNY', 'SHZ': 'USDCNY', 'AMS': 'EURUSD', 'MIL': 'EURUSD',
               'PAR': 'EURUSD', 'GER': 'EURUSD', 'MCE': 'EURUSD', 'HEL': 'EURUSD', 'BRU': 'EURUSD'}

logging.getLogger('tvDatafeed.main').setLevel(logging.CRITICAL)
os.makedirs(RAW_DIR, exist_ok=True)
universe = pd.read_csv(UNIVERSE_CSV)
fx = pd.read_csv(FX_CSV, index_col='date', parse_dates=True)
failed_path = f'{RAW_DIR}/failed.txt'
open(failed_path, 'w').close()
tv = TvDatafeed()

for i, (ticker, exchange) in enumerate(zip(universe.ticker, universe.exchange), 1):
    path = f"{RAW_DIR}/{ticker.replace(':', '_')}.csv"
    if os.path.exists(path):
        continue
    ex, symbol = ticker.split(':', 1)
    df = tv.get_hist(symbol=symbol, exchange=ex, interval=Interval.in_daily, n_bars=N_BARS)
    if df is None or df.empty:
        print(f'[{i}/{len(universe)}] {ticker}: no data')
        open(failed_path, 'a').write(ticker + '\n')
        continue

    # Bars come as UTC; futures/FX daily bars open the evening before, so the date is taken in
    # Istanbul time as the paper-era library did (it used the local clock of the machine).
    df.index = df.index.tz_localize('UTC').tz_convert('Europe/Istanbul').tz_localize(None).normalize()
    pair = EXCHANGE_FX.get(exchange)
    if pair:
        df['currency_price'] = fx[pair].reindex(df.index)
    if pair == 'EURUSD':
        df['open_usd'], df['close_usd'] = df.open * df.currency_price, df.close * df.currency_price
    elif pair:
        df['open_usd'], df['close_usd'] = df.open / df.currency_price, df.close / df.currency_price
    else:
        df['open_usd'], df['close_usd'] = df.open, df.close
    df['currency_pair_used'] = pair or 'USD'
    df.index.name = 'date'
    df.to_csv(path)
    print(f'[{i}/{len(universe)}] {ticker}: {len(df)} rows')
