"""Combine the raw per-ticker files into the long price table and the daily-return pivot used by the models."""
import glob

import pandas as pd

from config import DATA_END, DATA_START, PRICES_CSV, RAW_DIR, RETURNS_PIVOT_CSV, UNIVERSE_CSV

COLUMNS = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'currency_price',
           'open_usd', 'close_usd', 'currency_pair_used']


def fold_crypto_weekends(df):
    """Merge Sat/Sun bars into the preceding Friday (Friday open, weekend high/low/close, summed volume); weekends without a Friday bar are dropped."""
    friday = df.date - pd.to_timedelta((df.date.dt.day_of_week - 4).clip(lower=0), unit='D')
    keep = friday.isin(df.date)
    df = df.assign(weekend_volume=df.volume.where(df.date.dt.day_of_week >= 5, 0))
    out = df[keep].groupby(friday[keep]).agg(
        symbol=('symbol', 'first'), open=('open', 'first'), high=('high', 'max'), low=('low', 'min'),
        close=('close', 'last'), volume=('volume', 'first'), weekend_volume=('weekend_volume', 'sum'),
        currency_pair_used=('currency_pair_used', 'first'))
    out['volume'] += out.pop('weekend_volume')  # Friday + (Sat + Sun)
    out['open_usd'], out['close_usd'] = out.open, out.close
    return out.rename_axis('date').reset_index()


crypto = set(pd.read_csv(UNIVERSE_CSV).query("sector == 'Crypto'").ticker)
frames = []
for path in sorted(glob.glob(f'{RAW_DIR}/*.csv')):
    df = pd.read_csv(path, parse_dates=['date'])
    df = df[df.date <= DATA_END].dropna(subset=['close_usd']).drop_duplicates('date').sort_values('date')
    if df.empty:
        continue
    if df.symbol.iloc[0] in crypto:
        df = fold_crypto_weekends(df)
    df['return'] = df.close_usd.pct_change()
    frames.append(df[df.date >= DATA_START].reindex(columns=COLUMNS + ['return']))

prices = pd.concat(frames, ignore_index=True).rename(columns={'symbol': 'ticker'})
prices.to_csv(PRICES_CSV, index=False, date_format='%Y-%m-%d')
pivot = prices.pivot_table(index='date', columns='ticker', values='return')
pivot.to_csv(RETURNS_PIVOT_CSV, date_format='%Y-%m-%d')
print(f'{PRICES_CSV}: {len(prices)} rows, {prices.ticker.nunique()} tickers, {prices.date.min().date()} -> {prices.date.max().date()}')
print(f'{RETURNS_PIVOT_CSV}: {pivot.shape[0]} days x {pivot.shape[1]} tickers')
