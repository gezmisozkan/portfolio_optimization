"""Download USDTRY/USDCNY/EURUSD from Yahoo and build a daily FX table for the USD conversion."""
import os

import pandas as pd
import yfinance as yf

from config import FX_CSV

PAIRS = {'USDTRY': 'USDTRY=X', 'USDCNY': 'USDCNY=X', 'EURUSD': 'EURUSD=X'}
# Hand-checked rates for years Yahoo does not cover; days in between are time-interpolated.
EARLY_POINTS = {
    'USDTRY': {'1989-11-08': 0.0023, '1994-01-14': 0.0156, '1998-08-25': 0.27730, '2001-02-21': 0.68730, '2004-12-31': 1.3393},
    'USDCNY': {'1981-01-01': 1.5341, '1985-11-01': 3.2095, '1986-07-03': 3.2063, '1986-07-07': 3.7129,
               '1989-12-13': 3.7314, '1989-12-18': 4.7339, '1990-11-15': 4.7339, '1990-11-19': 5.2352,
               '1993-12-31': 5.8145, '1994-01-03': 8.7217, '2001-06-01': 8.276},
    'EURUSD': {'1988-03-23': 1.156, '1994-01-01': 1.1147, '1997-01-01': 1.279, '1998-07-01': 1.095, '2000-01-01': 1.0074},
}

real = pd.DataFrame({name: yf.download(t, period='max', progress=False, auto_adjust=True)['Close'][t] for name, t in PAIRS.items()})
early = pd.DataFrame(EARLY_POINTS)
early.index = pd.to_datetime(early.index)

fx = pd.concat([early, real]).sort_index()
fx = fx[~fx.index.duplicated(keep='last')]
fx = fx.reindex(pd.date_range(fx.index.min(), fx.index.max(), freq='D')).interpolate(method='time').ffill().bfill()
fx.index.name = 'date'
os.makedirs(os.path.dirname(FX_CSV), exist_ok=True)
fx.to_csv(FX_CSV)
print(f'{FX_CSV}: {len(fx)} days, {fx.index.min().date()} -> {fx.index.max().date()}')
