"""Backtest the saved portfolios against the paper's baselines and regional indices.

Usage: python -m portfolio_optimization.backtest [preset]
Prints the results table and writes results/<DATASET>/backtest_<preset>.csv.
"""
import ast
import os
import random
import sys

import numpy as np
import pandas as pd

from config import PRICES_CSV, RESULTS_DIR, TEST_END, TEST_START, UNIVERSE_CSV

REGIONS = {'US': {'NMS', 'NYQ', 'BTS'}, 'Europe': {'AMS', 'BRU', 'GER', 'HEL', 'MCE', 'MIL', 'PAR'},
           'Crypto': {'BINANCE', 'COINBASE', 'INDEX'}, 'China': {'SHH', 'SHZ'}, 'Commodities': {'World'}, 'BIST': {'IST'}}
HOLD_US = ['NASDAQ:AAPL', 'NASDAQ:MSFT', 'NASDAQ:NVDA', 'NASDAQ:AMZN', 'NYSE:JPM']
HOLD_BIST = ['BIST:AKBNK', 'BIST:SISE', 'BIST:THYAO', 'BIST:KCHOL', 'BIST:TUPRS']


def load_portfolio(path):
    """Read a 3-row portfolio CSV into {date: (tickers, weights)}."""
    df = pd.read_csv(path)
    df['List'] = df.List.apply(ast.literal_eval)
    out = {}
    for date, g in df.groupby('Date'):
        rows = dict(zip(g.Type.str.lower(), g.List))
        out[pd.to_datetime(date)] = (rows['ticker'], rows['weight'])
    return out


def filter_available_stocks(tickers, start, end, pivot):
    """Tickers with at least one price after start and up to end."""
    window = pivot.loc[start:end]
    has_price = window[window.index > start].notna().any()
    return [t for t in tickers if has_price.get(t, False)]


def compute_portfolio_returns(history, prices):
    """Hold each rebalance's weights from the first price after the rebalance day to the next rebalance day."""
    recs, dates = [], sorted(history)
    for i in range(len(dates) - 1):
        day, next_day = dates[i], dates[i + 1]
        entry = history[day]
        tickers, weights = entry if isinstance(entry, tuple) else (entry, [1 / len(entry)] * len(entry))
        period = prices.loc[day:next_day, tickers].dropna(axis=1, how='all')
        if period.shape[0] < 2:
            continue
        rets, used = [], []
        for t in period.columns:
            s = period[t].dropna()
            s = s[s.index > day]
            if s.shape[0] < 2:
                continue
            rets.append(s.iloc[-1] / s.iloc[0] - 1)
            used.append(weights[tickers.index(t)])
        if not rets:
            continue
        w = np.array(used)
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        recs.append({'end_date': pd.Timestamp(next_day).normalize(), 'period_ret': np.dot(w, rets)})
    df = pd.DataFrame(recs, columns=['end_date', 'period_ret'])
    df['cumulative_return'] = (1 + df.period_ret).cumprod()
    return df


def average_runs(dfs):
    """Average period and cumulative returns of several runs aligned on end_date."""
    m = pd.concat([d.set_index('end_date') for d in dfs], axis=1, sort=True)
    return pd.DataFrame({'period_ret': m.period_ret.mean(axis=1),
                         'cumulative_return': m.cumulative_return.mean(axis=1)}).rename_axis('end_date').reset_index()


def top_volume(combined, tickers, day):
    """Ten tickers with the largest rolling 21-day USD volume on or before day."""
    snap = (combined[combined.ticker.isin(tickers)].loc[combined.date <= day]
            .sort_values(['ticker', 'date']).groupby('ticker').last().reset_index()
            .dropna(subset=['Rolling_Monthly_USD_Volume']))
    return snap.nlargest(10, 'Rolling_Monthly_USD_Volume').ticker.tolist() if len(snap) >= 10 else None


def summarize(df, periods_per_year=12):
    """Final return, std of period returns, sqrt(periods_per_year)-annualized Sharpe and max drawdown (%)."""
    vol = df.period_ret.std(ddof=0)
    peak = df.cumulative_return.cummax()
    return {'Final Return': df.cumulative_return.iloc[-1], 'Volatility': vol,
            'Sharpe': df.period_ret.mean() / vol * np.sqrt(periods_per_year) if vol > 0 else np.nan,
            'Max Drawdown (%)': ((df.cumulative_return - peak) / peak).min() * 100, 'Num Periods': len(df)}


def main():
    """Run every strategy and baseline over the test window and print the table."""
    preset = sys.argv[1] if len(sys.argv) > 1 else 'paper'
    combined = pd.read_csv(PRICES_CSV, parse_dates=['date']).rename(columns={'volume': 'volume_org'})
    combined = combined[(combined.date >= TEST_START) & (combined.date <= TEST_END)]
    prices = combined.pivot(index='date', columns='ticker', values='close_usd').ffill(limit=10)

    combined = combined.merge(pd.read_csv(UNIVERSE_CSV), on='ticker', how='left').sort_values(['ticker', 'date'])
    combined['adj_volume'] = combined.volume_org * combined.sector.map({'Commodities': 100, 'Crypto': 20}).fillna(1)
    combined['Rolling_Monthly_USD_Volume'] = (combined.close_usd * combined.adj_volume).groupby(combined.ticker) \
        .transform(lambda x: x.rolling(window=21, min_periods=1).sum()).round(0)

    curves = {}
    for name in ['clustered', 'sector', 'top_n']:
        path = f'{RESULTS_DIR}/portfolios/{preset}/{name}.csv'
        if os.path.exists(path):
            curves[name] = compute_portfolio_returns(load_portfolio(path), prices)
    dates = sorted(load_portfolio(f'{RESULTS_DIR}/portfolios/{preset}/top_n.csv'))
    static = lambda tickers: compute_portfolio_returns({d: tickers for d in dates}, prices)

    curves['Equal Weighted'] = static(prices.columns.tolist())
    runs = []
    for seed in range(5):
        random.seed(seed)
        hist = {}
        for d1, d2 in zip(dates[:-1], dates[1:]):
            valid = filter_available_stocks(prices.columns, d1, d2, prices)
            if len(valid) >= 10:
                hist[d1] = random.sample(valid, 10)
        runs.append(compute_portfolio_returns(hist, prices))
    curves['Random 10'] = average_runs(runs)

    ultimate = {}
    for s, e in zip(dates[:-1], dates[1:]):
        window = prices.loc[s:e].dropna(axis=1, how='all')
        rets = (window.iloc[-1] / window.iloc[0] - 1).to_dict()
        ultimate[s] = sorted(rets, key=rets.get, reverse=True)[:10]
    curves['Ultimate 10'] = compute_portfolio_returns(ultimate, prices)

    monthly = {}
    for d1, d2 in zip(dates[:-1], dates[1:]):
        d1, d2 = pd.to_datetime(d1).normalize(), pd.to_datetime(d2).normalize()
        top = top_volume(combined, filter_available_stocks(prices.columns, d1, d2, prices), d1)
        if top:
            monthly[d1] = top
    curves['Top 10 Volume (Monthly)'] = compute_portfolio_returns(monthly, prices)

    annual = {}
    for s in dates[::12]:
        s = pd.to_datetime(s).normalize()
        top = top_volume(combined, filter_available_stocks(prices.columns, s, s + pd.DateOffset(years=1), prices), s)
        if top:
            annual[s] = top
    curves['Top 10 Volume (Annual)'] = compute_portfolio_returns(annual, prices)

    curves['Gold (XAU:USD)'] = static(['OANDA:XAUUSD'])
    curves['Hold US Stocks'] = static(HOLD_US)
    curves['Hold BIST Stocks'] = static(HOLD_BIST)

    exchange = combined.drop_duplicates('ticker').set_index('ticker').exchange
    for region, codes in REGIONS.items():
        df = static([t for t in prices.columns if exchange.get(t) in codes])
        df['period_ret'] = df.cumulative_return.pct_change().fillna(0)  # first period's return set to 0, as in the paper's results
        curves[f'{region} Index'] = df

    table = pd.DataFrame({name: summarize(df, 1 if 'Annual' in name else 12) for name, df in curves.items()}).T.sort_values('Final Return', ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    table.to_csv(f'{RESULTS_DIR}/backtest_{preset}.csv')
    pd.concat({k: v.set_index('end_date').cumulative_return for k, v in curves.items()}, axis=1, sort=True) \
        .to_csv(f'{RESULTS_DIR}/cumulative_{preset}.csv')
    with pd.option_context('display.float_format', '{:.4g}'.format, 'display.width', 120, 'display.max_columns', None):
        print(table)


if __name__ == '__main__':
    main()
