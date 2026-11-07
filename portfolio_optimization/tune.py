"""Tune one strategy with Optuna on the development period (Jan 2010 - Dec 2017), as in the paper.

Usage: python -m portfolio_optimization.tune <clustered|sector|top_n> [n_trials, default 500]
Saves every trial to results/<DATASET>/optuna_<strategy>.csv and prints the top-50 average as a PRESETS entry.
No price after DEV_END is read.
"""
import functools
import os
import sys

import optuna
import pandas as pd

from config import RESULTS_DIR, RETURNS_PIVOT_CSV, UNIVERSE_CSV
from portfolio_optimization.backtest import compute_portfolio_returns
from portfolio_optimization.build_portfolios import build_factor_table, load_clusters, load_prices, quality_on, select
from stock_embeddings.utils.returns_data_class import ReturnsData

DEV_START, DEV_END = '2010-01-01', '2018-01-01'
TOP_TRIALS = 50

strategy = sys.argv[1]
n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 500

prices = load_prices()
prices = prices[prices.date < DEV_END]
fac = build_factor_table(prices[prices.date >= pd.Timestamp(DEV_START) - pd.DateOffset(days=400)].copy())
fac = fac[fac.date >= DEV_START]
first_dates = prices.groupby('ticker').date.min()
sectors = prices.groupby('ticker').sector.last().to_dict()
tickers = ReturnsData(RETURNS_PIVOT_CSV, UNIVERSE_CSV).tickers
pivot = prices.pivot(index='date', columns='ticker', values='close_usd').ffill(limit=10)
dates = fac.date.dt.to_period('M').drop_duplicates().dt.to_timestamp(how='end').tolist()
months = {dt: fac[(fac.date >= dt.replace(day=1)) & (fac.date <= dt)].groupby('ticker').tail(1) for dt in dates}
clusters = functools.lru_cache(None)(lambda dt, k: load_clusters(dt, k, tickers, first_dates))


def objective(trial):
    """Final cumulative return over the development rebalances for the trial's parameters."""
    w = {f: trial.suggest_float(f, 0, 1) for f in ('w_mom', 'w_vol', 'w_liq')}
    p = {f: v / sum(w.values()) for f, v in w.items()} | {'n_stocks': trial.suggest_int('n_stocks', 5, 50),
         'weight_method': trial.suggest_categorical('weight_method', ['equal', 'quality', 'risk_parity'])}
    if strategy != 'top_n':
        p['pick_method'] = trial.suggest_categorical('pick_method', ['fixed', 'dynamic'])
        if p['pick_method'] == 'fixed':
            p['per_group'] = trial.suggest_int('per_group', 1, 5)
    if strategy == 'clustered':
        p['n_clusters'] = trial.suggest_int('n_clusters', 5, 30)
    history = {}
    for dt in dates:
        q = quality_on(months[dt], dt, p)
        q = q[q.index.isin(first_dates[first_dates <= dt].index)]
        groups = clusters(dt, p['n_clusters']) if strategy == 'clustered' else sectors if strategy == 'sector' else None
        if groups is not None:  # ungrouped tickers are dropped before sizing the pool
            g = q.index.map(groups)
            q = q[g.notna() & (g != -1)]
        top = select(q, p, groups)
        if not top.empty:
            history[dt] = (top.Ticker.tolist(), top.weight.tolist())
    return compute_portfolio_returns(history, pivot).cumulative_return.iloc[-1]


os.makedirs(RESULTS_DIR, exist_ok=True)
out_csv = f'{RESULTS_DIR}/optuna_{strategy}.csv'
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=n_trials, callbacks=[lambda s, t: s.trials_dataframe().to_csv(out_csv, index=False)])

# the best trial is likely lucky, so the paper averages the top 50: numbers averaged (raw weights, then normalized),
# choices by majority; per_group is averaged over the top trials that used fixed picks
trials = study.trials_dataframe().nlargest(TOP_TRIALS, 'value').filter(like='params_').rename(columns=lambda c: c[7:])
top = trials.select_dtypes('number').mean().dropna()
weights = top[['w_mom', 'w_vol', 'w_liq']] / top[['w_mom', 'w_vol', 'w_liq']].sum()
preset = weights.to_dict() | {k: int(round(v)) for k, v in top.drop(weights.index).items()} | trials.select_dtypes(exclude='number').mode().iloc[0].to_dict()
print(f'best {study.best_value:.2f}x | top-{TOP_TRIALS} average:\n    {strategy!r}: dict({", ".join(f"{k}={v!r}" for k, v in preset.items())}),')
