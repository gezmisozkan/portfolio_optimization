"""Paths, dates and strategy presets shared by the pipeline scripts (run them from the repo root).

The DATASET environment variable picks the universe: unset = 'paper', DATASET=current = data/universe_current.csv.
"""
import os

DATASET = os.environ.get('DATASET', 'paper')
UNIVERSE_CSV = f'data/universe_{DATASET}.csv'
FX_CSV = f'data/{DATASET}/fx_daily.csv'
RAW_DIR = f'data/{DATASET}/raw'
PRICES_CSV = f'data/{DATASET}/prices_long.csv'
RETURNS_PIVOT_CSV = f'data/{DATASET}/returns_pivot.csv'
CHECKPOINT_DIR = f'data/{DATASET}/checkpoints'
RESULTS_DIR = f'results/{DATASET}'

DATA_START = '2000-01-01'
TEST_START = '2018-01-01'
# (last trading day kept, backtest end); update 'current' when refetching
DATA_END, TEST_END = {'paper': ('2025-05-02', '2025-05-01'),
                      'current': ('2026-10-28', '2026-10-01')}[DATASET]

# Paper Table 2: top 50 of 500 Optuna trials (2010-2017), raw weights averaged then normalized.
PRESETS = {
    'paper': {
        'clustered': dict(n_stocks=7, n_clusters=11, per_group=2, weight_method='quality',
                          w_mom=0.8178087343630099, w_vol=0.1713166412466622, w_liq=0.010874624390327865),
        'sector': dict(n_stocks=12, per_group=5, weight_method='quality',
                       w_mom=0.925217096381337, w_vol=0.062359680294678296, w_liq=0.012423223323984602),
        'top_n': dict(n_stocks=14, weight_method='quality',
                      w_mom=0.9336334828060444, w_vol=0.06018171896173537, w_liq=0.006184798232220372),
    },
}
