"""Train one embedding model every 6 months on all returns since 2000 (the June 2025 paper version).

Usage: python -m stock_embeddings.train_embeddings [first_window_start, default 2017-07-01 = backtest windows only] [last_window_end, default TEST_END]
last_epoch.pt is written after the final epoch, so finished windows are skipped and an interrupted one retrains.
"""
import os
import sys
from datetime import datetime

import pandas as pd
import torch
from dateutil.relativedelta import relativedelta

from config import CHECKPOINT_DIR, DATA_START, RETURNS_PIVOT_CSV, TEST_END, UNIVERSE_CSV
from stock_embeddings.models.embedding_models import ClassificationEmbeddings
from stock_embeddings.utils.pointwise_context import get_pointwise_tgt_context_sets
from stock_embeddings.utils.returns_data_class import ReturnsData
from stock_embeddings.utils.training_helpers import train_embeddings_from_tgt_context_sets

REBALANCE_MONTHS = 6
CONTEXT_SIZE, EMBEDDING_DIM, PERIODS = 32, 50, [1, 5, 10]
EPOCHS, BATCH_SIZE = 100, 512
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # on Apple M1 the CPU was faster than mps


def main():
    """Train every window from the given start to the given end (default TEST_END), expanding history from DATA_START."""
    full_returns = pd.read_csv(RETURNS_PIVOT_CSV, parse_dates=['date'])
    window_start = datetime.strptime(sys.argv[1] if len(sys.argv) > 1 else '2017-07-01', '%Y-%m-%d')
    end_dt = datetime.strptime(sys.argv[2] if len(sys.argv) > 2 else TEST_END, '%Y-%m-%d')
    while window_start < end_dt:
        window_end = min(window_start + relativedelta(months=REBALANCE_MONTHS), end_dt)
        window_start = window_end
        folder = (f"{CHECKPOINT_DIR}/{DATA_START.replace('-', '')}_{window_end:%Y%m%d}"
                  f"_C{CONTEXT_SIZE}_D{EMBEDDING_DIM}_P{'-'.join(map(str, PERIODS))}_E{EPOCHS}")
        if os.path.exists(f'{folder}/last_epoch.pt'):
            continue

        train_df = full_returns[(full_returns.date >= DATA_START) & (full_returns.date < window_end)].set_index('date')
        pairs = []
        for period in PERIODS:
            data = ReturnsData(daily_returns_path=train_df.copy(), extras_path=UNIVERSE_CSV)
            data.change_returns_period(period)
            data.train_test_split(1)
            pairs += get_pointwise_tgt_context_sets(data.train_returns_df, context_size=CONTEXT_SIZE,
                                                    iqr_noise_reduction=True, drop_nans=True)
        print(f'{folder}: {len(pairs)} target-context pairs, device {DEVICE}', flush=True)
        os.makedirs(folder, exist_ok=True)

        torch.manual_seed(42)  # fixes the initial weights and the DataLoader shuffle order
        model =ClassificationEmbeddings(n_time_series=len(data.tickers), embedding_dim=EMBEDDING_DIM)
        _, losses = train_embeddings_from_tgt_context_sets(
            n_time_series=len(data.tickers), tgt_context_sets=pairs, model=model, epochs=EPOCHS,
            batch_size=BATCH_SIZE, early_stopping=False, device=DEVICE, verbose=False, save_checkpoint=False)
        model.save_model(f'{folder}/last_epoch.pt')
        print(f'{folder}: saved after {len(losses)} epochs, final loss {losses[-1]:.4f}', flush=True)


if __name__ == '__main__':
    main()
