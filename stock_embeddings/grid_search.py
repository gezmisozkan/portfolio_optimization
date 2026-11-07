"""Embedding configuration search: 12 configurations trained on the full history, scored by unsupervised metrics only.

Usage: python -m stock_embeddings.grid_search
Trains data/<DATASET>/grid/C<context>_D<dim>_P<periods>/last_epoch.pt (existing ones are skipped), then prints the models
ranked by 0.6 * silhouette + 0.4 * cohesion ratio (both min-max normalized) and writes results/<DATASET>/embedding_grid.csv.
"""
import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

from config import CHECKPOINT_DIR, RESULTS_DIR, RETURNS_PIVOT_CSV, UNIVERSE_CSV
from stock_embeddings.models.embedding_models import ClassificationEmbeddings
from stock_embeddings.train_embeddings import BATCH_SIZE, DEVICE, EPOCHS
from stock_embeddings.utils.pointwise_context import get_pointwise_tgt_context_sets
from stock_embeddings.utils.returns_data_class import ReturnsData
from stock_embeddings.utils.training_helpers import train_embeddings_from_tgt_context_sets

GRID_DIR = CHECKPOINT_DIR.replace('checkpoints', 'grid')
CONTEXT_SIZES, EMBEDDING_DIMS, PERIOD_SETS = [8, 16, 32], [50, 100], [[1], [1, 5, 10]]


def cohesion_ratio(emb, labels, rng, sample_size=10, trials=5):
    """Mean pairwise cosine similarity of names drawn from one cluster over that of names drawn at random."""
    pair_mean = lambda idx: cosine_similarity(emb[idx])[np.triu_indices(sample_size, k=1)].mean()
    big = [c for c in np.unique(labels) if (labels == c).sum() >= sample_size]
    random_sim = [pair_mean(rng.choice(len(emb), sample_size, replace=False)) for _ in range(trials)]
    cluster_sim = [pair_mean(rng.choice(np.flatnonzero(labels == rng.choice(big)), sample_size, replace=False)) for _ in range(trials)]
    return np.mean(cluster_sim) / np.mean(random_sim)


def score(emb):
    """Best KMeans silhouette over k = 2..39, its k, and the cohesion ratio of that clustering."""
    best = max((silhouette_score(emb, labels := KMeans(k, random_state=42, n_init='auto').fit_predict(emb)), k, labels)
               for k in range(2, 40))
    return best[0], best[1], cohesion_ratio(emb, best[2], np.random.default_rng(42))


def main():
    """Train the missing grid models on all returns, then rank every configuration."""
    returns = pd.read_csv(RETURNS_PIVOT_CSV, parse_dates=['date']).set_index('date')
    rows = []
    for context, dim, periods in itertools.product(CONTEXT_SIZES, EMBEDDING_DIMS, PERIOD_SETS):
        name = f"C{context}_D{dim}_P{'-'.join(map(str, periods))}"
        path = f'{GRID_DIR}/{name}/last_epoch.pt'
        if not os.path.exists(path):
            pairs = []
            for period in periods:
                data = ReturnsData(daily_returns_path=returns.copy(), extras_path=UNIVERSE_CSV)
                data.change_returns_period(period)
                data.train_test_split(1)
                pairs += get_pointwise_tgt_context_sets(data.train_returns_df, context_size=context,
                                                        iqr_noise_reduction=True, drop_nans=True)
            print(f'{name}: {len(pairs)} target-context pairs, device {DEVICE}', flush=True)
            torch.manual_seed(42)
            model = ClassificationEmbeddings(n_time_series=len(data.tickers), embedding_dim=dim)
            train_embeddings_from_tgt_context_sets(
                n_time_series=len(data.tickers), tgt_context_sets=pairs, model=model, epochs=EPOCHS,
                batch_size=BATCH_SIZE, early_stopping=False, device=DEVICE, verbose=False, save_checkpoint=False)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            model.save_model(path)
        emb = ClassificationEmbeddings.load_model(path=path, device='cpu', verbose=False).embeddings.weight.detach().cpu().numpy()
        rows.append(dict(zip(['model', 'silhouette', 'best_k', 'cohesion_ratio'], [name, *score(emb)])))

    df = pd.DataFrame(rows)
    norm = lambda s: (s - s.min()) / (s.max() - s.min())
    df['composite'] = 0.6 * norm(df.silhouette) + 0.4 * norm(df.cohesion_ratio)
    df = df.sort_values('composite', ascending=False)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(f'{RESULTS_DIR}/embedding_grid.csv', index=False)
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()
