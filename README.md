# Portfolio Optimization via Learned Stock Embeddings

Code for **"Learning Stock Embeddings for Multi-Asset Portfolio Optimization"** (IDEAL 2026).

Instead of forecasting individual asset prices, the pipeline learns embeddings from return similarity, clusters assets into latent sectors, ranks them within each cluster by momentum/volatility/liquidity, and tunes construction parameters with Optuna. In a sequential backtest from January 2018 to May 2025 (89 monthly rebalances) over a USD-normalized dataset of 1,093 instruments (U.S., European, Turkish, and Chinese equities, commodities, and cryptocurrencies), the clustered-embedding strategy reached **48.78x** final return, more than double the sector-based (20.66x) and Top-N quality (19.47x) strategies.

## Pipeline

<img src="assets/pipeline_diagram.png" alt="Pipeline architecture" width="520">

Raw OHLCV data are normalized, turned into daily return contexts, embedded, clustered into latent sectors, scored with financial factors, and converted into monthly portfolios.

## Learned Embeddings vs. Static Sectors

![2D PCA projection of stock embeddings](assets/embedding_plot_2d.png)

A 2D PCA projection of one embedding checkpoint. The Turkish banks BIST:YKBNK and BIST:GARAN sit close together and far from NYSE:JPM, even though all three carry the same financial-sector label — the embeddings capture relationships static sector taxonomies miss.

## Backtest Results

![Cumulative returns of all strategies](assets/cum_returns_all.png)

| Strategy | Return | Vol. | Sharpe | MDD |
|---|---|---|---|---|
| **Clustered embeddings** | **48.78x** | 0.134 | 1.39 | -41.70% |
| Sector-based | 20.66x | 0.157 | 1.03 | -55.28% |
| Top-N quality | 19.47x | 0.177 | 0.96 | -70.90% |
| Buy-and-hold U.S. | 4.32x | 0.073 | 0.93 | -35.81% |
| Volume, annual | 3.55x | 0.309 | 0.78 | -24.52% |
| Equal-weighted | 2.16x | 0.054 | 0.66 | -22.09% |

Full results, including regional index baselines and the perfect-foresight upper bound, are in the paper.

## Data

1,093 instruments: every S&P 500 constituent, 49 EURO STOXX 50 stocks, the full BIST 100, the CSI 300, 15 commodities (spot gold and silver plus 13 front-month futures), and 129 liquid cryptocurrencies. Daily OHLCV from January 2000 onward, sourced from TradingView (prices) and Yahoo Finance (sector metadata), USD-normalized and calendar-harmonized. Commodity volume is scaled by 100 and cryptocurrency volume by 20 so the liquidity factor stays comparable across asset classes. See the paper for full preprocessing details.

Price data are not redistributed; the scripts below download them. The repository ships only the paper's ticker list, `data/universe_paper.csv` (1,093 instruments with Yahoo Finance sectors), since the index memberships it was built from can no longer be scraped. Step 0 builds a current universe from today's S&P 500, EURO STOXX 50, BIST 100 and CSI 300 constituents, the same 15 commodities, and the top 250 Yahoo Finance cryptocurrencies (stable and wrapped coins removed).

TradingView data change over time. Re-fetching the paper universe today returns about 1,040 of the 1,093 instruments: delisted or renamed stocks and several Binance USD pairs are no longer available, and some tickers now point to a different instrument. Results will therefore differ from the paper. The current universe is complete, but backtesting past years on today's index members carries survivorship bias.

## Setup

Python 3.12 with a CUDA GPU is recommended for embedding training.

```bash
pip install -r requirements.txt
```

## Datasets

The `DATASET` environment variable picks the universe; every step reads and writes only that dataset's folders. Paths, dates and the paper's strategy parameters are in `config.py`.

| `DATASET` | Universe | Data | Backtest |
| --- | --- | --- | --- |
| unset (`paper`) | `data/universe_paper.csv` | to 2025-05-02 | 2018-01-01 to 2025-05-01 |
| `current` | `data/universe_current.csv`, built by step 0 | to 2026-10-28 | 2018-01-01 to 2026-10-01 |

To refresh `current`, rerun step 0 and update its dates in `config.py`.

## Run order

Run every step from the repository root. For the current universe, prefix each command with `DATASET=current`.

| Step | Command | Writes |
| --- | --- | --- |
| 0. Universe (`current` only) | `python -m data_gathering.build_universe` | `data/universe_current.csv` |
| 1. FX rates | `python -m data_gathering.fetch_fx` | `data/<DATASET>/fx_daily.csv` |
| 2. Prices | `python -m data_gathering.fetch_prices` | `data/<DATASET>/raw/*.csv`, failures in `raw/failed.txt` (rerun to retry them) |
| 3. Dataset | `python -m data_gathering.build_dataset` | `data/<DATASET>/prices_long.csv`, `data/<DATASET>/returns_pivot.csv` |
| 4. Embeddings | `python -m stock_embeddings.train_embeddings [first_window_start] [last_window_end]` | `data/<DATASET>/checkpoints/*/last_epoch.pt` |
| 5. Tuning (optional) | `python -m portfolio_optimization.tune <clustered\|sector\|top_n> [n_trials]` | `results/<DATASET>/optuna_<strategy>.csv` |
| 6. Portfolios | `python -m portfolio_optimization.build_portfolios [preset] [strategy ...]` | `results/<DATASET>/portfolios/<preset>/<strategy>.csv` |
| 7. Backtest | `python -m portfolio_optimization.backtest [preset]` | `results/<DATASET>/backtest_<preset>.csv`, `results/<DATASET>/cumulative_<preset>.csv` |

- Step 4 by default trains the six-monthly windows the backtest uses (ending 2018-01-01 up to the backtest end). Tuning on 2010-2017 also needs the earlier windows: `python -m stock_embeddings.train_embeddings 2005-01-01 2018-01-01`. Windows can be split across GPUs with `CUDA_VISIBLE_DEVICES`.
- Step 5 runs Optuna (500 trials by default) on the January 2010 to December 2017 rebalances only. It searches the factor weights (0-1 each, normalized), portfolio size (5-50), weighting scheme (equal, quality-proportional, or inverse 63-day volatility), and for the grouped strategies the pick method (a fixed 1-5 names per group, or `n_stocks` split across groups by 21-day dollar volume) and the cluster count (5-30). The minimum quality score is fixed at 0. It prints the top-50 average (numbers averaged, choices by majority), to be added to `PRESETS` in `config.py`.
- The `paper` preset holds the paper's Table 2 parameters. Averaged over their top 50 trials, all three strategies chose quality-proportional weighting with momentum weights above 0.8; the grouped strategies use fixed picks (clustered: 2 names from each of 11 clusters, 7 in total; sector-based: up to 5 per sector, 12 in total; Top-N: 14 names).
- Embedding training is seeded, but retrained embeddings differ from the paper's checkpoints, so the clustered strategy will not reproduce the paper's figures exactly.

## License

MIT, see `LICENSE`.
