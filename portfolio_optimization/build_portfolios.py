"""Build monthly portfolios for the clustered, sector and Top-N strategies.

Usage: python -m portfolio_optimization.build_portfolios [preset] [strategy ...]
       e.g. python -m portfolio_optimization.build_portfolios paper top_n sector
Writes results/<DATASET>/portfolios/<preset>/<strategy>.csv in a 3-row format (Date, Type, List).
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.cluster import KMeans

from config import CHECKPOINT_DIR, PRESETS, PRICES_CSV, RESULTS_DIR, RETURNS_PIVOT_CSV, TEST_END, TEST_START, UNIVERSE_CSV
from stock_embeddings.models.embedding_models import ClassificationEmbeddings
from stock_embeddings.utils.returns_data_class import ReturnsData


def load_prices():
    """Long price table merged with sectors, with volume scaled for commodities (x100) and crypto (x20)."""
    df = pd.read_csv(PRICES_CSV, parse_dates=['date']).rename(columns={'volume': 'volume_org'})
    df = df.merge(pd.read_csv(UNIVERSE_CSV)[['ticker', 'sector']], on='ticker', how='left')
    df['adj_volume'] = df.volume_org * df.sector.map({'Commodities': 100, 'Crypto': 20}).fillna(1)
    df['adj_volume'] = df.adj_volume.round(0)
    return df.sort_values(['ticker', 'date']).reset_index(drop=True)


def build_factor_table(df):
    """MOM (12-1 month), VOL (3-month annualized, negated) and LIQ (3-month log dollar volume), winsorized and z-scored per month."""
    grp = df.groupby('ticker')
    df['LogClose'] = np.log(df.close_usd)
    df['Return'] = grp.LogClose.diff()
    df['MOM'] = grp.close_usd.transform(lambda s: s.pct_change(252 - 21).shift(21))
    df['VOL'] = grp.Return.transform(lambda s: s.rolling(63).std(ddof=0)) * np.sqrt(252)
    df['DollarVol'] = np.log1p(df.close_usd * df.adj_volume)
    df['LIQ'] = df.groupby('ticker').DollarVol.transform(lambda s: s.rolling(63).mean())
    df['TRAIL_VOL'] = grp.Return.transform(lambda s: s.rolling(63).std(ddof=0))  # risk_parity weights
    df['USD_VOL_21D'] = (df.close_usd * df.adj_volume).groupby(df.ticker).transform(lambda s: s.rolling(21, min_periods=1).sum())  # dynamic picks
    monthly = df.date.dt.to_period('M')
    for f in ['MOM', 'VOL', 'LIQ']:
        # clip at the 1%/99% quantiles, then center with the unclipped mean/std, as used for the paper's results
        df[f] = df.groupby(monthly)[f].transform(lambda x: (x.clip(x.quantile(.01), x.quantile(.99)) - x.mean()) / x.std())
    df['VOL'] = -df.VOL
    return df


def quality_on(fac, dt, p):
    """Last row of the rebalance month per ticker (NaN kept): weighted MOM/VOL/LIQ quality score, TRAIL_VOL and USD_VOL_21D."""
    month = fac[(fac.date >= dt.replace(day=1)) & (fac.date <= dt)].groupby('ticker').tail(1).set_index('ticker')
    month['quality_score'] = p['w_mom'] * month.MOM + p['w_vol'] * month.VOL + p['w_liq'] * month.LIQ
    return month[['quality_score', 'TRAIL_VOL', 'USD_VOL_21D']]


def load_clusters(dt, n_clusters, tickers, first_dates):
    """KMeans labels of the latest checkpoint ending on/before dt; tickers listed after (end - 2 months) get -1."""
    ckpts = [(datetime.strptime(d.split('_')[1], '%Y%m%d'), d) for d in os.listdir(CHECKPOINT_DIR)]
    end_dt, best = max(c for c in ckpts if pd.Timestamp(c[0]) <= dt)
    model = ClassificationEmbeddings.load_model(path=f'{CHECKPOINT_DIR}/{best}/last_epoch.pt', device='cpu', verbose=False)
    emb = model.embeddings.weight.detach().cpu().numpy()
    eligible = np.array([first_dates.get(t, pd.NaT) <= end_dt - relativedelta(months=2) for t in tickers])
    labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(emb[eligible])
    clusters = dict.fromkeys(tickers, -1)
    clusters.update(zip(np.array(tickers)[eligible], labels))
    return clusters


def select(q, p, groups=None):
    """Pick the portfolio: top n_stocks, or per group from a quality-ranked pool.

    pick_method 'fixed' (default): per_group names per group from the top per_group*n_groups.
    pick_method 'dynamic': n_stocks split across groups by 21-day dollar volume, from the top 2*n_stocks.
    weight_method 'quality', 'risk_parity' (1 / 63-day return std) or anything else for equal weights.
    """
    month_q = q.rename_axis('Ticker').reset_index()
    dynamic = p.get('pick_method') == 'dynamic'
    if groups is None:
        top = month_q.sort_values('quality_score', ascending=False).head(p['n_stocks'])
    else:
        n_groups = month_q.Ticker.map(groups).replace(-1, np.nan).dropna().nunique()
        pool = month_q.sort_values('quality_score', ascending=False).head(2 * p['n_stocks'] if dynamic else p['per_group'] * n_groups)
        pool['cluster'] = pool.Ticker.map(groups)
        pool = pool[pool.cluster.notna() & (pool.cluster != -1)]
        picks = pd.Series(p.get('per_group', 0), index=pool.cluster.unique())
        if dynamic and len(pool):
            vol = pool.groupby('cluster').USD_VOL_21D.sum()
            picks = (vol / vol.sum() * p['n_stocks']).round().astype(int).clip(lower=1)
            while picks.sum() > p['n_stocks']:
                picks[picks.idxmax()] -= 1
            while picks.sum() < p['n_stocks']:
                picks[picks.idxmin()] += 1
        picked = pd.concat([pool[pool.cluster == g].head(n) for g, n in picks.items()]) if len(pool) else pool
        if len(picked) < p['n_stocks']:
            picked = pd.concat([picked, pool[~pool.Ticker.isin(picked.Ticker)].head(p['n_stocks'] - len(picked))])
        top = picked.sort_values('quality_score', ascending=False).head(p['n_stocks'])
    top = top[top.quality_score >= p.get('min_quality_score', 0.0)]
    w = {'quality': top.quality_score, 'risk_parity': 1 / top.TRAIL_VOL}.get(p['weight_method'], pd.Series(1.0, index=top.index))
    top = top[(w > 0) & w.notna()]
    top['weight'] = w[top.index] / w[top.index].sum()
    return top


def save_3row(details, path):
    """Write {date: DataFrame} in a 3-row format: one Ticker, Weight and Quality row per date."""
    rows = []
    for dt, df in details.items():
        rows += [{'Date': dt, 'Type': 'Ticker', 'List': str(df.Ticker.tolist())},
                 {'Date': dt, 'Type': 'Weight', 'List': str(df.weight.tolist())},
                 {'Date': dt, 'Type': 'Quality', 'List': str(df.quality_score.tolist())}]
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    """Build and save the requested strategies for the test period."""
    preset = sys.argv[1] if len(sys.argv) > 1 else 'paper'
    strategies = sys.argv[2:] or ['clustered', 'sector', 'top_n']
    params = PRESETS[preset]

    prices = load_prices()
    start, end = pd.Timestamp(TEST_START), pd.Timestamp(TEST_END)
    fac = build_factor_table(prices[(prices.date >= start - pd.DateOffset(days=400)) & (prices.date <= end)].copy())
    fac = fac[fac.date >= start]
    first_dates = prices.groupby('ticker').date.min()
    sectors = prices.groupby('ticker').sector.last().to_dict()
    tickers = ReturnsData(RETURNS_PIVOT_CSV, UNIVERSE_CSV).tickers
    rebalance_dates = fac.date.dt.to_period('M').drop_duplicates().dt.to_timestamp(how='end')

    out_dir = f'{RESULTS_DIR}/portfolios/{preset}'
    os.makedirs(out_dir, exist_ok=True)
    for name in strategies:
        p, details = params[name], {}
        for dt in rebalance_dates:
            q = quality_on(fac, dt, p)
            q = q[q.index.isin(first_dates[first_dates <= dt].index)]  # tickers with a price on/before dt
            if q.empty:  # a month without scores gets no portfolio (May 2025)
                continue
            if name == 'clustered':
                groups = load_clusters(dt, p['n_clusters'], tickers, first_dates)
            else:
                groups = sectors if name == 'sector' else None
            top = select(q, p, groups)
            if not top.empty:
                details[dt] = top
        save_3row(details, f'{out_dir}/{name}.csv')
        last = details[rebalance_dates.iloc[-3]]
        print(f'{name}: {len(details)} rebalances -> {out_dir}/{name}.csv | {rebalance_dates.iloc[-3].date()}:')
        print(last[['Ticker', 'weight', 'quality_score']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
