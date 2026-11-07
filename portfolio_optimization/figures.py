"""Paper figures: Fig. 2 (PCA projection of one embedding checkpoint) and Fig. 3 (cumulative returns).

Usage: python -m portfolio_optimization.figures [preset, default paper] [checkpoint folder in data/<DATASET>/checkpoints, default the latest]
Fig. 3 reads results/<DATASET>/cumulative_<preset>.csv from portfolio_optimization.backtest.
Writes results/<DATASET>/fig2_embedding_projection.png and results/<DATASET>/fig3_cumulative_returns_<preset>.png.
"""
import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from config import CHECKPOINT_DIR, RESULTS_DIR, RETURNS_PIVOT_CSV, UNIVERSE_CSV
from stock_embeddings.models.embedding_models import ClassificationEmbeddings
from stock_embeddings.utils.returns_data_class import ReturnsData

FIG2_TICKERS = ['NASDAQ:AAPL', 'NASDAQ:MSFT', 'NYSE:JPM', 'NASDAQ:GOOGL', 'NASDAQ:AMZN', 'NASDAQ:TSLA', 'BIST:YKBNK',
                'BIST:THYAO', 'BIST:GARAN', 'COINBASE:BTCUSD', 'COINBASE:ETHUSD', 'OANDA:XAUUSD', 'OANDA:XAGUSD']
# (column in cumulative_<preset>.csv, legend name, line dash), in the paper's legend order
FIG3_SERIES = [('clustered', 'Clustered Embeddings', 'solid'), ('top_n', 'Top_N Quality', 'solid'),
               ('sector', 'Sector-Based', 'solid'), ('Equal Weighted', 'Equal-Weighted All', 'dash'),
               ('Top 10 Volume (Annual)', 'Top 10 Volume (Annual)', 'longdash'), ('Gold (XAU:USD)', 'Gold (XAU:USD)', 'solid'),
               ('BIST Index', 'BIST Index', 'longdashdot'), ('US Index', 'US Index', 'longdashdot'),
               ('Crypto Index', 'Crypto Index', 'longdashdot'), ('Hold US Stocks', 'Buy-and-Hold (US)', 'solid'),
               ('Hold BIST Stocks', 'Buy-and-Hold (BIST)', 'solid')]


def fig2(checkpoint, path):
    """PCA of all embeddings, KMeans(7) clusters as colors and sectors as symbols, showing FIG2_TICKERS only."""
    emb = ClassificationEmbeddings.load_model(path=f'{CHECKPOINT_DIR}/{checkpoint}/last_epoch.pt', device='cpu',
                                              verbose=False).embeddings.weight.detach().cpu().numpy()
    tickers = ReturnsData(RETURNS_PIVOT_CSV, UNIVERSE_CSV).tickers
    df = pd.DataFrame(PCA(n_components=2).fit_transform(emb), columns=['PC1', 'PC2'])
    df['ticker'] = tickers
    df['sector'] = df.ticker.map(pd.read_csv(UNIVERSE_CSV).set_index('ticker').sector)
    df['cluster'] = KMeans(n_clusters=7, random_state=42, n_init='auto').fit_predict(emb).astype(str)
    df = df[df.ticker.isin(FIG2_TICKERS)]
    fig = px.scatter(df, x='PC1', y='PC2', color='cluster', symbol='sector', text=df.ticker.str.removesuffix('USD'),
                     color_discrete_sequence=px.colors.qualitative.Vivid, template='plotly_white')
    fig.update_traces(marker_size=40, textposition='top center', textfont_size=25)
    fig.update_layout(showlegend=False, width=1000, height=600, margin=dict(t=30, r=30, l=30, b=30),
                      xaxis_title_font_size=25, yaxis_title_font_size=25,
                      xaxis_range=[df.PC1.min() - 0.3, df.PC1.max() + 0.3], yaxis_range=[df.PC2.min() - 0.3, df.PC2.max() + 0.3])
    fig.write_image(path, scale=3)


def fig3(cumulative_csv, path):
    """Cumulative return curves of the strategies and baselines in FIG3_SERIES."""
    cum = pd.read_csv(cumulative_csv, index_col='end_date', parse_dates=True)
    fig = go.Figure([go.Scatter(x=cum[col].dropna().index, y=cum[col].dropna(), mode='lines', name=name, line_dash=dash)
                     for col, name, dash in FIG3_SERIES])
    # plotly's default palette has 10 colors, so the 11th line gets black instead of repeating the first
    fig.update_layout(colorway=px.colors.qualitative.Plotly + ['black'], xaxis_title='Date', yaxis_title='Cumulative Return',
                      font_size=23, legend=dict(title='Strategy Groups', x=1.01, y=1), margin=dict(r=300, t=30),
                      height=750, width=1150)
    fig.write_image(path, scale=3)


def main():
    """Draw both figures into RESULTS_DIR."""
    preset = sys.argv[1] if len(sys.argv) > 1 else 'paper'
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else sorted(os.listdir(CHECKPOINT_DIR))[-1]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fig2(checkpoint, f'{RESULTS_DIR}/fig2_embedding_projection.png')
    fig3(f'{RESULTS_DIR}/cumulative_{preset}.csv', f'{RESULTS_DIR}/fig3_cumulative_returns_{preset}.png')
    print(f'{checkpoint} -> {RESULTS_DIR}/fig2_embedding_projection.png\n{preset} -> {RESULTS_DIR}/fig3_cumulative_returns_{preset}.png')


if __name__ == '__main__':
    main()
