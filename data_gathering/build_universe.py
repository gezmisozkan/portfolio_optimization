"""Build today's universe (S&P 500, EURO STOXX 50, CSI 300, BIST 100, top cryptos, commodities) in the universe_paper.csv format."""
import re
from io import StringIO

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tvDatafeed import Interval, TvDatafeed
from yahooquery import Screener

OUT_CSV = 'data/universe_current.csv'
UA = {'User-Agent': 'Mozilla/5.0'}
# yfinance exchange code -> TradingView exchange
TV_EXCHANGE = {'NYQ': 'NYSE', 'NMS': 'NASDAQ', 'NGM': 'NASDAQ', 'IST': 'BIST', 'SHH': 'SSE', 'SHZ': 'SZSE',
               'PAR': 'EURONEXT', 'AMS': 'EURONEXT', 'BRU': 'EURONEXT', 'GER': 'XETR', 'MIL': 'MIL',
               'MCE': 'BME', 'HEL': 'OMXHEX', 'BTS': 'BATS'}
COMMODITIES = {'GC=F': 'OANDA:XAUUSD', 'SI=F': 'OANDA:XAGUSD', 'CL=F': 'NYMEX:CL1!', 'BZ=F': 'NYMEX:BZ1!',
               'NG=F': 'NYMEX:NG1!', 'ZC=F': 'CBOT:ZC1!', 'ZW=F': 'CBOT:ZW1!', 'ZS=F': 'CBOT:ZS1!',
               'HG=F': 'COMEX:HG1!', 'PL=F': 'NYMEX:PL1!', 'PA=F': 'NYMEX:PA1!', 'CT=F': 'ICEUS:CT1!',
               'KC=F': 'ICEUS:KC1!', 'SB=F': 'ICEUS:SB1!', 'CC=F': 'ICEUS:CC1!'}


def wiki_column(url, col):
    """First column named col among the page's wikitables, footnote marks removed."""
    soup = BeautifulSoup(requests.get(url, headers=UA).text, 'html.parser')
    for tbl in soup.find_all('table', class_='wikitable'):
        df = pd.read_html(StringIO(str(tbl)))[0]
        if col in df.columns:
            return [re.sub(r'\[.*?\]', '', str(t)).strip() for t in df[col]]


def index_stocks():
    """yfinance tickers of the four indices' current constituents."""
    sp500 = [t.replace('.', '-') for t in wiki_column('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 'Symbol')]
    stoxx = wiki_column('https://en.wikipedia.org/wiki/EURO_STOXX_50', 'Ticker')
    csi = [m[2] + ('.SS' if m[1] == 'SSE' else '.SZ')
           for t in wiki_column('https://en.wikipedia.org/wiki/CSI_300_Index', 'Ticker') if (m := re.match(r'(\w+):\s*(\w+)', t))]
    soup = BeautifulSoup(requests.get('https://uzmanpara.milliyet.com.tr/canli-borsa/bist-100-hisseleri/', headers=UA).content,
                         'html.parser')
    bist = [tr['id'].removeprefix('h_tr_id_') + '.IS' for tr in soup.find_all('tr', id=re.compile(r'^h_tr_id_'))]
    return sorted(set(sp500 + stoxx + csi + bist))


def stock_row(yf_ticker):
    """Universe row from yfinance info, e.g. BRK-B on NYQ -> NYSE:BRK.B, NDA-FI.HE on HEL -> OMXHEX:NDA_FI."""
    info = yf.Ticker(yf_ticker).info
    tv_ex = TV_EXCHANGE.get(info.get('exchange'))
    symbol = yf_ticker.split('.')[0].replace('-', '.' if tv_ex == 'NYSE' else '_')
    return dict(ticker=f'{tv_ex}:{symbol}' if tv_ex else None, yfinance_ticker=yf_ticker,
                name=info.get('longName') or info.get('shortName'), exchange=info.get('exchange'), sector=info.get('sector'),
                industry=info.get('industry'), country=info.get('country'), market_cap=info.get('marketCap'))


def has_volume(tv, symbol, exchange):
    """True if TradingView has the symbol with non-zero volume over the last 10 days."""
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_daily, n_bars=10)
    return df is not None and df.volume.sum() > 0


def crypto_rows(tv):
    """Top 250 Yahoo cryptos without stable or wrapped coins, each on the first TradingView exchange with volume."""
    rows = []
    for q in Screener().get_screeners('all_cryptocurrencies_us', count=250)['all_cryptocurrencies_us']['quotes']:
        name = q.get('longName') or q.get('shortName')
        if q['symbol'].startswith('USD') or 'wrapped' in name.lower():
            continue
        base = q['symbol'].split('-')[0].rstrip('0123456789')  # HYPE32196-USD -> HYPE
        venues = ([('INDEX', 'USD')] if base in ('BTC', 'ETH') else []) + [('COINBASE', 'USD'), ('BINANCE', 'USDT')]
        ex, quote = next(((ex, quote) for ex, quote in venues if has_volume(tv, base + quote, ex)), (None, None))
        rows.append(dict(ticker=f'{ex}:{base}{quote}' if ex else None, yfinance_ticker=q['symbol'], name=name, exchange=ex,
                         sector='Crypto', industry='Crypto', country='World', market_cap=q.get('marketCap')))
    return rows


tv = TvDatafeed()
rows = [stock_row(t) for t in index_stocks()]
rows += [dict(ticker=tv_t, yfinance_ticker=yf_t, name=yf.Ticker(yf_t).info.get('shortName'), exchange='World',
              sector='Commodities', industry='Commodities', country='World') for yf_t, tv_t in COMMODITIES.items()]
rows += crypto_rows(tv)

universe = pd.DataFrame(rows)
print('dropped, no TradingView ticker:', universe[universe.ticker.isna()].yfinance_ticker.tolist())
universe = universe.dropna(subset=['ticker']).drop_duplicates('ticker')
universe.to_csv(OUT_CSV, index=False)
print(f'{OUT_CSV}: {len(universe)} tickers')
print(universe.sector.value_counts().to_string())
