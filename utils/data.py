"""
Utilitários para download e preparação de dados de ações.
Usa yfinance para buscar dados históricos e salva localmente para evitar
requisições repetidas à API.
"""
import os
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def load_stock(ticker: str, start: str = "2015-01-01", end: str = "2024-12-31", force_download: bool = False) -> pd.DataFrame:
    """
    Downloads historical stock data via yfinance and caches it locally.
    On subsequent calls, loads from the local CSV cache (faster).

    Returns a DataFrame with DatetimeIndex and OHLCV columns.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    cache_path = os.path.join(RAW_DIR, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(cache_path) and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"[data] Loaded from cache: {cache_path} ({len(df)} rows)")
        return df

    print(f"[data] Downloading {ticker} from {start} to {end}...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance may return MultiIndex columns — flatten if needed
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    raw.to_csv(cache_path)
    print(f"[data] Saved to: {cache_path} ({len(raw)} rows)")
    return raw


def get_close(ticker: str, **kwargs) -> pd.Series:
    """Retorna apenas a série de preço de fechamento ajustado."""
    df = load_stock(ticker, **kwargs)
    return df["Close"].rename(ticker)
