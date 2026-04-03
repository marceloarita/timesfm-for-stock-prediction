"""
Utilitários para download e preparação de dados de ações.
Usa yfinance para buscar dados históricos e salva localmente para evitar
requisições repetidas à API.
"""
import os
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def load_stock(ticker: str, start: str = "2015-01-01", end: str = "2024-12-31", force_download: bool = False) -> pd.DataFrame:
    """
    Baixa dados históricos de um ticker via yfinance e salva em CSV.
    Na segunda chamada, carrega do CSV local (mais rápido).

    Retorna um DataFrame com índice DatetimeIndex e colunas OHLCV.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    cache_path = os.path.join(DATA_DIR, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(cache_path) and not force_download:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(f"[data] Carregado do cache: {cache_path} ({len(df)} linhas)")
        return df

    print(f"[data] Baixando {ticker} de {start} até {end}...")
    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    # yfinance pode retornar MultiIndex nas colunas — achata se necessário
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw.index.name = "Date"
    raw.to_csv(cache_path)
    print(f"[data] Salvo em: {cache_path} ({len(raw)} linhas)")
    return raw


def get_close(ticker: str, **kwargs) -> pd.Series:
    """Retorna apenas a série de preço de fechamento ajustado."""
    df = load_stock(ticker, **kwargs)
    return df["Close"].rename(ticker)
