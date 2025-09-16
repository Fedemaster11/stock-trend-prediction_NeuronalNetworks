




import re
import pandas as pd
from pathlib import Path

SRC = "data/raw/stock_yfinance_data.csv"   # <- tu CSV combinado
DEFAULT_TICKER = "MSFT"               # <- cámbialo a "PLTR" o "RHM.DE" si el CSV es de un solo ticker

OUTDIR = Path("data/raw/prices")
OUTDIR.mkdir(parents=True, exist_ok=True)

# Lee CSV (intenta UTF-8 y UTF-8 con BOM)
try:
    df = pd.read_csv(SRC, encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv(SRC, encoding="utf-8-sig")

# Asegura columna Date
if "Date" not in df.columns:
    df.rename(columns={df.columns[0]: "Date"}, inplace=True)

# Normaliza posibles variantes de nombres de columnas
colmap = {c.lower(): c for c in df.columns}
need_names = {
    "open": "Open", "high": "High", "low": "Low",
    "close": "Close", "adj close": "Adj Close", "volume": "Volume"
}
for low, proper in need_names.items():
    if low not in colmap:
        raise SystemExit(f" Falta la columna '{proper}' en {SRC}")
    if colmap[low] != proper:
        df.rename(columns={colmap[low]: proper}, inplace=True)

# Si falta Adj Close pero hay Close, duplica
if "Adj Close" not in df.columns and "Close" in df.columns:
    df["Adj Close"] = df["Close"]

# Si no hay columna Ticker, asumimos 1 ticker
if "Ticker" not in df.columns:
    df["Ticker"] = DEFAULT_TICKER

# Guarda un CSV por cada ticker: data/raw/prices/<TICKER>_1d.csv
count = 0
for t, g in df.groupby("Ticker"):
    safe = re.sub(r"\.", "-", str(t))
    out = OUTDIR / f"{safe}_1d.csv"
    g.sort_values("Date").to_csv(out, index=False)
    count += 1
    print(f"✅ Escribí {len(g)} filas -> {out}")

print(f"✅ Listo: {count} archivo(s) creados en {OUTDIR}")
