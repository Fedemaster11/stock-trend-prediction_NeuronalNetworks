#!/usr/bin/env python3
import argparse, os
from datetime import datetime
import pandas as pd, yfinance as yf

def fetch_and_save_prices(tickers, start, end, interval, outdir):
    os.makedirs(outdir, exist_ok=True)
    for t in tickers:
        print(f"[get_prices] {t} {start}->{end} interval={interval}")
        df = yf.download(t, start=start, end=end, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            print(f"[get_prices] WARNING no data for {t}"); continue
        df=df.reset_index(); df["Ticker"]=t
        out=os.path.join(outdir, f"{t.replace('.','-')}_{interval}.csv")
        df.to_csv(out, index=False); print(f"[get_prices] saved {len(df)} -> {out}")

if __name__=="__main__":
    pa=argparse.ArgumentParser()
    pa.add_argument("--tickers", nargs="+", required=True)
    pa.add_argument("--start", default="2019-01-01")
    pa.add_argument("--end", default=None)
    pa.add_argument("--interval", default="1d")
    pa.add_argument("--outdir", default="data/raw/prices")
    a=pa.parse_args()
    end=a.end or datetime.today().strftime("%Y-%m-%d")
    fetch_and_save_prices(a.tickers, a.start, end, a.interval, a.outdir)
