# app/batch_update.py
# -*- coding: utf-8 -*-
"""
Batch updater untuk mengunduh harga saham via yfinance lalu menyimpannya ke Aiven MySQL.
Dijalankan harian via GitHub Actions (cron) atau manual (workflow_dispatch).
"""

from __future__ import annotations
import argparse
import sys
from typing import List, Optional

import pandas as pd
import yfinance as yf

# Import helper dari app/db_utils.py
# db_utils kita sudah mendukung ENV fallback untuk kredensial & CA
from db_utils import (
    get_distinct_tickers_from_price_history_with_suffix,
    insert_stock_price_data,
)

def download_prices(ticker: str, period: str = "5ds") -> pd.DataFrame:
    """Unduh harga historis via yfinance untuk periode tertentu."""
    try:
        df = yf.download(ticker, period=period, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        # Samakan kolom dengan skema DB
        df = df.rename(
            columns={
                "Open": "Open",
                "High": "High",
                "Low": "Low",
                "Close": "Close",
                "Adj Close": "Adj_Close",
                "Volume": "Volume",
            }
        )
        # Pastikan hanya kolom yang diperlukan
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        return df
    except Exception:
        return pd.DataFrame()

def run_update(period: str, suffix: Optional[str], max_tickers: Optional[int]) -> None:
    # Ambil daftar ticker dari DB
    tickers: List[str] = get_distinct_tickers_from_price_history_with_suffix(suffix)
    if not tickers:
        print("Tidak ada ticker di database. Keluar.")
        sys.exit(0)

    if max_tickers and max_tickers > 0:
        tickers = tickers[:max_tickers]

    print(f"Total ticker yang akan diupdate: {len(tickers)} (periode={period})")

    updated = 0
    for i, t in enumerate(tickers, start=1):
        df = download_prices(t, period=period)
        if df is None or df.empty:
            print(f"[{i}/{len(tickers)}] {t}: kosong/failed")
            continue
        try:
            rows = insert_stock_price_data(df, t)
            updated += rows
            print(f"[{i}/{len(tickers)}] {t}: +{rows} baris")
        except Exception as e:
            print(f"[{i}/{len(tickers)}] {t}: error simpan -> {e}")

    print(f"Selesai. Total baris tersimpan: {updated}")

def main():
    p = argparse.ArgumentParser(description="Batch update harga saham ke MySQL")
    p.add_argument("--period", default="5d", help="Periode yfinance (contoh: 5d, 1mo, 3mo)")
    p.add_argument("--suffix", default=".JK", help="Filter ticker berakhiran suffix ini (contoh .JK). Kosongkan untuk semua.")
    p.add_argument("--max-tickers", type=int, default=0, help="Batas jumlah ticker (0=semua yang cocok)")
    args = p.parse_args()

    suffix = args.suffix if args.suffix else None
    run_update(args.period, suffix, args.max_tickers)

if __name__ == "__main__":
    main()
