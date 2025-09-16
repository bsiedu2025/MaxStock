#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bulk_importer.py
- Ambil harga saham via yfinance untuk single ticker atau CSV.
- FORMAT DF DISAMAKAN DENGAN VERSI WEB:
  * Index: DatetimeIndex (tz-naive)
  * Kolom: ['Open','High','Low','Close','Volume']  (tanpa 'Date'/'Ticker')
  * (Opsional) bisa ikutkan 'Dividends','Stock Splits' jika --keep-actions

- Insert ke DB lewat db_utils.insert_stock_price_data(df, ticker)
  (diasumsikan fungsi ini membaca tanggal dari df.index)

CLI contoh:
  python bulk_importer.py --ticker AMRT.JK --period max
  python bulk_importer.py --csv tickers.csv --period max --batch-size 8 --batch-cooldown 10
  python bulk_importer.py --csv tickers.csv --incremental

Compat: Python 3.8+
"""

import argparse
import logging
import random
import sys
import time
from datetime import timedelta
from typing import Optional, List, Tuple

import pandas as pd
import yfinance as yf

# -----------------------------------------------------------
# db_utils.py harus menyediakan:
#   insert_stock_price_data(df: pd.DataFrame, ticker: str) -> int
#   (opsional) get_last_stored_date(ticker: str) -> Optional[datetime/date]
# -----------------------------------------------------------
from db_utils import insert_stock_price_data  # type: ignore
try:
    from db_utils import get_last_stored_date  # type: ignore
except Exception:
    get_last_stored_date = None  # fallback kalau belum ada


# ---------------------- Logging ---------------------- #
def setup_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


# ---------------------- Utils ---------------------- #
def read_tickers_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    if "Ticker" not in df.columns:
        raise ValueError("File CSV harus memiliki kolom 'Ticker'.")
    tickers = (
        df["Ticker"]
        .astype(str)
        .str.strip()
        .replace("", pd.NA)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    # normalize ke UPPER
    return [t.upper() for t in tickers]


def normalize_history_df_like_web(df: pd.DataFrame, keep_actions: bool) -> pd.DataFrame:
    """
    Samakan dengan alur Web:
      - index: DatetimeIndex tz-naive
      - kolom: OHLCV (+ optional actions)
      - TANPA kolom 'Date' atau 'Ticker'; biar insert_stock_price_data yang handle.
    """
    if df.empty:
        return df

    out = df.copy()

    # Pastikan index berupa DatetimeIndex tz-naive
    if not isinstance(out.index, pd.DatetimeIndex):
        # yfinance.default: harusnya sudah DatetimeIndex; kalau belum, coba parse
        out.index = pd.to_datetime(out.index, errors="coerce")
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)

    base = ["Open", "High", "Low", "Close", "Volume"]
    cols = [c for c in base if c in out.columns]

    if keep_actions:
        for c in ["Dividends", "Stock Splits"]:
            if c not in out.columns:
                out[c] = 0.0
        cols += [c for c in ["Dividends", "Stock Splits"] if c in out.columns]

    out = out[cols].copy()

    # Cast numerik aman
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Drop baris tanpa tanggal (index NaT) jika ada
    if isinstance(out.index, pd.DatetimeIndex):
        out = out[~out.index.isna()]

    return out


def exponential_backoff_sleep(attempt: int, base: float = 1.0, cap: float = 30.0) -> None:
    """
    Sleep dengan exponential backoff + full jitter.
    attempt mulai dari 1.
    """
    delay = base * (2 ** (attempt - 1))
    if delay > cap:
        delay = cap
    delay = random.uniform(0.5 * delay, delay)
    time.sleep(delay)


def derive_date_range_from_incremental(ticker: str, default_period: str, incremental: bool):
    """
    Jika incremental=True & get_last_stored_date tersedia:
      start = last_date + 1 hari (format 'YYYY-MM-DD'), period tidak dipakai.
    else:
      gunakan period default.
    """
    if incremental and get_last_stored_date is not None:
        try:
            last_dt = get_last_stored_date(ticker)  # type: ignore
        except Exception as e:
            logging.warning("Gagal cek last stored date untuk %s: %s", ticker, e)
            last_dt = None

        if last_dt is not None:
            # dukung date atau datetime
            try:
                start = (pd.to_datetime(last_dt) + timedelta(days=1)).date().isoformat()
            except Exception:
                start = None
            if start:
                return {"start": start, "end": None, "period": None}

    return {"start": None, "end": None, "period": default_period}


def fetch_history(
    ticker: str,
    period: Optional[str],
    start: Optional[str],
    end: Optional[str],
    interval: str,
    auto_adjust: bool,
    keep_actions: bool,
) -> pd.DataFrame:
    """
    Wrapper yfinance.Ticker(...).history()
    - Default auto_adjust=True agar match Web (Adj Close diserap ke Close).
    - actions=True agar tetap bisa ikut 'Dividends'/'Stock Splits' saat diminta.
    """
    tkr = yf.Ticker(ticker)
    kwargs = {
        "interval": interval,
        "auto_adjust": auto_adjust,
        "actions": True if keep_actions else False,
    }
    if period:
        kwargs["period"] = period
    else:
        if start:
            kwargs["start"] = start
        if end:
            kwargs["end"] = end
    return tkr.history(**kwargs)


# ---------------------- Core Processing ---------------------- #
def process_single_ticker(
    ticker: str,
    default_period: str,
    interval: str,
    max_retries: int,
    auto_adjust: bool,
    incremental: bool,
    keep_actions: bool,
) -> Tuple[int, bool, Optional[str]]:
    """
    Proses satu ticker dengan retry dan backoff.
    Return: (rows_inserted_from_db_func, success, error_message)
    """
    ticker = ticker.strip().upper()
    attempt = 0
    while attempt < max_retries:
        attempt += 1
        logging.info("(%s) Unduh data percobaan %d/%d ...", ticker, attempt, max_retries)
        try:
            dr = derive_date_range_from_incremental(ticker, default_period, incremental)
            df_hist = fetch_history(
                ticker,
                period=dr["period"],
                start=dr["start"],
                end=dr["end"],
                interval=interval,
                auto_adjust=auto_adjust,
                keep_actions=keep_actions,
            )

            if df_hist.empty:
                msg = "Data kosong (ticker invalid/libur pasar/tidak ada riwayat)."
                logging.warning("(%s) %s", ticker, msg)
                exponential_backoff_sleep(attempt)
                continue

            df_norm = normalize_history_df_like_web(df_hist, keep_actions=keep_actions)
            rows_plan = len(df_norm.index)
            logging.info("(%s) Rows siap insert (like web): %d", ticker, rows_plan)

            if rows_plan == 0:
                exponential_backoff_sleep(attempt)
                continue

            # Penting: df_norm TANPA kolom Date/Ticker; biar db_utils yang handle
            inserted = insert_stock_price_data(df_norm, ticker)
            logging.info("(%s) Berhasil simpan %d baris.", ticker, inserted)
            return inserted, True, None

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logging.error("(%s) Error: %s", ticker, e)
            exponential_backoff_sleep(attempt)

    return 0, False, "Gagal setelah %d percobaan." % max_retries


def run_bulk(
    csv_path: str,
    period: str,
    interval: str,
    batch_size: int,
    batch_cooldown: float,
    max_retries: int,
    auto_adjust: bool,
    incremental: bool,
    keep_actions: bool,
) -> None:
    tickers = read_tickers_from_csv(csv_path)
    n = len(tickers)
    if n == 0:
        logging.warning("Tidak ada ticker valid di CSV.")
        return

    logging.info(
        "Mulai impor massal: %d tickers | period=%s | interval=%s | auto_adjust=%s | incremental=%s | keep_actions=%s",
        n, period, interval, auto_adjust, incremental, keep_actions
    )

    total_rows = 0
    skipped = []
    start_time = time.time()

    for i, t in enumerate(tickers, start=1):
        rows, ok, err = process_single_ticker(
            t,
            default_period=period,
            interval=interval,
            max_retries=max_retries,
            auto_adjust=auto_adjust,
            incremental=incremental,
            keep_actions=keep_actions,
        )
        total_rows += rows
        if not ok:
            skipped.append((t, err or "unknown"))

        if i % batch_size == 0 and i < n:
            logging.info("Cooldown %ss untuk menghindari rate-limit...", batch_cooldown)
            time.sleep(batch_cooldown)

    dur = time.time() - start_time
    logging.info(
        "Selesai. Rows inserted: %d | Gagal: %d/%d | Durasi: %.1fs",
        total_rows, len(skipped), n, dur
    )
    if skipped:
        logging.warning(
            "Ticker gagal: %s",
            ", ".join(["%s(%s)" % (t, reason) for t, reason in skipped])
        )


def run_single(
    ticker: str,
    period: str,
    interval: str,
    max_retries: int,
    auto_adjust: bool,
    incremental: bool,
    keep_actions: bool,
) -> None:
    rows, ok, err = process_single_ticker(
        ticker,
        default_period=period,
        interval=interval,
        max_retries=max_retries,
        auto_adjust=auto_adjust,
        incremental=incremental,
        keep_actions=keep_actions,
    )
    if ok:
        logging.info("Done. Rows inserted: %d", rows)
    else:
        logging.error("Gagal: %s", err or "unknown")


# ---------------------- CLI ---------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Bulk import harga saham ke DB (yfinance) — format DF seperti versi Web.")

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ticker", help="Kode ticker tunggal (mis. BBCA.JK)")
    mode.add_argument("--csv", help="Path CSV berkolom 'Ticker'")

    p.add_argument("--period", default="max",
                   help="yfinance period (mis. 5y, 1y, ytd, max). Abaikan jika --incremental aktif.")
    p.add_argument("--interval", default="1d",
                   help="Interval data (default 1d). Contoh: 1d,1wk,1mo.")
    p.add_argument("--max-retries", type=int, default=6, help="Jumlah maksimal percobaan per ticker.")
    p.add_argument("--no-auto-adjust", action="store_true",
                   help="Matikan auto_adjust (default ON agar match Web).")
    p.add_argument("--incremental", action="store_true",
                   help="Ambil mulai tanggal terakhir di DB (butuh get_last_stored_date di db_utils).")
    p.add_argument("--keep-actions", action="store_true",
                   help="Ikutkan kolom Dividends & Stock Splits (opsional).")

    # Hanya untuk bulk
    p.add_argument("--batch-size", type=int, default=10,
                   help="Cooldown setiap N ticker (default 10).")
    p.add_argument("--batch-cooldown", type=float, default=8.0,
                   help="Durasi cooldown batch dalam detik (default 8.0).")

    # Logging
    p.add_argument("--verbose", action="store_true", help="Mode verbose logging (DEBUG).")
    p.add_argument("--log-file", default=None, help="Tulis log ke file (opsional).")

    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(args.verbose, args.log_file)

    auto_adjust = not args.no_auto_adjust  # default True
    if args.ticker:
        run_single(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            max_retries=args.max_retries,
            auto_adjust=auto_adjust,
            incremental=args.incremental,
            keep_actions=args.keep_actions,
        )
    else:
        run_bulk(
            csv_path=args.csv,
            period=args.period,
            interval=args.interval,
            batch_size=args.batch_size,
            batch_cooldown=args.batch_cooldown,
            max_retries=args.max_retries,
            auto_adjust=auto_adjust,
            incremental=args.incremental,
            keep_actions=args.keep_actions,
        )


if __name__ == "__main__":
    main()
