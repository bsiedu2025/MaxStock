# -*- coding: utf-8 -*-
"""
app/signals_daily.py
Generate daily FF signals and upsert into MySQL table `signals_daily`.

Rules:
- Compute ADV20 and FF_intensity = foreign_net / ADV20.
- Threshold = 95th percentile of |FF_intensity| over lookback window (default 180d).
- Signal:
  - 'FF_BUY'  if FF_intensity >= +threshold and close > MA20
  - 'FF_SELL' if FF_intensity <= -threshold and close < MA20
  - else 'NEUTRAL'
"""

import os
import sys
import argparse
import tempfile
from urllib.parse import quote_plus
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# ────────────────────────────────────────────────────────────────────────────────
def build_engine():
    host = os.getenv("DB_HOST")
    port = int(os.getenv("DB_PORT", "3306"))
    database = os.getenv("DB_NAME")
    user = os.getenv("DB_USER")
    password = os.getenv("DB_PASSWORD", "")
    ssl_ca = os.getenv("DB_SSL_CA", "")

    if not all([host, database, user]):
        raise RuntimeError("Missing DB envs: DB_HOST/DB_NAME/DB_USER")

    pwd = quote_plus(password)
    connect_args = {}
    if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(ssl_ca.encode("utf-8")); tmp.flush()
        connect_args["ssl_ca"] = tmp.name

    url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{database}"
    return create_engine(url, connect_args=connect_args, pool_recycle=300, pool_pre_ping=True)

def table_exists(engine, name: str) -> bool:
    try:
        with engine.connect() as con:
            q = text("""
                SELECT COUNT(*) FROM information_schema.tables
                WHERE table_schema = DATABASE() AND table_name = :t
            """)
            return bool(con.execute(q, {"t": name}).scalar())
    except Exception:
        return False

def ensure_table(engine):
    ddl = """
    CREATE TABLE IF NOT EXISTS signals_daily (
      base_symbol VARCHAR(32) NOT NULL,
      trade_date DATE NOT NULL,
      ff_intensity DECIMAL(18,6) NULL,
      adv20 DECIMAL(18,2) NULL,
      foreign_net BIGINT NULL,
      close DECIMAL(19,4) NULL,
      ma20 DECIMAL(19,4) NULL,
      threshold_p95 DECIMAL(18,6) NULL,
      signal VARCHAR(16) NOT NULL,
      reason VARCHAR(255) NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (base_symbol, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    with engine.begin() as con:
        con.execute(text(ddl))

# ────────────────────────────────────────────────────────────────────────────────
def fetch_symbols(engine):
    with engine.connect() as con:
        # Prefer `eod` if exists
        if table_exists(engine, "eod"):
            sql = "SELECT DISTINCT base_symbol FROM eod WHERE is_foreign_flow=0 ORDER BY base_symbol"
        else:
            sql = """
                SELECT DISTINCT Ticker AS base_symbol
                FROM eod_prices_raw
                WHERE Ticker NOT LIKE '% FF'
                ORDER BY base_symbol
            """
        sym = pd.read_sql(text(sql), con)
    return sym["base_symbol"].astype(str).tolist()

def fetch_bars(engine, symbol: str, days: int, use_eod: bool):
    if use_eod:
        sql = f"""
            SELECT p.trade_date, p.close, p.volume AS volume_price,
                   COALESCE(f.foreign_net,0) AS foreign_net
            FROM eod p
            LEFT JOIN eod f
              ON f.trade_date = p.trade_date AND f.base_symbol = p.base_symbol AND f.is_foreign_flow = 1
            WHERE p.base_symbol = :sym AND p.is_foreign_flow = 0
              AND p.trade_date >= CURDATE() - INTERVAL :n DAY
            ORDER BY p.trade_date
        """
    else:
        sql = f"""
            SELECT p.trade_date, p.close, p.volume_price,
                   COALESCE(f.foreign_net,0) AS foreign_net
            FROM
              (SELECT DATE(Tanggal) AS trade_date, `Close` AS close, Volume AS volume_price
               FROM eod_prices_raw
               WHERE Ticker = :sym AND Tanggal >= CURDATE() - INTERVAL :n DAY
              ) p
            LEFT JOIN
              (SELECT DATE(Tanggal) AS trade_date, Volume AS foreign_net
               FROM eod_prices_raw
               WHERE TRIM(REPLACE(Ticker,' FF','')) = :sym AND Tanggal >= CURDATE() - INTERVAL :n DAY
              ) f
              ON f.trade_date = p.trade_date
            ORDER BY p.trade_date
        """
    with engine.connect() as con:
        df = pd.read_sql(text(sql), con, params={"sym": symbol, "n": days})
    return df

def compute_signal(df: pd.DataFrame):
    if df.empty:
        return None
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    for c in ["close","volume_price","foreign_net"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ADV20"] = df["volume_price"].rolling(20, min_periods=5).mean()
    df["FF_intensity"] = df["foreign_net"] / df["ADV20"]
    df["MA20"] = df["close"].rolling(20, min_periods=1).mean()

    if df["FF_intensity"].notna().sum() == 0:
        return None

    thr = np.nanpercentile(np.abs(df["FF_intensity"].dropna()), 95)

    last = df.iloc[-1]
    ffi = float(last["FF_intensity"]) if pd.notna(last["FF_intensity"]) else np.nan
    close = float(last["close"]) if pd.notna(last["close"]) else np.nan
    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else np.nan

    sig = "NEUTRAL"
    reason = []
    if pd.notna(ffi) and pd.notna(thr):
        if ffi >= thr and (pd.isna(ma20) or close > ma20):
            sig = "FF_BUY"; reason.append(f"FF_intensity {ffi:.2f} ≥ p95 {thr:.2f}")
            if pd.notna(ma20): reason.append("Close>MA20")
        elif ffi <= -thr and (pd.isna(ma20) or close < ma20):
            sig = "FF_SELL"; reason.append(f"FF_intensity {ffi:.2f} ≤ -p95 {-thr:.2f}")
            if pd.notna(ma20): reason.append("Close<MA20")

    out = {
        "trade_date": last["trade_date"].date(),
        "ff_intensity": None if pd.isna(ffi) else float(ffi),
        "adv20": None if pd.isna(last["ADV20"]) else float(last["ADV20"]),
        "foreign_net": None if pd.isna(last["foreign_net"]) else int(last["foreign_net"]),
        "close": None if pd.isna(close) else float(close),
        "ma20": None if pd.isna(ma20) else float(ma20),
        "threshold_p95": None if pd.isna(thr) else float(thr),
        "signal": sig,
        "reason": "; ".join(reason)[:240] if reason else None,
    }
    return out

def upsert_signal(engine, symbol: str, rec: dict):
    sql = """
        INSERT INTO signals_daily
        (base_symbol, trade_date, ff_intensity, adv20, foreign_net, close, ma20, threshold_p95, signal, reason)
        VALUES
        (:sym, :d, :ffi, :adv, :ff, :cls, :ma, :thr, :sig, :rsn)
        ON DUPLICATE KEY UPDATE
          ff_intensity=VALUES(ff_intensity),
          adv20=VALUES(adv20),
          foreign_net=VALUES(foreign_net),
          close=VALUES(close),
          ma20=VALUES(ma20),
          threshold_p95=VALUES(threshold_p95),
          signal=VALUES(signal),
          reason=VALUES(reason)
    """
    with engine.begin() as con:
        con.execute(text(sql), {
            "sym": symbol,
            "d": rec["trade_date"],
            "ffi": rec["ff_intensity"],
            "adv": rec["adv20"],
            "ff": rec["foreign_net"],
            "cls": rec["close"],
            "ma": rec["ma20"],
            "thr": rec["threshold_p95"],
            "sig": rec["signal"],
            "rsn": rec["reason"],
        })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180, help="Lookback window (days) for percentile computation")
    ap.add_argument("--max-tickers", type=int, default=0, help="Limit number of tickers to process (0 = all)")
    args = ap.parse_args()

    eng = build_engine()
    ensure_table(eng)

    use_eod = table_exists(eng, "eod")
    symbols = fetch_symbols(eng)
    if args.max_tickers > 0:
        symbols = symbols[: args.max_tickers]

    ok, fail = 0, 0
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_bars(eng, sym, args.days, use_eod)
            rec = compute_signal(df)
            if rec is not None:
                upsert_signal(eng, sym, rec)
                ok += 1
            else:
                # still ensure neutrality row? skip to avoid clutter
                pass
            if i % 50 == 0:
                print(f"[{i}/{len(symbols)}] processed: {sym}")
        except Exception as e:
            fail += 1
            print(f"[ERROR] {sym}: {e}", file=sys.stderr)

    print(f"Done. Success: {ok}, Failed: {fail}, Symbols total: {len(symbols)}")

if __name__ == "__main__":
    main()