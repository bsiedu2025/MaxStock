# app/batch_update.py
# --------------------------------------------------------------------------------------------------
# Batch updater harga saham yang tahan banting untuk GitHub Actions:
# - Tidak import streamlit (menghindari "missing ScriptRunContext")
# - Fetch data via yfinance
# - Upsert ke Postgres/Supabase pakai psycopg2 + execute_values
# - Hindari "truth value of a Series is ambiguous" (gunakan .empty, len(), pd.notna(), dst)
# - Retry & batching, laporan progress, exit code !=0 kalau ada gagal
# --------------------------------------------------------------------------------------------------

import os
import sys
import time
import math
import argparse
import logging
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values

# ---------------------------- Logging -----------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("batch_update")

# ---------------------------- Args --------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch update harga saham ke database")
    p.add_argument("--period", type=str, default="5d",
                   help="Periode fetch yfinance (contoh: 5d, 1mo, 3mo, 1y)")
    p.add_argument("--suffix", type=str, default=".JK",
                   help="Suffix bursa (contoh: .JK). Akan ditambahkan jika belum ada")
    p.add_argument("--max-tickers", type=int, default=0,
                   help="Batas jumlah ticker; 0 berarti semua")
    p.add_argument("--tickers", type=str, default="",
                   help='Daftar ticker dipisah koma, contoh: "BBRI, BMRI, BRIS"')
    p.add_argument("--tickers-csv", type=str, default="",
                   help='Path CSV yang punya kolom "symbol"')
    p.add_argument("--batch-size", type=int, default=1000,
                   help="Ukuran batch INSERT untuk execute_values")
    p.add_argument("--dry-run", action="store_true",
                   help="Jalankan tanpa tulis DB (debug)")
    return p.parse_args()

# ---------------------------- Utils -------------------------------------------

def now_utc() -> datetime:
    return datetime.now(timezone.utc)

def normalize_symbol(sym: str, suffix: str) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return s
    if suffix and not s.endswith(suffix):
        s += suffix
    return s

def load_universe(args: argparse.Namespace) -> List[str]:
    ticks: List[str] = []
    if args.tickers.strip():
        ticks = [t.strip() for t in args.tickers.split(",") if t.strip()]
    elif args.tickers_csv:
        if not os.path.exists(args.tickers_csv):
            raise FileNotFoundError(f"tickers-csv not found: {args.tickers_csv}")
        df = pd.read_csv(args.tickers_csv)
        if "symbol" not in df.columns:
            raise ValueError('CSV harus punya kolom "symbol"')
        ticks = [str(x).strip() for x in df["symbol"].tolist() if str(x).strip()]
    else:
        raise ValueError("Harus isi --tickers atau --tickers-csv")

    ticks = [normalize_symbol(t, args.suffix) for t in ticks if t]
    # unik + jaga urutan
    seen = set()
    uniq = []
    for t in ticks:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    if args.max_tickers and args.max_tickers > 0:
        uniq = uniq[: args.max_tickers]
    return uniq

# ---------------------------- DB ----------------------------------------------

def get_conn() -> psycopg2.extensions.connection:
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("Env DATABASE_URL tidak diset")
    # Supabase PG bouncer port 6543 umumnya butuh sslmode=require
    if "sslmode" not in dsn:
        if "?" in dsn:
            dsn += "&sslmode=require"
        else:
            dsn += "?sslmode=require"
    return psycopg2.connect(dsn)

def ensure_schema(conn) -> None:
    """Bikin tabel & index kalau belum ada."""
    sql = """
    CREATE TABLE IF NOT EXISTS prices (
        symbol      text NOT NULL,
        dt          date NOT NULL,
        open        double precision,
        high        double precision,
        low         double precision,
        close       double precision,
        volume      bigint,
        updated_at  timestamptz default now(),
        PRIMARY KEY (symbol, dt)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()

def upsert_prices(conn, df: pd.DataFrame) -> int:
    """Upsert DataFrame ke tabel prices. Return jumlah baris."""
    if df is None or df.empty:
        return 0

    # Pastikan kolom ada dan tipe aman
    need_cols = ["symbol", "date", "open", "high", "low", "close", "volume"]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Kolom '{c}' tidak ada di dataframe")
    out = df.copy()

    # Normalisasi tipe
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date
    num_cols = ["open", "high", "low", "close"]
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0).astype("int64")
    out["symbol"] = out["symbol"].astype(str)

    # Drop baris tanpa tanggal (safety)
    out = out[~out["date"].isna()]
    if out.empty:
        return 0

    records = [
        (
            r["symbol"],
            r["date"],
            None if pd.isna(r.get("open")) else float(r.get("open")),
            None if pd.isna(r.get("high")) else float(r.get("high")),
            None if pd.isna(r.get("low")) else float(r.get("low")),
            None if pd.isna(r.get("close")) else float(r.get("close")),
            int(r.get("volume", 0)),
        )
        for r in out.to_dict("records")
    ]

    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO prices(symbol, dt, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, dt) DO UPDATE SET
              open   = EXCLUDED.open,
              high   = EXCLUDED.high,
              low    = EXCLUDED.low,
              close  = EXCLUDED.close,
              volume = EXCLUDED.volume,
              updated_at = now()
            """,
            records,
            page_size=10_000,
        )
    conn.commit()
    return len(records)

# ---------------------------- Fetch -------------------------------------------

def fetch_yf(symbol: str, period: str) -> pd.DataFrame:
    """Ambil harga via yfinance dan kembalikan DF [date, open, high, low, close, volume, symbol]."""
    # yfinance kadang throttle; kasih jeda configurable
    sleep_ms = int(os.environ.get("YF_SLEEP_MS", "200"))
    retries = int(os.environ.get("YF_RETRIES", "3"))

    last_exc: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period=period, auto_adjust=False)
            # Hindari truth-value ambiguous: jangan "if hist:", pakai .empty
            if hist is None or hist.empty:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])
            hist = hist.reset_index()
            # yfinance pakai 'Date' atau 'Datetime' tergantung interval
            date_col = "Date" if "Date" in hist.columns else "Datetime"
            out = pd.DataFrame({
                "date":   pd.to_datetime(hist[date_col], errors="coerce"),
                "open":   pd.to_numeric(hist.get("Open"), errors="coerce"),
                "high":   pd.to_numeric(hist.get("High"), errors="coerce"),
                "low":    pd.to_numeric(hist.get("Low"), errors="coerce"),
                "close":  pd.to_numeric(hist.get("Close"), errors="coerce"),
                "volume": pd.to_numeric(hist.get("Volume"), errors="coerce").fillna(0).astype("int64"),
            })
            out["symbol"] = symbol
            # Drop NA tanggal
            out = out[~out["date"].isna()]
            return out
        except Exception as e:
            last_exc = e
            log.warning("Fetch gagal %s (attempt %d/%d): %s", symbol, attempt, retries, e)
            time.sleep(sleep_ms / 1000.0)
    # kalau semua attempt gagal, lempar exception terakhir
    raise last_exc if last_exc else RuntimeError(f"Fetch gagal: {symbol}")

# ---------------------------- Main --------------------------------------------

def main() -> int:
    args = parse_args()
    log.info("Start batch_update period=%s suffix=%s", args.period, args.suffix)

    # Load universe
    try:
        tickers = load_universe(args)
    except Exception as e:
        log.error("Gagal load universe: %s", e)
        return 2

    total = len(tickers)
    log.info("Total tickers: %d", total)
    if total == 0:
        log.warning("Tidak ada ticker untuk diproses")
        return 0

    # DB conn
    conn = None
    if not args.dry_run:
        try:
            conn = get_conn()
            ensure_schema(conn)
        except Exception as e:
            log.error("Gagal konek/prepare DB: %s", e)
            return 3

    ok, fail = 0, 0
    t0 = time.time()

    for i, sym in enumerate(tickers, start=1):
        t1 = time.time()
        try:
            df = fetch_yf(sym, args.period)
            if df is None or df.empty:
                log.info("[%d/%d] %s kosong (no data)", i, total, sym)
                ok += 1  # consider ok walau kosong
            else:
                if not args.dry_run:
                    n = upsert_prices(conn, df)
                    log.info("[%d/%d] %s upsert %d baris (%.2fs)", i, total, sym, n, time.time() - t1)
                else:
                    log.info("[%d/%d] %s (dry-run) %d baris", i, total, sym, len(df))
                ok += 1
        except Exception as e:
            # Jangan ambiguous check di sini; cukup logging & lanjut
            log.error("[%d/%d] %s error simpan -> %s", i, total, sym, e)
            fail += 1

    dt = time.time() - t0
    log.info("Selesai: ok=%d fail=%d (%.2fs)", ok, fail, dt)

    if conn:
        try:
            conn.close()
        except Exception:
            pass

    # kalau ada gagal, kembalikan exit code 1 biar Actions merah
    return 0 if fail == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
