# app/batch_update.py
# -----------------------------------------------------------------------------------
# Batch updater harga saham untuk jalan di GitHub Actions (tanpa Streamlit)
# - Ambil list symbol dari DB (meniru halaman "Update Data Harga Saham")
# - Periode fleksibel: 5d, 1mo, 3mo, 6mo, 1y, dst (format yfinance)
# - Log "Mengupdate i/total: SYMBOL (Periode: ...)" seperti di UI
# - Upsert ke tabel prices(symbol, dt, open, high, low, close, volume)
# - Hindari "truth value of a Series is ambiguous" (pakai .empty, len(), pd.notna())
# - Exit code != 0 jika ada gagal simpan (biar Actions merah)
# -----------------------------------------------------------------------------------

import os
import sys
import time
import argparse
import logging
from typing import List, Optional
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf
import psycopg2
from psycopg2.extras import execute_values

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("batch_update")


# ------------------------- args -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch update harga saham (model UI Update Data Harga Saham)")
    p.add_argument("--period", type=str, default="1mo", help="Periode yfinance (cth: 5d, 1mo, 3mo, 6mo, 1y)")
    p.add_argument("--suffix", type=str, default=".JK", help="Suffix bursa (akan ditambahkan jika belum ada)")
    p.add_argument("--max-tickers", type=int, default=0, help="Batas jumlah ticker, 0 = semua dari DB")
    p.add_argument("--batch-size", type=int, default=1000, help="Ukuran batch execute_values ke DB")
    p.add_argument("--dry-run", action="store_true", help="Simulasi tanpa tulis DB")
    return p.parse_args()


# ------------------------- util -------------------------
def now_utc_iso():
    return datetime.now(timezone.utc).isoformat()


def normalize_symbol(sym: str, suffix: str) -> str:
    s = (sym or "").strip().upper()
    if not s:
        return s
    if suffix and not s.endswith(suffix):
        s = s + suffix
    return s


# ------------------------- DB helpers -------------------------
def require_env_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL", "")
    if not dsn:
        raise RuntimeError("Env DATABASE_URL tidak diset")
    if "sslmode" not in dsn:
        dsn = dsn + ("&" if "?" in dsn else "?") + "sslmode=require"
    return dsn


def get_conn():
    return psycopg2.connect(require_env_dsn())


def ensure_schema_prices(conn) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS prices (
      symbol     text NOT NULL,
      dt         date NOT NULL,
      open       double precision,
      high       double precision,
      low        double precision,
      close      double precision,
      volume     bigint,
      updated_at timestamptz DEFAULT now(),
      PRIMARY KEY(symbol, dt)
    );
    """
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()


def table_exists(conn, name: str) -> bool:
    sql = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema='public' AND table_name=%s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (name,))
        return cur.fetchone() is not None


def get_symbols_from_prices(conn) -> List[str]:
    sql = "SELECT DISTINCT symbol FROM prices ORDER BY symbol"
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return [r[0] for r in rows]


def detect_symbols_from_any_table(conn) -> List[str]:
    """
    Fallback kalau tabel prices masih kosong atau belum ada.
    Cari tabel publik yang punya kolom kandidat symbol/ticker/kode*,
    ambil distinct  dan pilih yang paling banyak.
    """
    candidates = ("symbol", "ticker", "tickers", "kode", "kode_saham", "code", "emiten")
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name, array_agg(column_name)
            FROM information_schema.columns
            WHERE table_schema='public'
            GROUP BY table_name
            """
        )
        all_cols = cur.fetchall()

    best_table = None
    best_col = None
    best_count = 0

    with conn.cursor() as cur:
        for tname, cols in all_cols:
            cols_lower = [c.lower() for c in (cols or [])]
            col = next((c for c in candidates if c in cols_lower), None)
            if not col:
                continue
            try:
                cur.execute(f'SELECT COUNT(*) FROM "{tname}"')
                n = cur.fetchone()[0]
                if n and n > best_count:
                    best_count = n
                    best_table = tname
                    best_col = col
            except Exception:
                continue

    if not best_table or not best_col:
        return []

    with conn.cursor() as cur:
        cur.execute(f'SELECT DISTINCT "{best_col}" FROM "{best_table}" WHERE "{best_col}" IS NOT NULL')
        out = [str(r[0]) for r in cur.fetchall() if r and r[0]]
    return out


def get_universe_from_db(conn, suffix: str, max_tickers: int) -> List[str]:
    """
    Meniru halaman UI: gunakan saham yang SUDAH ada di DB.
    1) Kalau tabel prices ada -> DISTINCT symbol from prices
    2) Kalau kosong, cari di tabel publik lain yang punya kolom symbol/ticker/* (fallback)
    """
    symbols: List[str] = []
    if table_exists(conn, "prices"):
        symbols = get_symbols_from_prices(conn)

    if not symbols:
        symbols = detect_symbols_from_any_table(conn)

    # normalisasi + tambahkan suffix bila perlu
    symbols = [normalize_symbol(s, suffix) for s in symbols if s]
    # unik + jaga urutan
    seen = set()
    uniq = []
    for s in symbols:
        if s not in seen:
            uniq.append(s)
            seen.add(s)

    if max_tickers and max_tickers > 0:
        uniq = uniq[:max_tickers]

    return uniq


# ------------------------- fetch & upsert -------------------------
def fetch_yf(symbol: str, period: str) -> pd.DataFrame:
    """
    Ambil data via yfinance; kembalikan DF kolom:
    [date, open, high, low, close, volume, symbol]
    """
    sleep_ms = int(os.environ.get("YF_SLEEP_MS", "150"))
    retries = int(os.environ.get("YF_RETRIES", "3"))
    last_exc: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(period=period, auto_adjust=False)
            if hist is None or hist.empty:
                return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "symbol"])
            hist = hist.reset_index()
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
            out = out[~out["date"].isna()]
            return out
        except Exception as e:
            last_exc = e
            log.warning("Fetch gagal %s (attempt %d/%d): %s", symbol, attempt, retries, e)
            time.sleep(sleep_ms / 1000.0)

    raise last_exc if last_exc else RuntimeError(f"Fetch gagal: {symbol}")


def upsert_prices(conn, df: pd.DataFrame, batch_size: int = 1000) -> int:
    if df is None or df.empty:
        return 0

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    num_cols = ["open", "high", "low", "close"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
    df = df[~df["date"].isna()]
    if df.empty:
        return 0

    recs = [
        (
            r["symbol"],
            r["date"],
            None if pd.isna(r.get("open")) else float(r.get("open")),
            None if pd.isna(r.get("high")) else float(r.get("high")),
            None if pd.isna(r.get("low")) else float(r.get("low")),
            None if pd.isna(r.get("close")) else float(r.get("close")),
            int(r.get("volume", 0)),
        )
        for r in df.to_dict("records")
    ]

    with conn.cursor() as cur:
        execute_values(cur, """
            INSERT INTO prices(symbol, dt, open, high, low, close, volume)
            VALUES %s
            ON CONFLICT (symbol, dt) DO UPDATE SET
              open = EXCLUDED.open,
              high = EXCLUDED.high,
              low  = EXCLUDED.low,
              close= EXCLUDED.close,
              volume=EXCLUDED.volume,
              updated_at = now()
        """, recs, page_size=max(1, batch_size))
    conn.commit()
    return len(recs)


# ------------------------- main -------------------------
def main() -> int:
    args = parse_args()
    log.info("Start: period=%s suffix=%s", args.period, args.suffix)

    try:
        dsn = require_env_dsn()
    except Exception as e:
        log.error("Gagal konek/prepare DB: %s", e)
        return 3

    ok = 0
    fail = 0

    with psycopg2.connect(dsn) as conn:
        ensure_schema_prices(conn)

        # ambil universe dari DB (meniru UI)
        symbols = get_universe_from_db(conn, args.suffix, args.max_tickers)
        total = len(symbols)
        log.info("Total saham siap diupdate: %d", total)

        if total == 0:
            log.warning("Tidak ada saham di DB yang bisa diupdate.")
            return 0

        for i, sym in enumerate(symbols, start=1):
            t0 = time.time()
            try:
                log.info("Mengupdate %d/%d: %s (Periode: %s)", i, total, sym, args.period)
                df = fetch_yf(sym, args.period)
                if df is None or df.empty:
                    log.info("  -> kosong/no data")
                    ok += 1
                    continue

                if args.dry_run:
                    log.info("  -> dry-run: %d baris", len(df))
                else:
                    n = upsert_prices(conn, df, args.batch_size)
                    log.info("  -> upsert %d baris (%.2fs)", n, time.time() - t0)
                ok += 1
            except Exception as e:
                log.error("  -> error simpan %s", e)
                fail += 1

    log.info("Selesai. OK=%d, FAIL=%d", ok, fail)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
