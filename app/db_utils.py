# app/db_utils.py
import streamlit as st
import psycopg2
import psycopg2.extras
import pandas as pd
from typing import List, Dict, Any, Optional
import yfinance as yf

# -----------------------------
# KONEKSI
# -----------------------------
def _dsn_from_secrets() -> str:
    host = st.secrets.get("DB_HOST")
    port = int(st.secrets.get("DB_PORT", 6542))  # pooled PgBouncer
    db   = st.secrets.get("DB_NAME", "postgres")
    user = st.secrets.get("DB_USER", "postgres")
    pwd  = st.secrets.get("DB_PASSWORD")

    if not all([host, port, db, user, pwd]):
        raise RuntimeError(
            "Secrets DB belum lengkap. Harus ada DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD."
        )

    # SSL wajib untuk Supabase; keepalive agar stabil di serverless
    return (
        f"host={host} port={port} dbname={db} user={user} password={pwd} "
        "sslmode=require connect_timeout=10 keepalives=1 keepalives_idle=30 "
        "keepalives_interval=10 keepalives_count=3"
    )

@st.cache_resource(show_spinner=False)
def get_connection():
    """Satu koneksi global per proses Streamlit (jangan ditutup manual)."""
    try:
        dsn = _dsn_from_secrets()
        conn = psycopg2.connect(dsn)
        conn.autocommit = False
        return conn
    except Exception as e:
        st.error(f"Gagal terhubung ke database: {e}")
        return None

# Backward compatibility: beberapa halaman memanggil nama ini
def get_db_connection():
    return get_connection()

# -----------------------------
# HELPER QUERY
# -----------------------------
def fetch_data(query: str, params: Optional[tuple] = None) -> Optional[list]:
    conn = get_connection()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return cur.fetchall()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan query: {e}")
        return None

def execute_query(query: str, params: Optional[tuple] = None) -> bool:
    conn = get_connection()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Terjadi kesalahan saat mengeksekusi query: {e}")
        return False

# -----------------------------
# FUNGSI UTIL / KONSOL DB
# -----------------------------
def get_table_list(schema: str = "public") -> List[str]:
    """Kembalikan daftar tabel pada schema (default: public)."""
    q = """
    SELECT tablename
    FROM pg_catalog.pg_tables
    WHERE schemaname = %s
    ORDER BY tablename;
    """
    rows = fetch_data(q, (schema,))
    return [r["tablename"] for r in rows] if rows else []

# -----------------------------
# FUNGSI DOMAIN: SAHAM
# -----------------------------
def get_saved_tickers_summary() -> pd.DataFrame:
    q = """
    SELECT Ticker,
           COUNT(*) AS "Jumlah_Data",
           MIN(Tanggal) AS "Tanggal_Awal",
           MAX(Tanggal) AS "Tanggal_Terakhir",
           (SELECT "Close" FROM stock_prices_history
              WHERE Ticker = T.Ticker ORDER BY "Tanggal" DESC LIMIT 1) AS "Harga_Penutupan_Terakhir",
           MAX("High") AS "Harga_Tertinggi_Periode",
           MIN("Low")  AS "Harga_Terendah_Periode"
    FROM stock_prices_history T
    GROUP BY Ticker
    ORDER BY Ticker ASC;
    """
    rows = fetch_data(q)
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def fetch_stock_prices_from_db(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    q = """
    SELECT "Tanggal", "Open", "High", "Low", "Close", "Volume"
    FROM stock_prices_history
    WHERE "Ticker" = %s AND "Tanggal" BETWEEN %s AND %s
    ORDER BY "Tanggal" ASC;
    """
    rows = fetch_data(q, (ticker, start_date, end_date))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"])
    df.set_index("Tanggal", inplace=True)
    return df

def insert_stock_price_data(ticker: str, data: pd.DataFrame) -> bool:
    """UPSERT harga saham per tanggal."""
    conn = get_connection()
    if conn is None:
        st.error("Tidak dapat menyimpan data: koneksi database gagal.")
        return False
    try:
        with conn.cursor() as cur:
            for index, row in data.iterrows():
                q = """
                INSERT INTO stock_prices_history ("Ticker", "Tanggal", "Open", "High", "Low", "Close", "Volume")
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ("Ticker", "Tanggal") DO UPDATE
                SET "Open" = EXCLUDED."Open",
                    "High" = EXCLUDED."High",
                    "Low"  = EXCLUDED."Low",
                    "Close"= EXCLUDED."Close",
                    "Volume"=EXCLUDED."Volume";
                """
                cur.execute(q, (
                    ticker,
                    index.strftime("%Y-%m-%d"),
                    float(row.get("Open")) if row.get("Open") is not None else None,
                    float(row.get("High")) if row.get("High") is not None else None,
                    float(row.get("Low"))  if row.get("Low")  is not None else None,
                    float(row.get("Close"))if row.get("Close")is not None else None,
                    float(row.get("Volume")) if row.get("Volume") is not None else None,
                ))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Gagal menyimpan data ke database: {e}")
        return False

def get_distinct_tickers() -> List[str]:
    rows = fetch_data('SELECT DISTINCT "Ticker" FROM stock_prices_history ORDER BY "Ticker" ASC;')
    return [r["Ticker"] for r in rows] if rows else []

def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Info ringkas via yfinance agar import tidak error di halaman Harga Saham."""
    try:
        y = yf.Ticker(ticker)
        info = y.info or {}
        keep = ["trailingPE", "forwardPE", "marketCap", "shortName", "longName", "currency"]
        return {k: info.get(k) for k in keep}
    except Exception:
        return {}
