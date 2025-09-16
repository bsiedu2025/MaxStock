import os
import streamlit as st
import psycopg2
import psycopg2.extras
import pandas as pd
from typing import List, Dict, Any
import yfinance as yf

def _dsn_from_secrets() -> str:
    host = st.secrets.get("DB_HOST")
    port = int(st.secrets.get("DB_PORT", 6543))
    db   = st.secrets.get("DB_NAME", "postgres")
    user = st.secrets.get("DB_USER", "postgres")
    pwd  = st.secrets.get("DB_PASSWORD")

    if not all([host, port, db, user, pwd]):
        raise RuntimeError("Secrets DB belum lengkap. Harus ada DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD.")

    # DSN dengan SSL wajib + keepalive biar stabil di serverless
    return (
        f"host={host} port={port} dbname={db} user={user} password={pwd} "
        "sslmode=require connect_timeout=10 keepalives=1 keepalives_idle=30 keepalives_interval=10 keepalives_count=3"
    )

@st.cache_resource(show_spinner=False)
def get_connection():
    """Satu koneksi global per proses Streamlit (jangan ditutup manual)."""
    try:
        dsn = _dsn_from_secrets()
        conn = psycopg2.connect(dsn)
        conn.autocommit = False  # kita commit manual saat DML
        return conn
    except Exception as e:
        st.error(f"Gagal terhubung ke database: {e}")
        return None

def fetch_data(query: str, params: tuple = None) -> List[Dict[str, Any]] | None:
    """SELECT → list of dicts. Koneksi disimpan (tidak di-close)."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            data = cur.fetchall()
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan query: {e}")
        return None

def execute_query(query: str, params: tuple = None) -> bool:
    """INSERT/UPDATE/DELETE."""
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

def get_saved_tickers_summary() -> pd.DataFrame:
    query = """
    SELECT Ticker,
           COUNT(*) AS "Jumlah_Data",
           MIN(Tanggal) AS "Tanggal_Awal",
           MAX(Tanggal) AS "Tanggal_Terakhir",
           (SELECT "Close" FROM stock_prices_history WHERE Ticker = T.Ticker ORDER BY "Tanggal" DESC LIMIT 1) AS "Harga_Penutupan_Terakhir",
           MAX("High") AS "Harga_Tertinggi_Periode",
           MIN("Low") AS "Harga_Terendah_Periode"
    FROM stock_prices_history T
    GROUP BY Ticker
    ORDER BY Ticker ASC;
    """
    data = fetch_data(query)
    if data:
        return pd.DataFrame(data)
    return pd.DataFrame()

def fetch_stock_prices_from_db(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    query = """
    SELECT "Tanggal", "Open", "High", "Low", "Close", "Volume"
    FROM stock_prices_history
    WHERE "Ticker" = %s AND "Tanggal" BETWEEN %s AND %s
    ORDER BY "Tanggal" ASC;
    """
    data = fetch_data(query, (ticker, start_date, end_date))
    if data:
        df = pd.DataFrame(data)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.set_index('Tanggal', inplace=True)
        return df
    return pd.DataFrame()

def insert_stock_price_data(ticker: str, data: pd.DataFrame) -> bool:
    """Simpan harga saham (UPSERT)."""
    conn = get_connection()
    if conn is None:
        st.error("Tidak dapat menyimpan data: koneksi database gagal.")
        return False
    try:
        with conn.cursor() as cur:
            for index, row in data.iterrows():
                query = """
                INSERT INTO stock_prices_history ("Ticker", "Tanggal", "Open", "High", "Low", "Close", "Volume")
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT ("Ticker", "Tanggal") DO UPDATE
                SET "Open" = EXCLUDED."Open", "High" = EXCLUDED."High", "Low" = EXCLUDED."Low",
                    "Close" = EXCLUDED."Close", "Volume" = EXCLUDED."Volume";
                """
                cur.execute(query, (
                    ticker,
                    index.strftime('%Y-%m-%d'),
                    float(row['Open']) if 'Open' in row else None,
                    float(row['High']) if 'High' in row else None,
                    float(row['Low']) if 'Low' in row else None,
                    float(row['Close']) if 'Close' in row else None,
                    float(row['Volume']) if 'Volume' in row else None,
                ))
        conn.commit()
        st.success(f"Berhasil menyimpan {len(data)} baris data untuk ticker {ticker}.")
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Gagal menyimpan data ke database: {e}")
        return False

def get_distinct_tickers() -> List[str]:
    data = fetch_data('SELECT DISTINCT "Ticker" FROM stock_prices_history ORDER BY "Ticker" ASC;')
    if data:
        return [row['Ticker'] for row in data]
    return []

# --- Tambahan: fungsi yang memang dipanggil dari halaman 1 tapi belum ada di file lama
def get_stock_info(ticker: str) -> Dict[str, Any]:
    """Ambil info singkat via yfinance agar tidak error import di pages/1_Harga_Saham."""
    try:
        y = yf.Ticker(ticker)
        info = {}
        # Beberapa field populer; aman jika tidak ada
        for k in ["trailingPE", "forwardPE", "marketCap", "shortName", "longName"]:
            info[k] = y.info.get(k)
        return info
    except Exception:
        return {}
