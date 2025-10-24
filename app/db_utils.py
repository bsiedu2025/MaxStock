# app/db_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, tempfile, textwrap
from datetime import date, timedelta # Tambahkan timedelta untuk fungsi get_ticker_list_for_update
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd
import streamlit as st
import yfinance as yf
import mysql.connector
from mysql.connector import pooling

# Kunci wajib untuk koneksi
REQUIRED_KEYS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

# -----------------------------------------------------------------------------
# Secrets & Debug
# -----------------------------------------------------------------------------
def debug_secrets() -> None:
    """Tampilkan daftar keys yang terbaca dari st.secrets (debug UI)."""
    try:
        st.caption("🔑 Keys di st.secrets:")
        st.code(", ".join(list(st.secrets.keys())) or "(kosong)", language="text")
    except Exception as e:
        st.error(f"Gagal membaca st.secrets: {e}")

def _load_cfg() -> Tuple[Dict[str, Any], Optional[str]]:
    """Baca konfigurasi DB dari st.secrets/env, dan validasi wajib."""
    cfg: Dict[str, Any] = {}
    # secrets
    try:
        for k in REQUIRED_KEYS + ["DB_SSL_CA", "DB_KIND"]:
            if k in st.secrets:
                v = st.secrets[k]
                if isinstance(v, str):
                    v = v.strip()
                cfg[k] = v
    except Exception:
        # st.secrets bisa tidak tersedia saat run lokal tanpa secrets.toml
        pass

    # env
    for k in REQUIRED_KEYS + ["DB_SSL_CA", "DB_KIND"]:
        if k in os.environ:
            v = os.environ[k]
            if isinstance(v, str):
                v = v.strip()
            cfg[k] = v

    # Validation
    missing_keys = [k for k in REQUIRED_KEYS if k not in cfg or not cfg[k]]
    if missing_keys:
        return {}, f"Kunci konfigurasi DB yang hilang/kosong: {', '.join(missing_keys)}"
    
    return cfg, None

# -----------------------------------------------------------------------------
# Koneksi DB
# -----------------------------------------------------------------------------
_DB_POOL: Optional[pooling.MySQLConnectionPool] = None

def _get_db_pool() -> Optional[pooling.MySQLConnectionPool]:
    """Inisialisasi dan kembalikan koneksi pool."""
    global _DB_POOL
    if _DB_POOL:
        return _DB_POOL

    cfg, err = _load_cfg()
    if err:
        st.error(f"Error Konfigurasi Database: {err}")
        return None
    
    try:
        pool_args = {
            "pool_name": "st_maxstock_pool",
            "pool_size": 5, # Jumlah koneksi maksimal di pool
            "host": str(cfg["DB_HOST"]),
            "port": int(cfg["DB_PORT"]),
            "database": str(cfg["DB_NAME"]),
            "user": str(cfg["DB_USER"]),
            "password": str(cfg["DB_PASSWORD"]),
            "charset": "utf8mb4",
            "autocommit": False
        }

        # Handle SSL/TLS
        if cfg.get("DB_SSL_CA"):
            pool_args["ssl_ca"] = str(cfg["DB_SSL_CA"])
            pool_args["ssl_verify_cert"] = True
            st.info("Koneksi Database menggunakan SSL/TLS.")
        
        _DB_POOL = pooling.MySQLConnectionPool(**pool_args)
        return _DB_POOL
    except Exception as e:
        st.error(f"Gagal membuat koneksi pool DB: {e}")
        return None

def get_db_connection():
    """Ambil koneksi dari pool."""
    pool = _get_db_pool()
    if pool:
        try:
            return pool.get_connection()
        except Exception as e:
            st.error(f"Gagal mengambil koneksi dari pool: {e}")
            return None
    return None

# -----------------------------------------------------------------------------
# DDL (Data Definition Language)
# -----------------------------------------------------------------------------
def create_tables_if_not_exist() -> None:
    """Buat semua tabel yang dibutuhkan jika belum ada."""
    
    # [1] stock_prices_history (Data Harga Historis yfinance)
    stock_prices_history_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS stock_prices_history (
            Ticker VARCHAR(20) NOT NULL,
            Date DATE NOT NULL,
            Open DECIMAL(10, 2),
            High DECIMAL(10, 2),
            Low DECIMAL(10, 2),
            Close DECIMAL(10, 2),
            Volume BIGINT,
            PRIMARY KEY (Ticker, Date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    # [2] ksei_month (Data KSEI/Kustodian Bulanan)
    ksei_month_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS ksei_month (
            period DATE NOT NULL,
            ticker VARCHAR(20) NOT NULL,
            custodian_category VARCHAR(50) NOT NULL,
            local_foreign CHAR(1) NOT NULL,
            shares_held BIGINT,
            value_held DECIMAL(19, 4),
            PRIMARY KEY (period, ticker, custodian_category, local_foreign)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    
    # [3] stock_list (Daftar Ticker yang sudah disimpan)
    # Ini adalah tabel yang berisi daftar ticker yang datanya sudah pernah di-fetch dan disimpan
    stock_list_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS stock_list (
            Ticker VARCHAR(20) NOT NULL PRIMARY KEY,
            Name VARCHAR(255),
            LatestUpdate DATE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)
    
    # [4] daily_stock_market_data (BARU: Data Ringkasan Harian dari File Upload)
    daily_stock_market_data_query = textwrap.dedent("""
        CREATE TABLE IF NOT EXISTS daily_stock_market_data (
            ticker VARCHAR(20) NOT NULL,
            trade_date DATE NOT NULL,
            previous_close DECIMAL(10, 2),
            open_price DECIMAL(10, 2),
            high_price DECIMAL(10, 2),
            low_price DECIMAL(10, 2),
            close_price DECIMAL(10, 2),
            price_change DECIMAL(10, 2),
            volume BIGINT,
            value_turnover BIGINT,
            frequency INT,
            individual_index DECIMAL(19, 4),
            listed_shares BIGINT,
            tradeable_shares BIGINT,
            weight_for_index DECIMAL(19, 4),
            foreign_sell_volume BIGINT,
            foreign_buy_volume BIGINT,
            PRIMARY KEY (ticker, trade_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """)

    queries = [
        stock_prices_history_query, 
        ksei_month_query, 
        stock_list_query,
        daily_stock_market_data_query # <-- Tambahkan tabel baru di sini
    ]

    conn = get_db_connection()
    if not conn: return
    
    try:
        cur = conn.cursor()
        for q in queries:
            cur.execute(q)
        conn.commit()
        st.success("Tabel database berhasil dicek/dibuat.")
    except mysql.connector.Error as err:
        st.error(f"Gagal membuat tabel: {err}")
    finally:
        conn.close()

# -----------------------------------------------------------------------------
# DML (Data Manipulation Language) & Query Fetching
# -----------------------------------------------------------------------------
def execute_query(query: str, params: Optional[Tuple[Any, ...]] = None, fetch_all: bool = False, fetch_one: bool = False) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Jalankan query umum (SELECT, UPDATE, DELETE)."""
    conn = get_db_connection()
    if not conn:
        return None, "Koneksi database gagal."
    
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute(query, params)
        
        if query.strip().upper().startswith("SELECT"):
            result = cur.fetchall() if fetch_all else cur.fetchone() if fetch_one else []
            if fetch_all:
                return result, None
            elif fetch_one:
                return [result] if result else [], None
            else:
                return [], None
        else:
            conn.commit()
            return [{"rows_affected": cur.rowcount}], None
            
    except mysql.connector.Error as err:
        return None, str(err)
    except Exception as e:
        return None, f"Error tak terduga: {e}"
    finally:
        if conn and conn.is_connected():
            cur.close()
            conn.close()

def fetch_stock_prices_from_db(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Ambil data harga historis dari DB."""
    query = textwrap.dedent("""
        SELECT Date, Open, High, Low, Close, Volume 
        FROM stock_prices_history 
        WHERE Ticker = %s AND Date BETWEEN %s AND %s 
        ORDER BY Date ASC;
    """)
    params = (ticker, start_date, end_date)
    
    result, error = execute_query(query, params, fetch_all=True)
    
    if error:
        st.error(f"Gagal mengambil data harga dari DB: {error}")
        return pd.DataFrame()
        
    if result:
        df = pd.DataFrame(result)
        df = df.set_index('Date')
        df.index = pd.to_datetime(df.index)
        return df
    return pd.DataFrame()

def insert_stock_price_data(ticker: str, df_history: pd.DataFrame) -> int:
    """Insert atau Update (UPSERT) data harga saham dari yfinance."""
    if df_history is None or df_history.empty:
        st.warning(f"DataFrame {ticker} kosong; tidak ada yang disimpan.")
        return 0
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        q = """
        INSERT INTO stock_prices_history 
               (Ticker, Date, Open, High, Low, Close, Volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            Open=VALUES(Open), High=VALUES(High), Low=VALUES(Low), Close=VALUES(Close), Volume=VALUES(Volume);
        """
        
        records = []
        for index, row in df_history.iterrows():
            records.append((
                ticker, 
                index.date(), # Date (dari index DataFrame yfinance)
                row.get('Open'), 
                row.get('High'), 
                row.get('Low'), 
                row.get('Close'), 
                row.get('Volume')
            ))
            
        cur.executemany(q, records)
        conn.commit()
        
        # Update/Insert ke stock_list
        update_stock_list_query = """
        INSERT INTO stock_list (Ticker, Name, LatestUpdate)
        VALUES (%s, %s, %s)
        ON DUPLICATE KEY UPDATE
            LatestUpdate = VALUES(LatestUpdate);
        """
        latest_date = df_history.index[-1].date()
        # Asumsi nama saham tidak diketahui di sini, jadi diisi None atau biarkan kolom Name kosong
        cur.execute(update_stock_list_query, (ticker, None, latest_date)) 
        conn.commit()
        
        return cur.rowcount # Mengembalikan jumlah baris yang terpengaruh di stock_prices_history
    except mysql.connector.Error as err:
        st.error(f"Gagal menyimpan data harga {ticker} ke DB: {err}")
        conn.rollback()
        return -1
    finally:
        cur.close(); conn.close()

# -----------------------------------------------------------------------------
# [BARU] DML untuk Tabel daily_stock_market_data
# -----------------------------------------------------------------------------
def insert_daily_market_data(df: pd.DataFrame) -> int:
    """Insert atau Update (UPSERT) data market harian dari DataFrame (sesuai format Ringkasan Saham)."""
    if df is None or df.empty:
        # Gunakan print untuk logging jika dipanggil dari background process (bukan Streamlit UI)
        print("DataFrame kosong; tidak ada yang disimpan.")
        return 0
    
    conn = get_db_connection() 
    if conn is None:
        print("Koneksi database gagal dalam insert_daily_market_data.")
        return -1
        
    cur = conn.cursor()
    try:
        # Query UPSERT untuk tabel daily_stock_market_data
        # Ganti dengan nama tabel yang sudah lo buat: daily_stock_market_data
        q = """
        INSERT INTO daily_stock_market_data
               (ticker, trade_date, previous_close, open_price, high_price, low_price, 
                close_price, price_change, volume, value_turnover, frequency, 
                individual_index, listed_shares, tradeable_shares, weight_for_index, 
                foreign_sell_volume, foreign_buy_volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            previous_close=VALUES(previous_close), open_price=VALUES(open_price), high_price=VALUES(high_price),
            low_price=VALUES(low_price), close_price=VALUES(close_price), price_change=VALUES(price_change),
            volume=VALUES(volume), value_turnover=VALUES(value_turnover), frequency=VALUES(frequency),
            individual_index=VALUES(individual_index), listed_shares=VALUES(listed_shares),
            tradeable_shares=VALUES(tradeable_shares), weight_for_index=VALUES(weight_for_index),
            foreign_sell_volume=VALUES(foreign_sell_volume), foreign_buy_volume=VALUES(foreign_buy_volume);
        """
        records = []
        
        # Iterasi melalui DataFrame (pastikan header di file CSV lo sama persis)
        for idx, row in df.iterrows():
            # Asumsi nama kolom di DataFrame sudah dibersihkan dan UPPERCASE
            records.append((
                str(row.get('KODE_SAHAM', '')).upper(),
                pd.to_datetime(row.get('TANGGAL_PERDAGANGAN_TERAKHIR')).date(), 
                float(row.get('SEBELUMNYA')) if pd.notna(row.get('SEBELUMNYA')) else None,
                float(row.get('OPEN_PRICE')) if pd.notna(row.get('OPEN_PRICE')) else None,
                float(row.get('TERTINGGI')) if pd.notna(row.get('TERTINGGI')) else None,
                float(row.get('TERENDAH')) if pd.notna(row.get('TERENDAH')) else None,
                float(row.get('PENUTUPAN')) if pd.notna(row.get('PENUTUPAN')) else None,
                float(row.get('SELISIH')) if pd.notna(row.get('SELISIH')) else None,
                int(row.get('VOLUME')) if pd.notna(row.get('VOLUME')) else None,
                int(row.get('NILAI')) if pd.notna(row.get('NILAI')) else None,
                int(row.get('FREKUENSI')) if pd.notna(row.get('FREKUENSI')) else None,
                float(row.get('INDEX_INDIVIDUAL')) if pd.notna(row.get('INDEX_INDIVIDUAL')) else None,
                int(row.get('LISTED_SHARES')) if pd.notna(row.get('LISTED_SHARES')) else None,
                int(row.get('TRADEBLE_SHARES')) if pd.notna(row.get('TRADEBLE_SHARES')) else None,
                float(row.get('WEIGHT_FOR_INDEX')) if pd.notna(row.get('WEIGHT_FOR_INDEX')) else None,
                int(row.get('FOREIGN_SELL')) if pd.notna(row.get('FOREIGN_SELL')) else None,
                int(row.get('FOREIGN_BUY')) if pd.notna(row.get('FOREIGN_BUY')) else None,
            ))
            
        cur.executemany(q, records)
        conn.commit()
        return cur.rowcount
    except mysql.connector.Error as err:
        print(f"Gagal menyimpan data market harian ke DB: {err}")
        conn.rollback()
        return -1
    finally:
        cur.close(); conn.close()
        
# -----------------------------------------------------------------------------
# Fungsi Query Pendukung (Stock Tickers & Summary)
# -----------------------------------------------------------------------------
def get_saved_tickers_summary() -> pd.DataFrame:
    """Ambil ringkasan ticker yang sudah tersimpan di stock_list."""
    query = textwrap.dedent("""
        SELECT 
            t1.Ticker,
            t1.Name,
            t1.LatestUpdate,
            t2.rows_count
        FROM stock_list t1
        LEFT JOIN (
            SELECT Ticker, COUNT(*) as rows_count
            FROM stock_prices_history
            GROUP BY Ticker
        ) t2 ON t1.Ticker = t2.Ticker
        ORDER BY t1.LatestUpdate DESC;
    """)
    result, error = execute_query(query, fetch_all=True)
    if error:
        st.error(f"Gagal mengambil ringkasan ticker: {error}")
        return pd.DataFrame()
        
    if result:
        df = pd.DataFrame(result)
        # Handle datetime/date
        if 'LatestUpdate' in df.columns:
            df['LatestUpdate'] = pd.to_datetime(df['LatestUpdate']).dt.date
        return df
    return pd.DataFrame()

def get_ticker_list_for_update(max_age_days: int = 1) -> List[str]:
    """
    Ambil daftar ticker yang perlu diupdate: 
    1. Belum ada di stock_list
    2. Sudah ada, tapi LatestUpdate lebih lama dari max_age_days (default 1 hari)
    """
    check_date = date.today() - timedelta(days=max_age_days)
    
    query = textwrap.dedent("""
        SELECT Ticker FROM stock_list
        WHERE LatestUpdate IS NULL OR LatestUpdate < %s;
    """)
    params = (check_date,)
    
    result, error = execute_query(query, params, fetch_all=True)
    if error:
        print(f"Error fetching ticker list for update: {error}")
        return []
    
    # Konversi hasil list of dicts ke list of strings
    if result:
        return [row['Ticker'] for row in result]
    return []

def fetch_all_distinct_tickers() -> List[str]:
    """Mengambil semua ticker unik yang ada di stock_prices_history."""
    query = "SELECT DISTINCT Ticker FROM stock_prices_history ORDER BY Ticker ASC;"
    result, error = execute_query(query, fetch_all=True)
    if error or not result:
        return []
    return [row['Ticker'] for row in result if row['Ticker']]

def single_column_result_to_list(rows: List[Dict[str, Any]]) -> List[Any]:
    """Utility untuk mengubah hasil fetch_all berisi satu kolom menjadi list sederhana."""
    if not rows:
        return []
    # Coba identifikasi nama kolom pertama
    if isinstance(rows[0], dict) and rows[0]:
        key = list(rows[0].keys())[0]
        return [r[key] for r in rows]
    # Fallback jika cursor bukan dictionary (walaupun sudah diset dictionary=True)
    if isinstance(rows[0], (list, tuple)):
        return [r[0] for r in rows]
    return []

# -----------------------------------------------------------------------------
# Util eksternal (dipakai halaman lain)
# -----------------------------------------------------------------------------
def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Ambil info ringkas saham via yfinance.
    Menggunakan fast_info bila ada; fallback ke harga penutupan terbaru.
    """
    try:
        y = yf.Ticker(ticker)
        fi = getattr(y, "fast_info", None)
        if fi:
            return {
                "last_price": getattr(fi, "last_price", None),
                "previous_close": getattr(fi, "previous_close", None),
                "currency": getattr(fi, "currency", None),
                "market_cap": getattr(fi, "market_cap", None),
            }
        hist = y.history(period="5d")
        last = float(hist["Close"][-1]) if not hist.empty else None
        return {"last_price": last}
    except Exception as e:
        return {"error": str(e)}

# -----------------------------------------------------------------------------
# Ekspor konstanta DB_NAME (untuk kompatibilitas import langsung di halaman)
# -----------------------------------------------------------------------------
try:
    _cfg_for_export, _err_ = _load_cfg()
    DB_NAME: str = str(_cfg_for_export.get("DB_NAME", ""))
except Exception:
    DB_NAME: str = ""
