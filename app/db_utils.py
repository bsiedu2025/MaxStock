# app/db_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, tempfile, textwrap
from datetime import date
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
    # env fallback
    for k in REQUIRED_KEYS + ["DB_SSL_CA", "DB_KIND"]:
        if k not in cfg or cfg[k] in ("", None):
            v = os.getenv(k)
            if v:
                cfg[k] = v.strip()
    # validasi wajib
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        return cfg, f"Secrets DB belum lengkap. Missing: {', '.join(missing)}"
    return cfg, None

def check_secrets(show_in_ui: bool = True) -> bool:
    _, err = _load_cfg()
    if err and show_in_ui:
        st.error(f"Gagal terhubung ke database: {err}")
    return err is None

# -----------------------------------------------------------------------------
# SSL CA helper (cache sekali)
# -----------------------------------------------------------------------------
@st.cache_resource
def _get_ssl_ca_path(ca_text: Optional[str]) -> Optional[str]:
    """Tulis isi CA (multiline) ke file sementara, dikembalikan path-nya. Dicache sekali per session."""
    if not ca_text:
        return None
    ca_text = textwrap.dedent(str(ca_text)).strip()
    tmp = tempfile.NamedTemporaryFile(prefix="aiven_ca_", suffix=".pem", delete=False)
    tmp.write(ca_text.encode("utf-8"))
    tmp.flush()
    tmp.close()
    return tmp.name

# -----------------------------------------------------------------------------
# Connection Pool (cache sekali)
# -----------------------------------------------------------------------------
def _make_pool(cfg: Dict[str, Any]) -> pooling.MySQLConnectionPool:
    ca_path = _get_ssl_ca_path(cfg.get("DB_SSL_CA"))
    kwargs = dict(
        host=cfg["DB_HOST"],
        port=int(cfg["DB_PORT"]),
        database=cfg["DB_NAME"],
        user=cfg["DB_USER"],
        password=cfg["DB_PASSWORD"],
        charset="utf8mb4",
        autocommit=False,
        ssl_ca=ca_path if ca_path else None,
    )
    # Hapus field None agar tidak dipass ke driver
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return pooling.MySQLConnectionPool(pool_name="ms_pool", pool_size=5, **kwargs)

@st.cache_resource
def _get_pool() -> pooling.MySQLConnectionPool:
    cfg, err = _load_cfg()
    if err:
        raise RuntimeError(err)
    return _make_pool(cfg)

def get_db_connection():
    """Ambil koneksi dari pool (lebih hemat latency daripada bikin koneksi baru)."""
    pool = _get_pool()
    return pool.get_connection()

# Alias kompatibilitas
get_connection = get_db_connection
get_db_conn = get_db_connection

def get_db_name() -> str:
    """Ambil nama DB aktif (berguna untuk tampilan UI)."""
    cfg, _ = _load_cfg()
    return str(cfg.get("DB_NAME", "")).strip()

def get_connection_info() -> Dict[str, Any]:
    cfg, _ = _load_cfg()
    return {
        "host": cfg.get("DB_HOST", ""),
        "port": cfg.get("DB_PORT", ""),
        "name": cfg.get("DB_NAME", ""),
        "user": cfg.get("DB_USER", ""),
        "kind": cfg.get("DB_KIND", "mysql"),
        "ssl": "yes" if cfg.get("DB_SSL_CA") else "no",
    }

# -----------------------------------------------------------------------------
# Helper eksekusi SQL
# -----------------------------------------------------------------------------
def execute_query(query: str, params=None, fetch_one=False, fetch_all=False, is_dml_ddl=False):
    """Helper eksekusi SQL dengan commit/rollback & hasil fleksibel."""
    conn = get_db_connection()
    cursor = None
    try:
        cursor = conn.cursor(dictionary=(fetch_one or fetch_all))
        cursor.execute(query, params or ())
        if is_dml_ddl:
            conn.commit()
            return (cursor.rowcount if cursor.rowcount != -1 else True, None)
        if fetch_one:
            return (cursor.fetchone(), None)
        if fetch_all:
            return (cursor.fetchall(), None)
        return (True, None)
    except mysql.connector.Error as err:
        if conn.is_connected():
            conn.rollback()
        return (None if (fetch_one or fetch_all) else False, f"Error SQL: {err}")
    finally:
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()

# -----------------------------------------------------------------------------
# Schema & CRUD data saham
# -----------------------------------------------------------------------------
def create_tables_if_not_exist():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices_history (
            Ticker  VARCHAR(20) NOT NULL,
            Tanggal DATE NOT NULL,
            Open    DECIMAL(19,4),
            High    DECIMAL(19,4),
            Low     DECIMAL(19,4),
            Close   DECIMAL(19,4),
            Volume  BIGINT,
            PRIMARY KEY (Ticker, Tanggal)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        conn.commit()
        st.toast("Schema dicek/dibuat.", icon="✅")
    finally:
        cur.close(); conn.close()

def insert_stock_price_data(df_stock: pd.DataFrame, ticker_symbol: str) -> int:
    """Insert IGNORE data historis harga saham dari DataFrame ke tabel."""
    if df_stock is None or df_stock.empty:
        st.warning("DataFrame kosong; tidak ada yang disimpan.")
        return 0
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        q = """INSERT IGNORE INTO stock_prices_history
               (Ticker, Tanggal, Open, High, Low, Close, Volume)
               VALUES (%s,%s,%s,%s,%s,%s,%s);"""
        count = 0
        for idx, row in df_stock.iterrows():
            tanggal = idx.date() if hasattr(idx, "date") else idx
            cur.execute(q, (
                ticker_symbol, tanggal,
                float(row.get("Open")) if pd.notna(row.get("Open")) else None,
                float(row.get("High")) if pd.notna(row.get("High")) else None,
                float(row.get("Low"))  if pd.notna(row.get("Low"))  else None,
                float(row.get("Close"))if pd.notna(row.get("Close"))else None,
                int(row.get("Volume")) if pd.notna(row.get("Volume")) else None,
            ))
            count += cur.rowcount
        conn.commit()
        return count
    finally:
        cur.close(); conn.close()

def fetch_stock_prices_from_db(ticker_symbol: str, start_date: Optional[date]=None, end_date: Optional[date]=None) -> pd.DataFrame:
    """Ambil data historis suatu ticker dari DB sebagai DataFrame (index=Tanggal)."""
    conn = get_db_connection()
    q = "SELECT Tanggal, Open, High, Low, Close, Volume FROM stock_prices_history WHERE Ticker=%s"
    params = [ticker_symbol]
    if start_date and end_date:
        q += " AND Tanggal BETWEEN %s AND %s"; params += [start_date, end_date]
    elif start_date:
        q += " AND Tanggal >= %s"; params += [start_date]
    elif end_date:
        q += " AND Tanggal <= %s"; params += [end_date]
    q += " ORDER BY Tanggal ASC"
    try:
        df = pd.read_sql(q, conn, params=params, index_col="Tanggal")
        if not df.empty:
            df.index = pd.to_datetime(df.index)
        return df
    finally:
        conn.close()

def get_saved_tickers_summary() -> pd.DataFrame:
    """Ringkasan data per ticker (jumlah baris, rentang tanggal, harga terakhir, hi/lo)."""
    q = """
    SELECT sph.Ticker, COUNT(*) as Jumlah_Data,
           MIN(sph.Tanggal) as Tanggal_Awal,
           MAX(sph.Tanggal) as Tanggal_Terakhir,
           (SELECT sp_last.Close FROM stock_prices_history sp_last
            WHERE sp_last.Ticker = sph.Ticker
            ORDER BY sp_last.Tanggal DESC LIMIT 1) as Harga_Penutupan_Terakhir,
           MAX(sph.High) as Harga_Tertinggi_Periode,
           MIN(sph.Low)  as Harga_Terendah_Periode
    FROM stock_prices_history sph
    GROUP BY sph.Ticker
    ORDER BY sph.Ticker ASC;
    """
    rows, err = execute_query(q, fetch_all=True)
    if err or not rows:
        return pd.DataFrame(columns=[
            "Ticker","Jumlah_Data","Tanggal_Awal","Tanggal_Terakhir",
            "Harga_Penutupan_Terakhir","Harga_Tertinggi_Periode","Harga_Terendah_Periode"
        ])
    df = pd.DataFrame(rows)
    if "Tanggal_Awal" in df: df["Tanggal_Awal"] = pd.to_datetime(df["Tanggal_Awal"], errors="coerce")
    if "Tanggal_Terakhir" in df: df["Tanggal_Terakhir"] = pd.to_datetime(df["Tanggal_Terakhir"], errors="coerce")
    return df

def get_table_list() -> List[str]:
    """Daftar tabel di database aktif."""
    rows, err = execute_query("SHOW TABLES;", fetch_all=True)
    if err or not rows:
        return []
    # dict cursor -> key pertama adalah "Tables_in_<DBNAME>"
    if isinstance(rows[0], dict):
        key = list(rows[0].keys())[0]
        return [r[key] for r in rows]
    if isinstance(rows[0], (list, tuple)):
        return [r[0] for r in rows]
    return []

def get_distinct_tickers_from_price_history_with_suffix(suffix: Optional[str] = None) -> List[str]:
    """
    Ambil daftar DISTINCT Ticker dari stock_prices_history.
    Jika suffix diberikan (mis. '.JK'), filter yang berakhiran suffix tsb.
    """
    if suffix:
        q = "SELECT DISTINCT Ticker FROM stock_prices_history WHERE Ticker LIKE %s ORDER BY Ticker ASC;"
        params = [f"%{suffix}"]
    else:
        q = "SELECT DISTINCT Ticker FROM stock_prices_history ORDER BY Ticker ASC;"
        params = []
    rows, err = execute_query(q, params=params, fetch_all=True)
    if err or not rows:
        return []
    if isinstance(rows[0], dict):
        key = list(rows[0].keys())[0]
        return [r[key] for r in rows]
    if isinstance(rows[0], (list, tuple)):
        return [r[0] for r in rows]
    return []
# ----------------------------------------------------------------------------
# Import data harian
# ----------------------------------------------------------------------------

def insert_daily_market_data(df: pd.DataFrame) -> int:
    """Insert atau Update (UPSERT) data market harian dari DataFrame (sesuai format Ringkasan Saham)."""
    if df is None or df.empty:
        # st.warning tidak bisa dipakai di sini, gunakan print/log
        print("DataFrame kosong; tidak ada yang disimpan.")
        return 0
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Query UPSERT untuk tabel daily_stock_market_data
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
        
        # Iterasi melalui DataFrame
        for idx, row in df.iterrows():
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
    DB_NAME: str = str(_cfg_for_export.get("DB_NAME", "")).strip()
except Exception:
    DB_NAME = ""
