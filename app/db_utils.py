# app/db_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, tempfile, textwrap
from datetime import date
from typing import Dict, Any, Optional, Tuple

import mysql.connector
import pandas as pd
import streamlit as st
import yfinance as yf

REQUIRED_KEYS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

# -----------------------------
# Secrets & Debug
# -----------------------------
def debug_secrets() -> None:
    """Tampilkan daftar keys yang terbaca dari st.secrets (debug UI)."""
    try:
        st.caption("🔑 Keys di st.secrets:")
        st.code(", ".join(list(st.secrets.keys())) or "(kosong)", language="text")
    except Exception as e:
        st.error(f"Gagal membaca st.secrets: {e}")

def _load_cfg() -> Tuple[Dict[str, Any], Optional[str]]:
    """Baca konfigurasi DB dari st.secrets/env, dan validasi."""
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
        pass
    # env fallback
    for k in REQUIRED_KEYS + ["DB_SSL_CA", "DB_KIND"]:
        if k not in cfg or cfg[k] in ("", None):
            v = os.getenv(k)
            if v:
                cfg[k] = v.strip()
    # validasi
    missing = [k for k in REQUIRED_KEYS if not cfg.get(k)]
    if missing:
        return cfg, f"Secrets DB belum lengkap. Missing: {', '.join(missing)}"
    return cfg, None

def check_secrets(show_in_ui: bool = True) -> bool:
    _, err = _load_cfg()
    if err and show_in_ui:
        st.error(f"Gagal terhubung ke database: {err}")
    return err is None

def _prepare_ssl_ca_file(cfg: Dict[str, Any]) -> Optional[str]:
    """Tulis string CA (multiline) dari secrets menjadi file sementara .pem."""
    ca_text = cfg.get("DB_SSL_CA")
    if not ca_text:
        return None
    ca_text = textwrap.dedent(str(ca_text)).strip()
    tmp = tempfile.NamedTemporaryFile(prefix="aiven_ca_", suffix=".pem", delete=False)
    tmp.write(ca_text.encode("utf-8"))
    tmp.flush()
    tmp.close()
    return tmp.name

# -----------------------------
# Koneksi MySQL (Aiven)
# -----------------------------
def get_db_connection():
    """Kembalikan koneksi mysql.connector ke Aiven (SSL REQUIRED)."""
    cfg, err = _load_cfg()
    if err:
        raise RuntimeError(err)

    ssl_args = {}
    ca_path = _prepare_ssl_ca_file(cfg)
    if ca_path:
        ssl_args["ssl_ca"] = ca_path

    try:
        conn = mysql.connector.connect(
            host=cfg["DB_HOST"],
            port=int(cfg["DB_PORT"]),
            database=cfg["DB_NAME"],
            user=cfg["DB_USER"],
            password=cfg["DB_PASSWORD"],
            charset="utf8mb4",
            autocommit=False,
            **ssl_args,  # Aiven mewajibkan ssl_ca
        )
        return conn
    except mysql.connector.Error as e:
        st.error(f"Gagal membuka koneksi MySQL: {e}")
        raise

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

# -----------------------------
# Schema & CRUD data saham
# -----------------------------
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

# -----------------------------
# Tambahan yang diminta halaman: get_stock_info
# -----------------------------
def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Ambil info ringkas saham via yfinance.
    Menggunakan fast_info bila ada; fallback ke harga penutupan terbaru.
    """
    try:
        y = yf.Ticker(ticker)
        out: Dict[str, Any] = {}
        fi = getattr(y, "fast_info", None)
        if fi:
            # Beberapa atribut fast_info mungkin None, handle aman
            out = {
                "last_price": getattr(fi, "last_price", None),
                "previous_close": getattr(fi, "previous_close", None),
                "currency": getattr(fi, "currency", None),
                "market_cap": getattr(fi, "market_cap", None),
            }
        else:
            hist = y.history(period="5d")
            last = float(hist["Close"][-1]) if not hist.empty else None
            out = {"last_price": last}
        return out
    except Exception as e:
        return {"error": str(e)}
