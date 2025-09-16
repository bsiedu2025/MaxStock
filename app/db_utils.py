# D:\Docker\BrokerSummary\app\db_utils.py
import streamlit as st
import psycopg2
import pandas as pd
import os
from typing import List, Dict, Any

@st.cache_resource
def get_connection():
    """Membuat dan mengembalikan koneksi database."""
    try:
        conn = psycopg2.connect(st.secrets["connections"]["supabase"]["database_url"])
        return conn
    except Exception as e:
        st.error(f"Gagal terhubung ke database: {e}")
        return None

def fetch_data(query: str) -> List[Dict[str, Any]] | None:
    """Mengambil data dari database dan mengembalikannya sebagai list of dictionaries."""
    conn = get_connection()
    if conn is None:
        return None
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query)
            data = cur.fetchall()
        return data
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan query: {e}")
        return None
    finally:
        if conn:
            conn.close()

def execute_query(query: str) -> bool:
    """Menjalankan query DML (INSERT, UPDATE, DELETE) ke database."""
    conn = get_connection()
    if conn is None:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            conn.commit()
        return True
    except Exception as e:
        st.error(f"Terjadi kesalahan saat mengeksekusi query: {e}")
        return False
    finally:
        if conn:
            conn.close()

def get_saved_tickers_summary() -> pd.DataFrame:
    """Mengambil ringkasan ticker yang tersimpan dari database."""
    query = """
    SELECT Ticker, COUNT(*) as row_count, MIN(Tanggal) as min_date, MAX(Tanggal) as max_date
    FROM stock_prices_history
    GROUP BY Ticker
    ORDER BY Ticker ASC;
    """
    data = fetch_data(query)
    if data:
        df = pd.DataFrame(data)
        return df
    return pd.DataFrame()

def fetch_stock_prices_from_db(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Mengambil data harga saham dari database berdasarkan ticker dan rentang tanggal."""
    query = f"""
    SELECT Tanggal, Open, High, Low, Close, Volume
    FROM stock_prices_history
    WHERE Ticker = '{ticker}' AND Tanggal BETWEEN '{start_date}' AND '{end_date}'
    ORDER BY Tanggal ASC;
    """
    data = fetch_data(query)
    if data:
        df = pd.DataFrame(data)
        df['Tanggal'] = pd.to_datetime(df['Tanggal'])
        df.set_index('Tanggal', inplace=True)
        return df
    return pd.DataFrame()

def insert_stock_price_data(ticker: str, data: pd.DataFrame):
    """Menyimpan data harga saham ke database."""
    conn = get_connection()
    if conn is None:
        st.error("Tidak dapat menyimpan data: koneksi database gagal.")
        return False
    try:
        with conn.cursor() as cur:
            for index, row in data.iterrows():
                query = """
                INSERT INTO stock_prices_history (Ticker, Tanggal, Open, High, Low, Close, Volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (Ticker, Tanggal) DO UPDATE
                SET Open = EXCLUDED.Open, High = EXCLUDED.High, Low = EXCLUDED.Low, Close = EXCLUDED.Close, Volume = EXCLUDED.Volume;
                """
                cur.execute(query, (
                    ticker,
                    index.strftime('%Y-%m-%d'),
                    row['Open'],
                    row['High'],
                    row['Low'],
                    row['Close'],
                    row['Volume']
                ))
            conn.commit()
        st.success(f"Berhasil menyimpan {len(data)} baris data untuk ticker {ticker}.")
        return True
    except Exception as e:
        st.error(f"Gagal menyimpan data ke database: {e}")
        return False
    finally:
        if conn:
            conn.close()