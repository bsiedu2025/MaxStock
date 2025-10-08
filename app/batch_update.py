# batch_update.py

import pandas as pd
from datetime import datetime
import time
import mysql.connector
import yfinance as yf # WAJIB: untuk mengambil data saham
from typing import List, Tuple, Any, Dict, Optional

# MENGGUNAKAN FUNGSI KONEKSI DAN DML YANG ADA DI db_utils.py
from db_utils import get_db_connection, execute_query, insert_stock_price_data 

# Periode update yang aman untuk daily batch: ambil 1 bulan terakhir
UPDATE_PERIOD = '1mo' 

# --- Fungsi Bantuan dari 2_Update_Data_Harga_Saham.py ---
def get_distinct_tickers_from_db() -> List[str]:
    """Mengambil daftar ticker unik dari tabel stock_prices_history."""
    # Menggunakan execute_query dari db_utils.py
    query = "SELECT DISTINCT Ticker FROM stock_prices_history ORDER BY Ticker ASC;"
    # Asumsi execute_query dengan fetch_all=True mengembalikan List[Dict]
    result, error = execute_query(query, fetch_all=True) 
    
    if error or not result:
        print(f"ERROR mengambil list ticker dari DB: {error}")
        return []
        
    # Asumsi key adalah 'Ticker' atau key pertama dari dict
    if result and isinstance(result[0], dict):
        key = list(result[0].keys())[0]
        return [row[key] for row in result if row[key]]
        
    print("ERROR: Format hasil query DISTINCT Ticker tidak dikenal.")
    return []

# --- Fungsi Utama untuk Update Database ---
def batch_update_stock_prices():
    """Mengambil list saham dari DB, mengunduh data terbaru via yfinance, dan menyimpannya."""
    print("--- Memulai Proses Batch Update Harian ---")
    
    # 1. Ambil list ticker dari Database (sesuai referensi lo)
    tickers_in_db = get_distinct_tickers_from_db()
    
    if not tickers_in_db:
        print("⚠️ Database kosong. Tidak ada saham untuk diupdate.")
        return

    total_tickers = len(tickers_in_db)
    print(f"Ditemukan {total_tickers} saham di database yang siap untuk diupdate (Periode: {UPDATE_PERIOD}).")
    
    tickers_processed = 0
    total_data_inserted = 0
    
    for i, ticker in enumerate(tickers_in_db):
        print(f"[{i+1}/{total_tickers}] Mengupdate {ticker}...")
        try:
            # 2. Unduh data terbaru dari yfinance
            stock_data = yf.Ticker(ticker).history(period=UPDATE_PERIOD, auto_adjust=True)
            
            if not stock_data.empty:
                # 3. Filter kolom dan simpan
                columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume']
                df_to_save = stock_data[[col for col in columns_to_keep if col in stock_data.columns]].copy()
                
                # MENGGUNAKAN FUNGSI insert_stock_price_data DARI db_utils.py
                inserted_count = insert_stock_price_data(df_to_save, ticker)
                total_data_inserted += inserted_count
                print(f"    ✅ Berhasil menyimpan/mengupdate {inserted_count} baris data untuk {ticker}.")
            else:
                print(f"    🤷‍♂️ Tidak ada data baru ditemukan dari yfinance untuk {ticker} (Periode: {UPDATE_PERIOD}).")
                
        except Exception as e:
            print(f"    ❌ Gagal mengupdate {ticker}: {e}")
            
        tickers_processed += 1
        
    print(f"🎉 SEMUA PROSES UPDATE SELESAI! Total {tickers_processed} ticker diproses.")
    print(f"Total baris data yang disimpan/diupdate: {total_data_inserted}.")
    print("--- Proses Batch Update Selesai ---")

if __name__ == "__main__":
    # Karena ini bukan dijalankan di Streamlit, kita bisa memanggil fungsi langsung
    batch_update_stock_prices()
