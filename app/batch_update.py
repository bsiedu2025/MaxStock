# batch_update.py

import pandas as pd
from datetime import datetime
import time
# PERBAIKAN: Menggunakan absolute import agar dikenali saat dijalankan sebagai script
# Pastikan db_utils.py ada di dalam folder yang sama dengan batch_update.py (e.g., app/)
from db_utils import connect_db, execute_batch_query 
# Asumsi fungsi-fungsi di atas sudah tersedia di db_utils.py

# --- Fungsi Simulasi Pengambilan Data (Sama kayak sebelumnya) ---
def fetch_daily_stock_data():
    """
    Simulasi fungsi untuk mengambil data harga saham harian.
    """
    print("Mencoba mengambil data saham harian dari sumber...")
    time.sleep(2) # Simulasi delay API call
    
    # Contoh data saham (Lo sesuaikan dengan data fetch yang valid)
    today_date = datetime.now().strftime('%Y-%m-%d')
    data = [
        {'Kode': 'BBCA', 'Open': 9050, 'High': 9100, 'Low': 8980, 'Close': 9080, 'Volume': 1200000, 'Date': today_date},
        {'Kode': 'TLKM', 'Open': 3990, 'High': 4020, 'Low': 3950, 'Close': 4010, 'Volume': 5500000, 'Date': today_date},
    ]
    
    print(f"Berhasil mengambil {len(data)} data hari ini.")
    return pd.DataFrame(data)

# --- Fungsi Utama untuk Update Database ---
def batch_update_stock_prices():
    """Mengambil data saham harian dan menyimpannya ke database menggunakan db_utils."""
    print("--- Memulai Proses Batch Update Harian ---")
    
    df_data = fetch_daily_stock_data()
    if df_data.empty:
        print("Data harian kosong. Proses dihentikan.")
        return

    # MENGGUNAKAN FUNGSI KONEKSI DARI db_utils.py
    conn = connect_db() 
    if conn is None:
        print("Koneksi database gagal. Cek db_utils.py dan konfigurasi environment.")
        return

    # Query SQL untuk INSERT/UPDATE (UPSERT)
    insert_query = """
    INSERT INTO stock_prices (Kode, Date, Open, High, Low, Close, Volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        Open=VALUES(Open), 
        High=VALUES(High), 
        Low=VALUES(Low), 
        Close=VALUES(Close), 
        Volume=VALUES(Volume)
    """
    
    # Persiapan data untuk dimasukkan
    records_to_insert = []
    for index, row in df_data.iterrows():
        records_to_insert.append((
            row['Kode'], 
            row['Date'], 
            row['Open'], 
            row['High'], 
            row['Low'], 
            row['Close'], 
            row['Volume']
        ))
    
    # MENGGUNAKAN FUNGSI EKSEKUSI BATCH DARI db_utils.py
    success = execute_batch_query(conn, insert_query, records_to_insert)

    if success:
        print(f"✅ Berhasil mengupdate {len(records_to_insert)} baris data saham.")
    else:
        print("❌ Terjadi kegagalan saat eksekusi batch query.")

    conn.close()
    print("--- Proses Batch Update Selesai ---")

if __name__ == "__main__":
    batch_update_stock_prices()
