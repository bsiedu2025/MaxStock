# batch_update.py

import pandas as pd
from datetime import datetime
import time
import mysql.connector # Perlu di-import untuk handle error specific
from typing import List, Tuple, Any

# MENGGUNAKAN FUNGSI KONEKSI YANG ADA DI db_utils.py
# Fungsi yang tersedia: get_db_connection, get_connection
from db_utils import get_db_connection 


# --- Fungsi Simulasi Pengambilan Data (Diperbarui agar output sesuai DB) ---
# Asumsi: lo akan mengambil list of tuples atau list of dicts yang siap di-insert
def fetch_daily_stock_data_for_upsert() -> List[Tuple[Any, ...]]:
    """
    Mengambil data saham harian dan mengembalikan data dalam format siap UPSERT.
    Format data: (Kode, Tanggal, Open, High, Low, Close, Volume)
    """
    print("Mencoba mengambil data saham harian dari sumber...")
    time.sleep(2) # Simulasi delay API call
    
    today_date_str = datetime.now().strftime('%Y-%m-%d')
    today_date_obj = datetime.now().date() # Menggunakan object date untuk database
    
    # Data dalam format DataFrame
    data_df = pd.DataFrame([
        {'Kode': 'BBCA', 'Open': 9050, 'High': 9100, 'Low': 8980, 'Close': 9080, 'Volume': 1200000, 'Date': today_date_obj},
        {'Kode': 'TLKM', 'Open': 3990, 'High': 4020, 'Low': 3950, 'Close': 4010, 'Volume': 5500000, 'Date': today_date_obj},
    ])
    
    # Konversi DataFrame ke List of Tuples yang sesuai dengan SQL query
    records_to_insert = []
    for index, row in data_df.iterrows():
        records_to_insert.append((
            row['Kode'], 
            row['Date'], 
            float(row['Open']), 
            float(row['High']), 
            float(row['Low']), 
            float(row['Close']), 
            int(row['Volume'])
        ))
    
    print(f"Berhasil mengambil {len(records_to_insert)} data hari ini.")
    return records_to_insert

# --- Fungsi Eksekusi Batch (Manual di sini, karena db_utils tidak punya execute_batch_query) ---
def execute_stock_upsert(conn, query, records: List[Tuple[Any, ...]]) -> int:
    """
    Menjalankan UPSERT batch ke database.
    Harus dilakukan di sini karena db_utils lo tidak menyediakan helper batch.
    """
    if not records:
        print("Tidak ada data untuk di-upsert.")
        return 0
        
    cur = conn.cursor()
    total_rows_affected = 0
    try:
        # Menggunakan executemany untuk efisiensi
        cur.executemany(query, records)
        conn.commit()
        # NOTE: Pada UPSERT, rowcount bisa lebih dari 1 per baris (1 insert, 1-2 update)
        total_rows_affected = cur.rowcount 
        return total_rows_affected
    except mysql.connector.Error as err:
        print(f"❌ ERROR saat menjalankan UPSERT batch: {err}")
        conn.rollback() 
        return -1
    finally:
        cur.close()


# --- Fungsi Utama untuk Update Database ---
def batch_update_stock_prices():
    """Mengambil data saham harian dan menyimpannya ke database."""
    print("--- Memulai Proses Batch Update Harian ---")
    
    records_to_insert = fetch_daily_stock_data_for_upsert()
    
    if not records_to_insert:
        print("Data harian kosong. Proses dihentikan.")
        return

    # Panggil fungsi koneksi dari db_utils.py
    # GUE PAKE get_db_connection() SESUAI DENGAN ISI FILE LO
    conn = get_db_connection() 
    
    # Asumsi: Lo pakai tabel stock_prices_history (sesuai di db_utils.py lo)
    # Primary Key di db_utils.py adalah (Ticker, Tanggal)
    insert_query = """
    INSERT INTO stock_prices_history (Ticker, Tanggal, Open, High, Low, Close, Volume)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE 
        Open=VALUES(Open), 
        High=VALUES(High), 
        Low=VALUES(Low), 
        Close=VALUES(Close), 
        Volume=VALUES(Volume)
    """
    
    rows_affected = execute_stock_upsert(conn, insert_query, records_to_insert)

    if rows_affected > 0:
        print(f"✅ Berhasil memproses {len(records_to_insert)} data. Total baris terpengaruh: {rows_affected}.")
    elif rows_affected == 0:
         print("⚠️ Tidak ada perubahan yang terjadi (kemungkinan data sudah ada).")
    else:
        print("❌ Terjadi kegagalan saat eksekusi batch query.")

    if conn and conn.is_connected():
        conn.close()
    print("--- Proses Batch Update Selesai ---")

if __name__ == "__main__":
    batch_update_stock_prices()
