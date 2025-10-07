import yfinance as yf
import pandas as pd
import os
import time
from datetime import datetime

# --- Konfigurasi ---
# Ganti dengan daftar saham kamu yang sebenarnya.
# Contoh: Ambil dari file CSV, atau hardcode seperti di bawah.
# Asumsi file daftar saham berada di root project, misal: 'list_saham.csv'
# Jika kamu menggunakan database (misal Firestore/PostgreSQL), ganti logika ini.
DAFTAR_SAHAM = ['BBCA.JK', 'TLKM.JK', 'ASII.JK', 'BMRI.JK', 'BBNI.JK', 'GOTO.JK'] 
OUTPUT_DIR = 'data'
START_DATE = '2023-01-01' # Ganti sesuai kebutuhan

def get_list_of_stocks():
    """Mengambil daftar kode saham untuk di-update."""
    # Karena tidak bisa lihat struktur file, kita pakai DAFTAR_SAHAM dari atas.
    # Jika kamu punya file CSV berisi daftar kode, kamu bisa ganti ini:
    # try:
    #     df_list = pd.read_csv('list_saham.csv')
    #     return df_list['KodeSaham'].tolist()
    # except FileNotFoundError:
    #     print("ERROR: File list_saham.csv tidak ditemukan. Menggunakan daftar default.")
    return DAFTAR_SAHAM

def fetch_and_save_data(stock_code, start_date=START_DATE):
    """Mengambil data harga saham menggunakan yfinance dan menyimpannya ke CSV."""
    file_path = os.path.join(OUTPUT_DIR, f'{stock_code}.csv')
    
    try:
        # Ambil data dari yfinance
        df = yf.download(stock_code, start=start_date, progress=False)
        
        if df.empty:
            print(f"[{stock_code}] WARNING: Data kosong.")
            return

        # Simpan ke CSV di folder 'data/'
        df.to_csv(file_path)
        print(f"[{stock_code}] SUKSES: Data diupdate dan disimpan ke {file_path}")

    except Exception as e:
        print(f"[{stock_code}] ERROR: Gagal mengambil/menyimpan data. Detail: {e}")

def main():
    """Fungsi utama untuk menjalankan proses bulk update."""
    
    print(f"--- Proses Bulk Update Saham Dimulai: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---")

    # 1. Pastikan folder output ada
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Folder '{OUTPUT_DIR}' dibuat.")

    # 2. Ambil daftar saham
    stocks_to_update = get_list_of_stocks()
    print(f"Total {len(stocks_to_update)} saham akan di-update.")
    
    # 3. Iterasi dan update
    for i, stock in enumerate(stocks_to_update):
        # Tambahkan jeda waktu kecil untuk menghindari limit rate API (jika ada)
        time.sleep(0.5) 
        print(f"Memproses {i+1}/{len(stocks_to_update)}: {stock}")
        fetch_and_save_data(stock)

    print(f"--- Proses Bulk Update Selesai ---")

if __name__ == "__main__":
    main()
