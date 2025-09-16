# D:\Docker\BrokerSummary\app\db_utils.py
import mysql.connector
import pandas as pd
import streamlit as st 
import os
from datetime import datetime, date
import yfinance as yf

# --- Konfigurasi Koneksi Database ---
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = int(os.environ.get('DB_PORT', 3307))
DB_USER = os.environ.get('DB_USER', 'admin')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'adminpassword')
DB_NAME = os.environ.get('DB_NAME', 'Broker_DB')

def get_db_connection():
    """Membuat dan mengembalikan koneksi ke database MariaDB."""
    try:
        conn = mysql.connector.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME
        )
        return conn
    except mysql.connector.Error as err:
        print(f"Error koneksi ke MariaDB: {err}")
        if 'st' in globals() and hasattr(st, 'error'): 
            st.error(f"Gagal terhubung ke database: {err}")
        return None

def execute_query(query, params=None, fetch_one=False, fetch_all=False, is_dml_ddl=False):
    """
    Menjalankan query SQL secara generik.
    Mengembalikan tuple: (hasil, error_message).
    """
    conn = get_db_connection()
    if conn is None:
        return (None if fetch_one or fetch_all else False, "Gagal mendapatkan koneksi database.")

    cursor = None
    try:
        cursor = conn.cursor(dictionary=(fetch_one or fetch_all)) 
        cursor.execute(query, params or ())

        if is_dml_ddl:
            conn.commit()
            return (cursor.rowcount if cursor.rowcount != -1 else True, None)
        elif fetch_one:
            return (cursor.fetchone(), None)
        elif fetch_all:
            return (cursor.fetchall(), None)
        return (True, None) 
    except mysql.connector.Error as err:
        error_msg = f"Error SQL: {err}"
        print(error_msg) 
        if is_dml_ddl and conn.is_connected(): 
            try: conn.rollback(); print("Rollback berhasil dilakukan karena error SQL.")
            except mysql.connector.Error as rb_err: print(f"Error saat rollback: {rb_err}")
        return (None if fetch_one or fetch_all else False, error_msg)
    except Exception as e_gen: 
        error_msg = f"Error umum saat eksekusi query: {e_gen}"
        print(error_msg)
        if conn and conn.is_connected() and is_dml_ddl: 
             try: conn.rollback(); print("Rollback berhasil dilakukan karena error umum.")
             except mysql.connector.Error as rb_err: print(f"Error saat rollback: {rb_err}")
        return (None if fetch_one or fetch_all else False, error_msg)
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def create_tables_if_not_exist():
    """Membuat semua tabel yang dibutuhkan oleh aplikasi jika belum ada."""
    conn = get_db_connection()
    if conn is None: print("Tidak dapat membuat tabel, koneksi database gagal."); return
    cursor = conn.cursor()
    try:
        # Menghapus tabel yang tidak digunakan lagi (optional)
        # try: cursor.execute("DROP TABLE IF EXISTS transactions_raw;"); conn.commit()
        # except mysql.connector.Error: pass 
        # try: cursor.execute("DROP TABLE IF EXISTS broker_info_raw;"); conn.commit()
        # except mysql.connector.Error: pass 
        # try: cursor.execute("DROP TABLE IF EXISTS broker_summary;"); conn.commit()
        # except mysql.connector.Error: pass

        # BARU: Hanya membuat tabel yang relevan
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS stock_prices_history (
            Ticker VARCHAR(20) NOT NULL,       
            Tanggal DATE NOT NULL,             
            Open DECIMAL(19,4),                
            High DECIMAL(19,4),                
            Low DECIMAL(19,4),                 
            Close DECIMAL(19,4),               
            Volume BIGINT,                     
            PRIMARY KEY (Ticker, Tanggal)      
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """)
        
        conn.commit()
        print("Pemeriksaan dan/atau pembuatan tabel (stock_prices_history) selesai.")
    except mysql.connector.Error as err:
        print(f"Error saat memeriksa/membuat tabel: {err}")
        if conn.is_connected(): conn.rollback()
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()


def insert_stock_price_data(df_stock_prices, ticker_symbol):
    conn = get_db_connection()
    if conn is None or df_stock_prices.empty:
        if 'st' in globals() and hasattr(st, 'warning'): st.warning("Koneksi DB gagal atau DataFrame harga saham kosong.")
        return 0 
    cursor = None; inserted_count = 0
    query = "INSERT IGNORE INTO stock_prices_history (Ticker, Tanggal, Open, High, Low, Close, Volume) VALUES (%s, %s, %s, %s, %s, %s, %s);"
    try:
        cursor = conn.cursor()
        for index_date, row in df_stock_prices.iterrows():
            tanggal = index_date.date()
            open_price = float(row['Open']) if pd.notna(row['Open']) else None; high_price = float(row['High']) if pd.notna(row['High']) else None
            low_price = float(row['Low']) if pd.notna(row['Low']) else None; close_price = float(row['Close']) if pd.notna(row['Close']) else None
            volume = int(row['Volume']) if pd.notna(row['Volume']) else None
            if ticker_symbol and tanggal:
                cursor.execute(query, (ticker_symbol, tanggal, open_price, high_price, low_price, close_price, volume))
                inserted_count += cursor.rowcount
            else:
                if 'st' in globals() and hasattr(st, 'warning'): st.warning(f"Ticker/Tanggal kosong untuk indeks: {index_date}, dilewati.")
        conn.commit()
        if 'st' in globals():
            if inserted_count > 0: st.success(f"{inserted_count} data harga saham baru {ticker_symbol} berhasil dimasukkan/diabaikan.")
            else: st.info(f"Tidak ada data harga saham baru {ticker_symbol} dimasukkan (mungkin sudah ada/kosong).")
        return inserted_count
    except mysql.connector.Error as err:
        if 'st' in globals(): st.error(f"Error MySQL saat proses data harga saham {ticker_symbol}: {err}")
        if conn and conn.is_connected(): conn.rollback(); return 0
    except Exception as e_gen:
        if 'st' in globals(): st.error(f"Error umum saat proses data harga saham: {e_gen}")
        if conn and conn.is_connected(): conn.rollback(); return 0
    finally:
        if cursor: cursor.close()
        if conn and conn.is_connected(): conn.close()

def fetch_stock_prices_from_db(ticker_symbol, start_date=None, end_date=None):
    conn = get_db_connection()
    if conn is None: 
        if 'st' in globals() and hasattr(st, 'error'): st.error(f"Koneksi DB gagal ambil harga saham {ticker_symbol}.")
        return pd.DataFrame()
    query = "SELECT Tanggal, Open, High, Low, Close, Volume FROM stock_prices_history WHERE Ticker = %s"
    params = [ticker_symbol]
    start_date_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, date) else None
    end_date_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, date) else None
    
    if start_date_str and end_date_str: query += " AND Tanggal BETWEEN %s AND %s"; params.extend([start_date_str, end_date_str])
    elif start_date_str: query += " AND Tanggal >= %s"; params.append(start_date_str)
    elif end_date_str: query += " AND Tanggal <= %s"; params.append(end_date_str)
    query += " ORDER BY Tanggal ASC;"
    try:
        df = pd.read_sql_query(query, conn, params=params, index_col='Tanggal')
        if not df.empty: df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        if 'st' in globals() and hasattr(st, 'error'): st.error(f"Error ambil data harga saham dari DB {ticker_symbol}: {e}")
        return pd.DataFrame()
    finally:
        if conn and conn.is_connected(): conn.close()

def get_saved_tickers_summary():
    query = """
    SELECT sph.Ticker, COUNT(*) as Jumlah_Data, MIN(sph.Tanggal) as Tanggal_Awal, MAX(sph.Tanggal) as Tanggal_Terakhir,
           (SELECT sp_last.Close FROM stock_prices_history sp_last WHERE sp_last.Ticker = sph.Ticker ORDER BY sp_last.Tanggal DESC LIMIT 1) as Harga_Penutupan_Terakhir,
           MAX(sph.High) as Harga_Tertinggi_Periode, MIN(sph.Low) as Harga_Terendah_Periode
    FROM stock_prices_history sph GROUP BY sph.Ticker ORDER BY sph.Ticker ASC;
    """
    result, error = execute_query(query, fetch_all=True) 
    default_cols = ['Ticker', 'Jumlah_Data', 'Tanggal_Awal', 'Tanggal_Terakhir', 'Harga_Penutupan_Terakhir', 'Harga_Tertinggi_Periode', 'Harga_Terendah_Periode']
    if error:
        if 'st' in globals() and hasattr(st, 'warning'): st.warning(f"Gagal mengambil ringkasan ticker: {error}")
        return pd.DataFrame(columns=default_cols)
    if result:
        df = pd.DataFrame(result)
        if 'Tanggal_Awal' in df.columns: df['Tanggal_Awal'] = pd.to_datetime(df['Tanggal_Awal'], errors='coerce')
        if 'Tanggal_Terakhir' in df.columns: df['Tanggal_Terakhir'] = pd.to_datetime(df['Tanggal_Terakhir'], errors='coerce')
        return df
    return pd.DataFrame(columns=default_cols)

def get_table_list():
    query = "SHOW TABLES;"
    result, error = execute_query(query, fetch_all=True) 
    if error: 
        if 'st' in globals() and hasattr(st, 'error'): st.error(f"Gagal mengambil daftar tabel: {error}")
        return []
    if result and isinstance(result, list) and len(result) > 0:
        if isinstance(result[0], dict): 
            key_name = list(result[0].keys())[0] 
            return [row[key_name] for row in result]
        elif isinstance(result[0], tuple): return [row[0] for row in result] 
    return []

@st.cache_data(ttl=300) 
def get_stock_info(ticker_symbol):
    try:
        stock = yf.Ticker(ticker_symbol)
        info = stock.info 
        if not info or info.get('regularMarketPrice') is None and info.get('currentPrice') is None :
            fallback_hist = stock.history(period="5d", auto_adjust=True)
            if not fallback_hist.empty:
                if not isinstance(info, dict): info = {} 
                info['regularMarketPrice'] = fallback_hist['Close'].iloc[-1]
                info['previousClose'] = fallback_hist['Close'].iloc[-2] if len(fallback_hist) > 1 else info.get('regularMarketPrice')
                info['volume'] = fallback_hist['Volume'].iloc[-1]; info['dayHigh'] = fallback_hist['High'].iloc[-1]; info['dayLow'] = fallback_hist['Low'].iloc[-1]
                for key_original in ['marketCap', 'trailingPE', 'forwardPE', 'shortName', 'longName', 'currency', 'exchange']:
                    if key_original not in info and hasattr(stock, 'info') and stock.info and key_original in stock.info:
                        info[key_original] = stock.info[key_original]
            else:
                if not isinstance(info, dict): info = {} 
        return info if isinstance(info, dict) else {} 
    except Exception as e:
        print(f"Error yfinance saat ambil info {ticker_symbol}: {e}")
        return {}

def get_distinct_tickers_from_price_history_with_suffix():
    """Mengambil daftar ticker unik dari tabel stock_prices_history, mempertahankan suffix .JK"""
    query = "SELECT DISTINCT Ticker FROM stock_prices_history ORDER BY Ticker ASC;"
    result, error = execute_query(query, fetch_all=True)
    if error or not result:
        return []
    return [row['Ticker'] for row in result if row['Ticker']]