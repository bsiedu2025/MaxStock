# D:\Docker\BrokerSummary\app\pages\7_Konsol_Database.py
import streamlit as st
import pandas as pd
from db_utils import get_db_connection, execute_query, get_table_list, DB_NAME # Impor DB_NAME

st.set_page_config(page_title="Konsol Database", layout="wide")
st.title(" Database Console") # "Konsol Database" dalam Cyrillic untuk ikon terminal
st.markdown(f"Jalankan query SQL secara langsung pada database **`{DB_NAME}`**.")

# --- Menampilkan Informasi Database dan Tabel ---
st.sidebar.header("Info Database")
st.sidebar.info(f"Terhubung ke: **`{DB_NAME}`**")

st.sidebar.subheader("Daftar Tabel")
with st.spinner("Mengambil daftar tabel..."):
    tables = get_table_list()
if tables:
    for table_name in tables:
        st.sidebar.code(table_name, language='sql')
else:
    st.sidebar.warning("Tidak ada tabel ditemukan atau gagal mengambil daftar tabel.")
st.sidebar.markdown("---")

# --- Area Input Query ---
st.subheader("Masukkan Query SQL")
query_input = st.text_area("Query:", height=150, placeholder="Contoh: SELECT * FROM transactions_raw LIMIT 10;")

if st.button("🚀 Jalankan Query", type="primary"):
    if query_input.strip():
        with st.spinner("Menjalankan query..."):
            # Tentukan apakah query adalah SELECT atau DML/DDL
            # Ini adalah heuristik sederhana; query kompleks mungkin memerlukan penanganan lebih baik
            is_select_query = query_input.strip().upper().startswith("SELECT")
            
            if is_select_query:
                result, error_message = execute_query(query_input, fetch_all=True)
            else:
                # Untuk DML (INSERT, UPDATE, DELETE) atau DDL (CREATE, ALTER, DROP)
                result, error_message = execute_query(query_input, is_dml_ddl=True)

            st.markdown("---")
            st.subheader("Hasil Query")

            if error_message:
                st.error(f"Terjadi kesalahan saat menjalankan query:\n```\n{error_message}\n```")
            else:
                if is_select_query:
                    if result is not None:
                        if result: # Jika ada baris data
                            df_result = pd.DataFrame(result)
                            st.success(f"Query SELECT berhasil dijalankan, {len(df_result)} baris data ditemukan.")
                            st.dataframe(df_result, use_container_width=True)
                        else: # Query SELECT berhasil tapi tidak ada data
                            st.info("Query SELECT berhasil dijalankan, namun tidak ada data yang cocok.")
                    else: # Seharusnya tidak terjadi jika error_message None, tapi sebagai jaga-jaga
                        st.warning("Query SELECT mungkin berhasil tetapi tidak mengembalikan hasil yang diharapkan.")
                else: # Untuk DML/DDL
                    if isinstance(result, bool) and result:
                        st.success("Query DML/DDL berhasil dijalankan.")
                    elif isinstance(result, int): # rowcount dari DML
                        st.success(f"Query DML berhasil dijalankan, {result} baris terpengaruh.")
                    else: # Jika result bukan boolean True atau integer (misalnya, None dari error yang tidak tertangkap)
                        st.warning("Status eksekusi query DML/DDL tidak diketahui atau mungkin gagal tanpa pesan error eksplisit.")
    else:
        st.warning("Silakan masukkan query SQL untuk dijalankan.")

st.markdown("---")
st.caption("PERHATIAN: Menjalankan query DDL (seperti DROP TABLE) atau DML yang salah dapat merusak data Anda. Gunakan dengan hati-hati.")
st.caption("SELECT * FROM transactions_raw ORDER BY Tanggal_Transaksi DESC")
st.caption("SELECT Tanggal_Transaksi,COUNT(Kode_Broker) FROM transactions_raw GROUP  BY Tanggal_Transaksi ORDER BY Tanggal_Transaksi DESC")