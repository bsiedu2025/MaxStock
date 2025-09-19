# app_main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import runpy
from pathlib import Path
from typing import Dict

import streamlit as st
from db_utils import (
    check_secrets,
    debug_secrets,
    get_db_connection,
    create_tables_if_not_exist,
)

st.set_page_config(
    page_title="Max Stock",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent

st.sidebar.title("app main")
status_placeholder = st.sidebar.empty()
status_placeholder.info("Memeriksa koneksi database…")

st.sidebar.markdown("---")
st.sidebar.subheader("Debug")
debug_secrets()

db_ok = False
if check_secrets(show_in_ui=True):
    try:
        with st.spinner("Menyambungkan ke database…"):
            conn = get_db_connection()
            conn.close()
        db_ok = True
        status_placeholder.success("Status Database: Terhubung ✅")
        with st.spinner("Inisialisasi schema (jika perlu)…"):
            create_tables_if_not_exist()
    except Exception as e:
        db_ok = False
        status_placeholder.error(f"Status Database: Gagal Terhubung ❌\n\n{e}")
else:
    status_placeholder.error("Status Database: Gagal Terhubung ❌")

st.title("Max Stock")
st.caption("Terhubung ke MySQL (Aiven) via Streamlit Cloud.")

st.header("Selamat Datang!")
st.write(
    """
Aplikasi ini membantu analisis data harga saham.
Gunakan sidebar untuk memilih halaman:
- **Harga Saham**: Visualisasi & indikator.
- **Update Data Harga Saham**: Unduh & simpan ke DB.
- **Konsol Database**: Jalankan query SQL langsung.
- **Sinyal MACD**: Pindai sinyal histogram hijau.
- **Upload EOD (CSV)**: Unggah data harian tanpa duplikat.
"""
)

if not db_ok:
    st.error("Gagal terhubung ke database. Beberapa fitur tidak akan berfungsi.")
    st.warning(
        "Pastikan Secrets berisi DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, dan DB_SSL_CA (Aiven mewajibkan SSL)."
    )

# ====== Daftar halaman ======
PAGE_FILES = {
  "Harga Saham": APP_DIR / "1_Harga_Saham.py",
  "Update Data Harga Saham": APP_DIR / "2_Update_Data_Harga_Saham.py",
  "Konsol Database": APP_DIR / "3_Konsol_Database.py",
  "Sinyal MACD": APP_DIR / "4_Sinyal_MACD.py",
  "Upload EOD (CSV)": APP_DIR / "5_Upload_EOD.py",
  "Import KSEI Bulanan": APP_DIR / "7_Import_KSEI_Bulanan.py",  # <— baru
  "Sinyal Harian (FF)": APP_DIR / "8_Signals_Harian.py",
}
available_pages = {n: p for n, p in PAGE_FILES.items() if p.exists()}

st.markdown("---")
st.subheader("Navigasi")
if not available_pages:
    st.info(
        "File halaman tidak ditemukan. Pastikan 1_Harga_Saham.py, 2_Update_Data_Harga_Saham.py, "
        "3_Konsol_Database.py, 4_Sinyal_MACD.py, 5_Upload_EOD.py ada di direktori app."
    )
else:
    page = st.sidebar.radio("Pilih halaman:", list(available_pages.keys()), index=0)
    try:
        runpy.run_path(str(available_pages[page]), run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        st.error(f"Gagal memuat halaman **{page}**: {e}")

st.markdown("---")
st.caption("© 2025 — Max Stock • Streamlit + Aiven MySQL")
