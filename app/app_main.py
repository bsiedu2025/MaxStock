# app/app_main.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
import runpy
from pathlib import Path
from typing import Dict

import streamlit as st

# Ambil util DB (versi yang sudah kita buat sebelumnya)
from db_utils import (
    check_secrets,
    debug_secrets,
    get_connection,           # tidak dipakai langsung di sini, tapi berguna untuk halaman lain
    get_db_connection,        # alias (kompatibilitas)
    create_tables_if_not_exist,
)

# -----------------------------------------------------------------------------
# Konfigurasi halaman
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Max Stock",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Sidebar - Brand & Status
# -----------------------------------------------------------------------------
st.sidebar.title("app main")

st.sidebar.markdown("### Status Koneksi")
db_ok = False
status_placeholder = st.sidebar.empty()
with status_placeholder.container():
    st.info("Memeriksa koneksi database…")

# Debug secrets (agar terlihat key apa saja yang terbaca)
st.sidebar.markdown("---")
st.sidebar.subheader("Debug")
debug_secrets()

# -----------------------------------------------------------------------------
# Validasi secrets & inisialisasi schema
# -----------------------------------------------------------------------------
if check_secrets(show_in_ui=True):
    try:
        with st.spinner("Menyambungkan ke database…"):
            # Coba buka koneksi cepat untuk health-check
            _conn = get_db_connection()
            _conn.close()
        db_ok = True
        status_placeholder.success("Status Database: Terhubung ✅")
        # Buat tabel jika belum ada
        with st.spinner("Inisialisasi schema (jika perlu)…"):
            create_tables_if_not_exist()
    except Exception as e:
        db_ok = False
        status_placeholder.error(f"Status Database: Gagal Terhubung ❌\n\n{e}")
else:
    db_ok = False
    status_placeholder.error("Status Database: Gagal Terhubung ❌")

st.sidebar.markdown("---")
st.sidebar.success("Pilih halaman di atas.")

# -----------------------------------------------------------------------------
# Router halaman
# -----------------------------------------------------------------------------
st.title("Max Stock")
st.caption("Terhubung ke database Supabase.")

st.markdown("---")
st.header("Selamat Datang!")
st.write(
    """
Aplikasi ini dirancang untuk membantu Anda menganalisis data transaksi broker saham.
Gunakan sidebar di sebelah kiri untuk navigasi ke halaman:
- **Input Data**: Untuk memasukkan atau memperbarui data transaksi dan informasi broker.
- **Analisis Broker**: Untuk melihat visualisasi dan analisis mendalam dari data yang ada.
"""
)

# Jika DB belum ok, tampilkan peringatan global
if not db_ok:
    st.error(
        "Gagal terhubung ke database. Beberapa fitur memerlukan koneksi DB dan mungkin tidak berfungsi."
    )
    st.warning(
        "Pastikan **Secrets** berisi `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` "
        "dan kredensialnya benar (password **Postgres** Supabase, bukan anon key)."
    )

# Pemetaan nama menu -> file python halaman
PAGE_FILES: Dict[str, Path] = {
    "Pergerakan IHSG": APP_DIR / "1_Harga_Saham.py",          # gunakan jika file ini berisi IHSG
    "Harga Saham": APP_DIR / "1_Harga_Saham.py",
    "Update Data Harga Saham": APP_DIR / "2_Update_Data_Harga_Saham.py",
    "Konsol Database": APP_DIR / "3_Konsol_Database.py",
    "Sinyal MACD": APP_DIR / "1_Harga_Saham.py",              # opsional: jika MACD ada di halaman 1
}

st.markdown("---")
st.subheader("Navigasi")

# Tampilkan hanya halaman yang file-nya benar2 ada (supaya tidak error di Cloud)
available_pages = {name: path for name, path in PAGE_FILES.items() if path.exists()}

if not available_pages:
    st.info(
        "Tidak menemukan file halaman. Pastikan file-file berikut ada di direktori app:\n\n"
        "- 1_Harga_Saham.py\n- 2_Update_Data_Harga_Saham.py\n- 3_Konsol_Database.py"
    )
else:
    default_page = next(iter(available_pages.keys()))
    page = st.sidebar.radio("Pilih halaman:", list(available_pages.keys()), index=0)
    st.write("")  # spacer

    # Jalankan file halaman yang dipilih
    page_path = available_pages[page]
    try:
        # Beri konteks variabel global minimal yang mungkin diperlukan halaman
        runpy.run_path(str(page_path), run_name="__main__")
    except SystemExit:
        # Beberapa halaman bisa memanggil sys.exit() saat rerun, abaikan agar tidak jatuh ke error page
        pass
    except Exception as e:
        st.error(f"Gagal memuat halaman **{page}** dari `{page_path.name}`:\n\n{e}")
        st.exception(e)

# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("© 2025 — Max Stock • Streamlit + Supabase")
