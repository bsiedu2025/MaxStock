# app/db_utils.py
import os
import psycopg2
import streamlit as st
from typing import Dict, Any, Tuple, Optional


# ----------------------------
# Utilities
# ----------------------------
REQUIRED_KEYS = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]

def debug_secrets() -> None:
    """Tampilkan key yang terbaca dari st.secrets (untuk debug di UI)."""
    try:
        keys = list(st.secrets.keys())
        st.caption("🔑 Keys yang ditemukan di st.secrets:")
        st.code(", ".join(keys) if keys else "(kosong)")
    except Exception as e:
        st.error(f"Gagal membaca st.secrets: {e}")


def _load_db_config() -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Baca konfigurasi DB dari st.secrets (prioritas) atau ENV (fallback).
    Return: (config_dict, error_message_if_any)
    """
    cfg: Dict[str, Any] = {}
    missing = []

    # 1) Ambil dari st.secrets kalau ada
    try:
        for k in REQUIRED_KEYS:
            if k in st.secrets:
                cfg[k] = str(st.secrets[k]).strip()
    except Exception:
        # st.secrets belum tersedia (mis. saat run lokal tanpa .streamlit/secrets.toml)
        pass

    # 2) Fallback ke environment variables kalau masih kosong
    for k in REQUIRED_KEYS:
        if k not in cfg or cfg[k] == "":
            env_val = os.getenv(k)
            if env_val:
                cfg[k] = env_val.strip()

    # 3) Validasi
    for k in REQUIRED_KEYS:
        if k not in cfg or cfg[k] == "":
            missing.append(k)

    if missing:
        return cfg, (
            "Secrets DB belum lengkap. Harus ada "
            + ", ".join(REQUIRED_KEYS)
            + f". (Missing: {', '.join(missing)})"
        )

    return cfg, None


def check_secrets(show_in_ui: bool = True) -> bool:
    """Cek apakah semua key DB ada. Tampilkan pesan kalau belum lengkap."""
    _, err = _load_db_config()
    if err:
        if show_in_ui:
            st.error(f"Gagal terhubung ke database: {err}")
        return False
    return True


# ----------------------------
# Connection
# ----------------------------
def get_connection():
    """
    Buat koneksi psycopg2 ke Supabase Postgres (SSL required).
    Dipakai oleh halaman-halaman lain.
    """
    cfg, err = _load_db_config()
    if err:
        # Sudah ditampilkan oleh check_secrets() di app_main; 
        # di sini raise supaya caller bisa handle.
        raise RuntimeError(err)

    try:
        conn = psycopg2.connect(
            host=cfg["DB_HOST"],
            port=cfg["DB_PORT"],
            dbname=cfg["DB_NAME"],
            user=cfg["DB_USER"],
            password=cfg["DB_PASSWORD"],
            sslmode="require",
            connect_timeout=15,
        )
        return conn
    except Exception as e:
        # Perlihatkan error di UI agar mudah didiagnosa
        st.error(f"Gagal membuka koneksi ke Postgres: {e}")
        raise


# Alias untuk kompatibilitas impor lama
get_db_connection = get_connection


# ----------------------------
# Schema / Setup
# ----------------------------
DDL_STOCK_PRICES = """
CREATE TABLE IF NOT EXISTS public.stock_prices_history (
  "Ticker"  TEXT NOT NULL,
  "Tanggal" DATE NOT NULL,
  "Open"    DOUBLE PRECISION,
  "High"    DOUBLE PRECISION,
  "Low"     DOUBLE PRECISION,
  "Close"   DOUBLE PRECISION,
  "Volume"  DOUBLE PRECISION,
  CONSTRAINT stock_prices_history_pk PRIMARY KEY ("Ticker","Tanggal")
);
"""

def create_tables_if_not_exist() -> None:
    """
    Membuat tabel yang dibutuhkan jika belum ada.
    Aman dipanggil berkali-kali (idempotent).
    """
    if not check_secrets(show_in_ui=True):
        return

    conn = None
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(DDL_STOCK_PRICES)
        conn.commit()
        st.success("✅ Inisialisasi schema selesai (tabel dicek/dibuat).")
    except Exception as e:
        st.error(f"Gagal membuat/memvalidasi tabel: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
