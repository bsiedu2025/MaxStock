# -*- coding: utf-8 -*-
# app/pages/7_Import_KSEI_Bulanan.py
# Upload KSEI bulanan (RAW) → ksei_month. Tidak mengubah halaman lain.

import io
import os
import tempfile
import pandas as pd
import streamlit as st
import mysql.connector
from mysql.connector import pooling

st.set_page_config(page_title="📥 Import KSEI Bulanan", page_icon="📥", layout="wide")
st.title("📥 Import KSEI Bulanan → `ksei_month`")

# ── DB helpers
def _get_db_params():
    return {
        "host": os.getenv("DB_HOST", st.secrets.get("DB_HOST", "")),
        "port": int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306))),
        "database": os.getenv("DB_NAME", st.secrets.get("DB_NAME", "")),
        "user": os.getenv("DB_USER", st.secrets.get("DB_USER", "")),
        "password": os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", "")),
        "ssl_ca_str": os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", "")),
    }

def get_pool():
    params = _get_db_params()
    ssl_ca_file = None
    ssl_kwargs = {}
    if params["ssl_ca_str"] and "BEGIN CERTIFICATE" in params["ssl_ca_str"]:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(params["ssl_ca_str"].encode("utf-8")); tmp.flush()
        ssl_ca_file = tmp.name
        ssl_kwargs = {"ssl_ca": ssl_ca_file}

    pool = pooling.MySQLConnectionPool(
        pool_name="ksei_month_pool",
        pool_size=3,
        autocommit=False,
        host=params["host"],
        port=params["port"],
        database=params["database"],
        user=params["user"],
        password=params["password"],
        **ssl_kwargs,
    )
    return pool, ssl_ca_file

def ensure_table_ksei_month(conn):
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ksei_month (
      base_symbol VARCHAR(32)  NOT NULL,
      trade_date  DATE         NOT NULL,
      price       DECIMAL(19,4)     NULL,

      local_is BIGINT NULL, local_cp BIGINT NULL, local_pf BIGINT NULL, local_ib BIGINT NULL,
      local_id BIGINT NULL, local_mf BIGINT NULL, local_sc BIGINT NULL, local_fd BIGINT NULL, local_ot BIGINT NULL,
      local_total BIGINT NULL,

      foreign_is BIGINT NULL, foreign_cp BIGINT NULL, foreign_pf BIGINT NULL, foreign_ib BIGINT NULL,
      foreign_id BIGINT NULL, foreign_mf BIGINT NULL, foreign_sc BIGINT NULL, foreign_fd BIGINT NULL, foreign_ot BIGINT NULL,
      foreign_total BIGINT NULL,

      source_file VARCHAR(255) NULL,
      created_at  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (base_symbol, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """)
    cur.close()

# ── File parsing
def detect_sep(sample_line: str) -> str:
    if sample_line.count("|") >= 2: return "|"
    if sample_line.count("\t") >= 2: return "\t"
    if sample_line.count(";") >= 2: return ";"
    return ","

def read_any(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    first = text.splitlines()[0] if text else ""
    sep = detect_sep(first)
    try:
        df = pd.read_csv(io.StringIO(text), sep=sep)
    except Exception:
        df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    return df

def coerce_to_ksei_month(df: pd.DataFrame) -> pd.DataFrame:
    # Map kolom sesuai contoh KSEI monthly
    df2 = pd.DataFrame()
    # Date
    if "Date" in df.columns:
        df2["trade_date"] = pd.to_datetime(df["Date"], errors="coerce", format="%d-%b-%Y").dt.date
    else:
        df2["trade_date"] = pd.to_datetime(df.iloc[:,0], errors="coerce").dt.date
    # Code
    code_col = "Code" if "Code" in df.columns else df.columns[1]
    df2["base_symbol"] = df[code_col].astype(str).str.upper().str.strip().str.replace(r"\s+FF$", "", regex=True)
    # Price
    if "Price" in df.columns:
        df2["price"] = pd.to_numeric(df["Price"], errors="coerce")
    else:
        df2["price"] = None

    # Helper
    def num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else None

    # Local categories
    df2["local_is"] = num("Local IS"); df2["local_cp"] = num("Local CP"); df2["local_pf"] = num("Local PF")
    df2["local_ib"] = num("Local IB"); df2["local_id"] = num("Local ID"); df2["local_mf"] = num("Local MF")
    df2["local_sc"] = num("Local SC"); df2["local_fd"] = num("Local FD"); df2["local_ot"] = num("Local OT")
    # Foreign categories
    df2["foreign_is"] = num("Foreign IS"); df2["foreign_cp"] = num("Foreign CP"); df2["foreign_pf"] = num("Foreign PF")
    df2["foreign_ib"] = num("Foreign IB"); df2["foreign_id"] = num("Foreign ID"); df2["foreign_mf"] = num("Foreign MF")
    df2["foreign_sc"] = num("Foreign SC"); df2["foreign_fd"] = num("Foreign FD"); df2["foreign_ot"] = num("Foreign OT")

    # Totals: "Total" (local) dan "Total.1" (foreign) pada file contoh
    local_total = df["Total"] if "Total" in df.columns else None
    foreign_total = df["Total.1"] if "Total.1" in df.columns else None
    if local_total is None:
        loc_cols = [c for c in df.columns if isinstance(c,str) and c.startswith("Local ")]
        local_total = pd.to_numeric(df[loc_cols], errors="coerce").fillna(0).sum(axis=1) if loc_cols else None
    if foreign_total is None:
        for_cols = [c for c in df.columns if isinstance(c,str) and c.startswith("Foreign ")]
        foreign_total = pd.to_numeric(df[for_cols], errors="coerce").fillna(0).sum(axis=1) if for_cols else None

    df2["local_total"] = pd.to_numeric(local_total, errors="coerce")
    df2["foreign_total"] = pd.to_numeric(foreign_total, errors="coerce")

    # Source file (optional)
    df2["source_file"] = None

    # Clean
    df2 = df2.dropna(subset=["trade_date","base_symbol"])
    df2 = df2.drop_duplicates(subset=["base_symbol","trade_date"])
    return df2

# ── UI
up = st.file_uploader("Pilih file KSEI bulanan (CSV/TXT). Format seperti contoh resmi KSEI.", type=["csv","txt"])
if up is not None:
    df_raw = read_any(up)
    st.write("Preview file (atas 20 baris):")
    st.dataframe(df_raw.head(20), use_container_width=True)

    dfkm = coerce_to_ksei_month(df_raw)
    st.write("Preview data yang akan disimpan (atas 20 baris):")
    st.dataframe(dfkm.head(20), use_container_width=True)

    if st.button("🚀 INSERT ke `ksei_month` (INSERT IGNORE)"):
        pool, ssl_ca_file = get_pool()
        conn = pool.get_connection()
        try:
            ensure_table_ksei_month(conn)
            cur = conn.cursor()
            sql = """
            INSERT IGNORE INTO ksei_month
            (base_symbol, trade_date, price,
             local_is, local_cp, local_pf, local_ib, local_id, local_mf, local_sc, local_fd, local_ot, local_total,
             foreign_is, foreign_cp, foreign_pf, foreign_ib, foreign_id, foreign_mf, foreign_sc, foreign_fd, foreign_ot, foreign_total,
             source_file)
            VALUES
            (%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
             %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
             %s)
            """
            rows = [tuple(r) for r in dfkm[[
                "base_symbol","trade_date","price",
                "local_is","local_cp","local_pf","local_ib","local_id","local_mf","local_sc","local_fd","local_ot","local_total",
                "foreign_is","foreign_cp","foreign_pf","foreign_ib","foreign_id","foreign_mf","foreign_sc","foreign_fd","foreign_ot","foreign_total",
                "source_file"
            ]].itertuples(index=False, name=None)]
            cur.executemany(sql, rows)
            conn.commit()
            st.success(f"Berhasil insert (abaikan duplikat): {cur.rowcount} baris.")
            cur.close()
        finally:
            try:
                conn.close()
            except Exception:
                pass
            if ssl_ca_file and os.path.exists(ssl_ca_file):
                try: os.unlink(ssl_ca_file)
                except Exception: pass

st.info("Tips: Tambahkan file ini ke menu melalui app_main.py agar muncul sebagai halaman baru.")
