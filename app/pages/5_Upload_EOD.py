\
import io
import os
import re
import ssl
import tempfile
from datetime import datetime
from typing import Dict, List

import pandas as pd
import streamlit as st
import mysql.connector

st.set_page_config(page_title="Upload EOD (CSV)", page_icon="📤", layout="wide")

st.title("📤 Upload EOD (CSV) → `eod_prices_raw`")
st.caption("Auto-map header, preview, dan insert anti-duplikat (INSERT IGNORE).")

# --- DB utils
@st.cache_resource(show_spinner=False)
def get_conn():
    # Expect secrets/ENV: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, DB_SSL_CA
    host = os.getenv("DB_HOST", st.secrets.get("DB_HOST", ""))
    port = int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306)))
    database = os.getenv("DB_NAME", st.secrets.get("DB_NAME", ""))
    user = os.getenv("DB_USER", st.secrets.get("DB_USER", ""))
    password = os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", ""))
    ssl_ca = os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", ""))

    ssl_ca_file = None
    ssl_args = None
    if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(ssl_ca.encode("utf-8"))
        tmp.flush()
        ssl_ca_file = tmp.name
        ssl_args = {"ssl_ca": ssl_ca_file, "ssl_disabled": False}
    else:
        # Allow local testing without SSL_CA
        ssl_args = {"ssl_disabled": True}

    conn = mysql.connector.connect(
        host=host,
        port=port,
        database=database,
        user=user,
        password=password,
        **ssl_args,
        autocommit=True,
    )
    return conn, ssl_ca_file

def close_conn(conn_tuple):
    try:
        conn, ssl_ca_file = conn_tuple
        conn.close()
        if ssl_ca_file and os.path.exists(ssl_ca_file):
            os.remove(ssl_ca_file)
    except Exception:
        pass

def ensure_table(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS eod_prices_raw (
      Ticker     VARCHAR(32)  NOT NULL,
      Tanggal    DATE         NOT NULL,
      `Open`     DECIMAL(19,4)     NULL,
      `High`     DECIMAL(19,4)     NULL,
      `Low`      DECIMAL(19,4)     NULL,
      `Close`    DECIMAL(19,4)     NULL,
      Volume     BIGINT            NULL,
      OI         BIGINT            NULL,
      SourceFile VARCHAR(255)      NULL,
      created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (Ticker, Tanggal)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    cur = conn.cursor()
    cur.execute(ddl)
    cur.close()

# --- CSV mapping helpers
CANON = ["date","ticker","open","high","low","close","volume","oi"]

ALIASES = {
    "date": {"date","tanggal","tgl","<date>"},
    "ticker": {"ticker","symbol","kode","<ticker>"},
    "open": {"open","o","<open>"},
    "high": {"high","h","<high>"},
    "low": {"low","l","<low>"},
    "close": {"close","c","<close>","adjclose","adj_close"},
    "volume": {"volume","vol","<volume>"},
    "oi": {"oi","openinterest","open_interest","<oi>"},
}

def normalize_col(c: str) -> str:
    c = c.strip().lower()
    c = c.replace(" ", "").replace("-", "").replace(".", "")
    c = c.replace("[","").replace("]","")
    c = c.replace("<","<").replace(">",">")  # keep angle-brackets intact for matching
    return c

def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    mapping = {}
    cols = [normalize_col(c) for c in df.columns]
    for canon in CANON:
        found = None
        targets = ALIASES[canon]
        for raw, norm in zip(df.columns, cols):
            if norm in targets:
                found = raw
                break
        if found:
            mapping[canon] = found
    return mapping

def parse_date_any(s):
    # Accept mm/dd/yyyy, dd/mm/yyyy, yyyy-mm-dd, etc.
    if pd.isna(s):
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%d-%m-%Y","%m-%d-%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # Fallback: pandas to_datetime
    try:
        return pd.to_datetime(s, errors="coerce").date()
    except Exception:
        return None

# --- UI
uploaded = st.file_uploader("Pilih file CSV EOD", type=["csv"])
source_hint = st.text_input("Nama sumber/SourceFile (opsional)", value="EOD.csv")

if uploaded:
    raw = uploaded.read()
    # Try to read with utf-8 then latin-1
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    mapping = auto_map_columns(df)
    st.write("Pemetaan kolom otomatis →", mapping)

    missing = [k for k in CANON if k not in mapping]
    if missing:
        st.warning(f"Kolom wajib yang belum terdeteksi: {missing}. Silakan rename kolom di CSV Anda atau patch manual di bawah.")
        # Provide manual selectors
        cols = list(df.columns)
        for k in missing:
            mapping[k] = st.selectbox(f"Pilih kolom untuk **{k}**", options=["--"]+cols, key=f"sel_{k}")
        if any(v == "--" for v in mapping.values() if v is not None):
            st.stop()

    # Build canonical dataframe
    tmp = pd.DataFrame({
        "Tanggal": df[mapping["date"]].apply(parse_date_any),
        "Ticker": df[mapping["ticker"]].astype(str).str.strip(),
        "Open": pd.to_numeric(df[mapping["open"]], errors="coerce"),
        "High": pd.to_numeric(df[mapping["high"]], errors="coerce"),
        "Low": pd.to_numeric(df[mapping["low"]], errors="coerce"),
        "Close": pd.to_numeric(df[mapping["close"]], errors="coerce"),
        "Volume": pd.to_numeric(df[mapping["volume"]], errors="coerce").astype("Int64"),
        "OI": pd.to_numeric(df[mapping["oi"]], errors="coerce").astype("Int64"),
    })
    tmp["SourceFile"] = source_hint.strip() or os.path.basename(uploaded.name)

    # Drop bad rows
    before = len(tmp)
    tmp = tmp.dropna(subset=["Tanggal","Ticker"])
    st.info(f"Baris valid: {len(tmp)}/{before}")

    with st.expander("Lihat data siap insert"):
        st.dataframe(tmp.head(50), use_container_width=True)

    if st.button("🚀 Insert ke `eod_prices_raw` (INSERT IGNORE)", type="primary"):
        with st.spinner("Menulis ke database..."):
            conn, ssl_ca_file = get_conn()
            try:
                ensure_table(conn)
                cur = conn.cursor()
                sql = """
                INSERT IGNORE INTO eod_prices_raw
                (Ticker, Tanggal, `Open`, `High`, `Low`, `Close`, Volume, OI, SourceFile)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """
                data = [
                    (
                        r.Ticker, r.Tanggal, r.Open, r.High, r.Low, r.Close,
                        int(r.Volume) if pd.notna(r.Volume) else None,
                        int(r.OI) if pd.notna(r.OI) else None,
                        r.SourceFile
                    )
                    for r in tmp.itertuples(index=False)
                ]
                cur.executemany(sql, data)
                affected = cur.rowcount
                conn.commit()
                cur.close()
                st.success(f"Selesai. Row ditambahkan (non-duplikat): {affected}.")
            finally:
                close_conn((conn, ssl_ca_file))
