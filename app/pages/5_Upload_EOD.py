\
import io
import os
import re
import ssl
import tempfile
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import mysql.connector

st.set_page_config(page_title="Upload EOD & Import KSEI", page_icon="📤", layout="wide")

st.title("📤 Upload EOD (CSV) & 📥 Import KSEI")
st.caption("Satu halaman untuk input EOD raw ke `eod_prices_raw` dan data partisipasi ke `ksei_daily`.")

# -------------------- DB UTILS --------------------
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

def ensure_table_eod(conn):
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
    cur.execute(ddl); cur.close()

def ensure_table_ksei(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS ksei_daily (
      base_symbol  VARCHAR(32)  NOT NULL,
      trade_date   DATE         NOT NULL,
      foreign_pct  DECIMAL(5,2)     NULL,
      retail_pct   DECIMAL(5,2)     NULL,
      total_volume BIGINT            NULL,
      total_value  BIGINT            NULL,
      created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (base_symbol, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    cur = conn.cursor(); cur.execute(ddl)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ksei_trade_date ON ksei_daily (trade_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ksei_base_symbol ON ksei_daily (base_symbol)")
    except mysql.connector.Error:
        # MySQL <8 doesn't support IF NOT EXISTS on CREATE INDEX; try naive create
        try: cur.execute("CREATE INDEX idx_ksei_trade_date ON ksei_daily (trade_date)")
        except: pass
        try: cur.execute("CREATE INDEX idx_ksei_base_symbol ON ksei_daily (base_symbol)")
        except: pass
    cur.close()

# -------------------- HELPERS (EOD) --------------------
CANON = ["date","ticker","open","high","low","close","volume","oi"]

ALIASES = {
    "date": {"date","tanggal","tgl","<date>"},
    "ticker": {"ticker","symbol","kode","<ticker>","base_symbol"},
    "open": {"open","o","<open>"},
    "high": {"high","h","<high>"},
    "low": {"low","l","<low>"},
    "close": {"close","c","<close>","adjclose","adj_close"},
    "volume": {"volume","vol","<volume>","volume_price","total_volume"},
    "oi": {"oi","openinterest","open_interest","<oi>"},
}

def normalize_col(c: str) -> str:
    c = c.strip().lower()
    c = c.replace(" ", "").replace("-", "").replace(".", "")
    c = c.replace("[","").replace("]","")
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
    if pd.isna(s): return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%d-%m-%Y","%m-%d-%Y"):
        try: return datetime.strptime(s, fmt).date()
        except ValueError: continue
    try: return pd.to_datetime(s, errors="coerce").date()
    except Exception: return None

# -------------------- HELPERS (KSEI) --------------------
KSEI_ALIASES = {
    "base_symbol": {"base_symbol","symbol","ticker","kode","saham"},
    "trade_date": {"trade_date","date","tanggal","tgl"},
    "foreign_pct": {"foreign_pct","pa_pct","asing_pct","foreignpercent","asingpercent"},
    "retail_pct": {"retail_pct","ri_pct","retailpercent"},
    "total_volume": {"total_volume","volume","vol","vol_total"},
    "total_value": {"total_value","nilai","value","val_total"},
}

def _norm_map(df: pd.DataFrame, aliases: Dict[str, set]) -> Dict[str,str]:
    mapping = {}
    cols = [normalize_col(c) for c in df.columns]
    for canon, names in aliases.items():
        for raw, norm in zip(df.columns, cols):
            if norm in names:
                mapping[canon] = raw
                break
    return mapping

def coerce_ksei_df_local(df: pd.DataFrame) -> pd.DataFrame:
    mapping = _norm_map(df, KSEI_ALIASES)
    out = pd.DataFrame()
    # required
    if "trade_date" in mapping:
        out["trade_date"] = pd.to_datetime(df[mapping["trade_date"]], errors="coerce").dt.date
    else:
        out["trade_date"] = pd.to_datetime(df.iloc[:,0], errors="coerce").dt.date
    if "base_symbol" in mapping:
        out["base_symbol"] = df[mapping["base_symbol"]].astype(str).str.strip().str.upper()
    else:
        out["base_symbol"] = df.iloc[:,1].astype(str).str.strip().str.upper() if df.shape[1] > 1 else ""

    # optional numerics
    def to_num(s): return pd.to_numeric(s, errors="coerce")
    out["foreign_pct"]  = to_num(df[mapping["foreign_pct"]])  if "foreign_pct"  in mapping else None
    out["retail_pct"]   = to_num(df[mapping["retail_pct"]])   if "retail_pct"   in mapping else None
    out["total_volume"] = to_num(df[mapping["total_volume"]]) if "total_volume" in mapping else None
    out["total_value"]  = to_num(df[mapping["total_value"]])  if "total_value"  in mapping else None

    # cleanup
    out = out.dropna(subset=["trade_date","base_symbol"])
    out["base_symbol"] = out["base_symbol"].str.replace(r"\s+FF$", "", regex=True).str.strip()
    out = out.drop_duplicates(subset=["base_symbol","trade_date"])
    return out

def parse_balancepos_txt_local(uploaded_file) -> pd.DataFrame:
    # Try common delimiters
    raw = uploaded_file.read()
    for enc in ("utf-8","latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            continue
    # Detect delimiter by header
    head = text.splitlines()[0]
    delim = ";" if head.count(";") >= head.count(",") and head.count(";") >= head.count("\t") else ("," if head.count(",") >= head.count("\t") else "\t")
    df = pd.read_csv(io.StringIO(text), delimiter=delim)
    return df

# -------------------- UI (Tabs) --------------------
tab1, tab2 = st.tabs(["📤 Upload EOD (CSV)", "📥 Import KSEI (CSV/TXT)"])

with tab1:
    st.subheader("Upload EOD RAW → `eod_prices_raw`")
    source_hint = st.text_input("Nama sumber/SourceFile (opsional)", value="EOD.csv")
    uploaded = st.file_uploader("Pilih file CSV EOD", type=["csv"], key="eod_uploader")
    if uploaded:
        raw = uploaded.read()
        try:
            df = pd.read_csv(io.BytesIO(raw))
        except Exception:
            df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
        st.dataframe(df.head(20), use_container_width=True)

        mapping = auto_map_columns(df)
        st.write("Pemetaan kolom otomatis →", mapping)

        missing = [k for k in CANON if k not in mapping]
        if missing:
            st.warning(f"Kolom wajib yang belum terdeteksi: {missing}. Silakan rename kolom di CSV Anda atau pilih manual di bawah.")
            cols = list(df.columns)
            for k in missing:
                choice = st.selectbox(f"Pilih kolom untuk **{k}**", options=["--"]+cols, key=f"sel_{k}")
                if choice != "--":
                    mapping[k] = choice
        if not all(k in mapping for k in ["date","ticker","open","high","low","close","volume","oi"]):
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
        before = len(tmp)
        tmp = tmp.dropna(subset=["Tanggal","Ticker"])
        st.info(f"Baris valid: {len(tmp)}/{before}")

        with st.expander("Lihat data siap insert"):
            st.dataframe(tmp.head(50), use_container_width=True)

        if st.button("🚀 Insert ke `eod_prices_raw` (INSERT IGNORE)", type="primary"):
            conn, ssl_ca_file = get_conn()
            try:
                ensure_table_eod(conn)
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
                conn.commit(); cur.close()
                st.success(f"Selesai. Row ditambahkan (non-duplikat): {affected}.")
            finally:
                close_conn((conn, ssl_ca_file))

with tab2:
    st.subheader("Import KSEI → `ksei_daily`")
    st.caption("Menerima CSV/TXT hasil Balancepos. Otomatis normalisasi kolom (base_symbol, trade_date, foreign_pct, retail_pct, total_volume, total_value).")
    up = st.file_uploader("Pilih file KSEI (CSV/TXT)", type=["csv","txt"], key="ksei_uploader")

    # Toggle untuk memakai fungsi lokal atau fungsi utilitas eksternal (jika tersedia)
    use_utils = st.checkbox("Gunakan utils.coerce_ksei_df / parse_balancepos_txt jika tersedia", value=True)

    if up:
        filename = up.name.lower()
        df_ksei = None

        if use_utils:
            try:
                from utils import coerce_ksei_df as _coerce_ksei_df_ext
                from utils import parse_balancepos_txt as _parse_balancepos_txt_ext
                if filename.endswith(".csv"):
                    df_ksei = pd.read_csv(up)
                else:
                    df_ksei = _parse_balancepos_txt_ext(up)
                df_ksei = _coerce_ksei_df_ext(df_ksei)
            except Exception as e:
                st.warning(f"Gagal menggunakan utils eksternal: {e}. Pakai parser lokal.")
                up.seek(0)
                if filename.endswith(".csv"):
                    df_ksei = pd.read_csv(up)
                else:
                    df_ksei = parse_balancepos_txt_local(up)
                df_ksei = coerce_ksei_df_local(df_ksei)
        else:
            if filename.endswith(".csv"):
                df_ksei = pd.read_csv(up)
            else:
                df_ksei = parse_balancepos_txt_local(up)
            df_ksei = coerce_ksei_df_local(df_ksei)

        st.write("Preview Data (dinormalisasi):")
        st.dataframe(df_ksei.head(30), use_container_width=True)

        if st.button("🚀 Insert ke `ksei_daily` (INSERT IGNORE)"):
            conn, ssl_ca_file = get_conn()
            try:
                ensure_table_ksei(conn)
                cur = conn.cursor()
                sql = """
                INSERT IGNORE INTO ksei_daily
                (base_symbol, trade_date, foreign_pct, retail_pct, total_volume, total_value)
                VALUES (%s,%s,%s,%s,%s,%s)
                """
                data = [
                    (
                        str(r.base_symbol).strip().upper(),
                        r.trade_date,
                        float(r.foreign_pct) if pd.notna(r.foreign_pct) else None,
                        float(r.retail_pct) if pd.notna(r.retail_pct) else None,
                        int(r.total_volume) if pd.notna(r.total_volume) else None,
                        int(r.total_value) if pd.notna(r.total_value) else None,
                    )
                    for r in df_ksei.itertuples(index=False)
                ]
                cur.executemany(sql, data)
                affected = cur.rowcount
                conn.commit(); cur.close()
                st.success(f"Selesai. Row ksei_daily ditambahkan (non-duplikat): {affected}.")
            finally:
                close_conn((conn, ssl_ca_file))
