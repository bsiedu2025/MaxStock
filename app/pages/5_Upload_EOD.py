import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import streamlit as st
import mysql.connector

st.set_page_config(page_title="Upload EOD & Import KSEI", page_icon="📤", layout="wide")

st.title("📤 Upload EOD (CSV) & 📥 Import KSEI")
st.caption("Satu halaman untuk input EOD raw ke `eod_prices_raw` (mendukung **multi-file/batch**) dan import KSEI.")

# =========================================================
# DB UTILS
# =========================================================
def get_conn():
    """Kembalikan koneksi MySQL baru setiap kali (no cache)."""
    host = os.getenv("DB_HOST", st.secrets.get("DB_HOST", ""))
    port = int(os.getenv("DB_PORT", st.secrets.get("DB_PORT", 3306)))
    database = os.getenv("DB_NAME", st.secrets.get("DB_NAME", ""))
    user = os.getenv("DB_USER", st.secrets.get("DB_USER", ""))
    password = os.getenv("DB_PASSWORD", st.secrets.get("DB_PASSWORD", ""))
    ssl_ca = os.getenv("DB_SSL_CA", st.secrets.get("DB_SSL_CA", ""))

    ssl_ca_file = None
    if ssl_ca and "BEGIN CERTIFICATE" in ssl_ca:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pem")
        tmp.write(ssl_ca.encode("utf-8"))
        tmp.flush()
        ssl_ca_file = tmp.name
        ssl_args = {"ssl_ca": ssl_ca_file, "ssl_disabled": False}
    else:
        ssl_args = {"ssl_disabled": True}

    conn = mysql.connector.connect(
        host=host, port=port, database=database,
        user=user, password=password, autocommit=True, **ssl_args
    )
    try:
        if hasattr(conn, "is_connected") and not conn.is_connected():
            conn.reconnect(attempts=2, delay=1)
    except Exception:
        pass
    return conn, ssl_ca_file

def close_conn(pair):
    try:
        conn, ssl_ca_file = pair
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

def ensure_table_ksei_daily(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS ksei_daily (
      base_symbol  VARCHAR(32)  NOT NULL,
      trade_date   DATE         NOT NULL,
      foreign_pct  DECIMAL(7,3)     NULL,
      retail_pct   DECIMAL(7,3)     NULL,
      total_volume BIGINT            NULL,
      total_value  BIGINT            NULL,
      created_at   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
      PRIMARY KEY (base_symbol, trade_date)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
    """
    cur = conn.cursor()
    cur.execute(ddl)
    try:
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ksei_trade_date ON ksei_daily (trade_date)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_ksei_base_symbol ON ksei_daily (base_symbol)")
    except mysql.connector.Error:
        try: cur.execute("CREATE INDEX idx_ksei_trade_date ON ksei_daily (trade_date)")
        except: pass
        try: cur.execute("CREATE INDEX idx_ksei_base_symbol ON ksei_daily (base_symbol)")
        except: pass
    cur.close()

# =========================================================
# Helpers – EOD
# =========================================================
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

def _norm(c:str)->str:
    c = c.strip().lower()
    c = c.replace(" ", "").replace("-", "").replace(".", "")
    c = c.replace("[","").replace("]","")
    return c

def _auto_map(df: pd.DataFrame) -> Dict[str,str]:
    res = {}
    cols_norm = [_norm(c) for c in df.columns]
    for canon in CANON:
        for raw, norm in zip(df.columns, cols_norm):
            if norm in ALIASES[canon]:
                res[canon] = raw; break
    return res

def _parse_date(s):
    if pd.isna(s): return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d","%d/%m/%Y","%m/%d/%Y","%d-%m-%Y","%m-%d-%Y"):
        try: return datetime.strptime(s, fmt).date()
        except ValueError: pass
    try: return pd.to_datetime(s, errors="coerce").date()
    except Exception: return None

def _to_int_series(s: pd.Series)->pd.Series:
    if getattr(s, "dtype", None) == object:
        s = s.astype(str).str.replace(r"[^0-9\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def load_eod_file(uploaded, override_source: str) -> Tuple[pd.DataFrame, Dict[str,str], str]:
    raw = uploaded.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    mapping = _auto_map(df)
    out = pd.DataFrame({
        "Tanggal": df[mapping["date"]].apply(_parse_date) if "date" in mapping else None,
        "Ticker":  df[mapping["ticker"]].astype(str).str.strip() if "ticker" in mapping else None,
        "Open":    pd.to_numeric(df[mapping.get("open")], errors="coerce") if "open" in mapping else None,
        "High":    pd.to_numeric(df[mapping.get("high")], errors="coerce") if "high" in mapping else None,
        "Low":     pd.to_numeric(df[mapping.get("low")],  errors="coerce") if "low"  in mapping else None,
        "Close":   pd.to_numeric(df[mapping.get("close")],errors="coerce") if "close" in mapping else None,
        "Volume":  _to_int_series(df[mapping.get("volume")]) if "volume" in mapping else None,
        "OI":      _to_int_series(df[mapping.get("oi")])     if "oi" in mapping else None,
    })
    src = override_source.strip() or os.path.basename(uploaded.name)
    out["SourceFile"] = src
    out = out.dropna(subset=["Tanggal","Ticker"])
    return out, mapping, src

# =========================================================
# Helpers – KSEI (ringkas)
# =========================================================
KSEI_ALIASES = {
    "base_symbol": {"base_symbol","symbol","ticker","kode","saham","code"},
    "trade_date": {"trade_date","date","tanggal","tgl"},
    "foreign_pct": {"foreign_pct","pa_pct","asing_pct","foreignpercent","asingpercent"},
    "retail_pct": {"retail_pct","ri_pct","retailpercent"},
    "total_volume": {"total_volume","volume","vol","vol_total"},
    "total_value": {"total_value","nilai","value","val_total"},
}

def _map_ksei(df):
    res={}
    cols_norm=[_norm(c) for c in df.columns]
    for k, names in KSEI_ALIASES.items():
        for raw,n in zip(df.columns, cols_norm):
            if n in names: res[k]=raw; break
    return res

def coerce_ksei_df_local(df: pd.DataFrame)->pd.DataFrame:
    mp=_map_ksei(df)
    out=pd.DataFrame()
    out["trade_date"]=pd.to_datetime(df[mp.get("trade_date", df.columns[0])], errors="coerce").dt.date
    out["base_symbol"]=df[mp.get("base_symbol", df.columns[1] if df.shape[1]>1 else df.columns[0])].astype(str).str.upper().str.strip().str.replace(r"\s+FF$","",regex=True)
    to_num=lambda s: pd.to_numeric(s, errors="coerce")
    out["foreign_pct"]=to_num(df[mp["foreign_pct"]]) if "foreign_pct" in mp else None
    out["retail_pct"]=to_num(df[mp["retail_pct"]]) if "retail_pct" in mp else None
    out["total_volume"]=to_num(df[mp["total_volume"]]) if "total_volume" in mp else None
    out["total_value"]=to_num(df[mp["total_value"]]) if "total_value" in mp else None
    out=out.dropna(subset=["trade_date","base_symbol"]).drop_duplicates(subset=["base_symbol","trade_date"])
    return out

# =========================================================
# UI
# =========================================================
tab1, tab2 = st.tabs(["📤 Upload EOD (CSV) — Multi-file", "📥 Import KSEI (CSV/TXT)"])

with tab1:
    st.subheader("Upload EOD RAW → `eod_prices_raw` (Batch)")
    st.caption("Pilih **banyak file** sekaligus. Auto-map kolom, preview gabungan, lalu **INSERT IGNORE** ke DB.")
    source_hint = st.text_input("Nama sumber/SourceFile (opsional, override nama file)", value="")
    files = st.file_uploader(
        "Pilih satu/lebih file CSV EOD",
        type=["csv"],
        key="eod_uploader_multi",
        accept_multiple_files=True,
    )

    if files:
        combined=[]; per_file=[]
        for f in files:
            df_tmp, mapping, src = load_eod_file(f, source_hint or f.name)
            per_file.append((f.name, len(df_tmp)))
            combined.append(df_tmp)

        if combined:
            all_df = pd.concat(combined, ignore_index=True)
            st.success(f"{len(files)} file dipilih • total baris valid: {len(all_df):,}")
            st.write("Preview gabungan (maks 50 baris):")
            st.dataframe(all_df.head(50), use_container_width=True)

            with st.expander("📊 Ringkasan per file"):
                st.dataframe(pd.DataFrame(per_file, columns=["File","Baris Valid"]), use_container_width=True)

            if st.button("🚀 INSERT IGNORE semua ke `eod_prices_raw`", type="primary"):
                conn, tmp = get_conn()
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
                            str(r.Ticker).strip(),
                            r.Tanggal,
                            float(r.Open) if pd.notna(r.Open) else None,
                            float(r.High) if pd.notna(r.High) else None,
                            float(r.Low) if pd.notna(r.Low) else None,
                            float(r.Close) if pd.notna(r.Close) else None,
                            int(r.Volume) if pd.notna(r.Volume) else None,
                            int(r.OI) if pd.notna(r.OI) else None,
                            str(r.SourceFile) if pd.notna(r.SourceFile) else None,
                        )
                        for r in all_df.itertuples(index=False)
                    ]
                    CHUNK=5000; total=0
                    for i in range(0, len(data), CHUNK):
                        cur.executemany(sql, data[i:i+CHUNK]); total += cur.rowcount
                    conn.commit(); cur.close()
                    st.success(f"Selesai. Row ditambahkan (non-duplikat diabaikan oleh PK): {total}.")
                finally:
                    close_conn((conn, tmp))
    else:
        st.info("Belum ada file dipilih. Kamu bisa drag-and-drop **banyak file** sekaligus.")

with tab2:
    st.subheader("Import KSEI → `ksei_daily`")
    st.caption("Terima CSV/TXT KSEI (Balancepos). Dinormalisasi ke kolom: base_symbol, trade_date, foreign_pct, retail_pct, total_volume, total_value.")
    up = st.file_uploader("Pilih file KSEI (CSV/TXT)", type=["csv","txt"], key="ksei_uploader")
    if up:
        if up.name.lower().endswith(".csv"):
            df_raw = pd.read_csv(up)
            df_ksei = coerce_ksei_df_local(df_raw)
        else:
            # parser TXT sederhana (gunakan CSV reader dengan deteksi delimiter)
            raw = up.read().decode("utf-8", errors="ignore")
            first = raw.splitlines()[0] if raw else ""
            counts = {",": first.count(","), ";": first.count(";"), "\\t": first.count("\\t"), "|": first.count("|")}
            delim = max(counts, key=counts.get)
            if delim == "\\t": delim = "\t"
            df_raw = pd.read_csv(io.StringIO(raw), sep=delim)
            df_ksei = coerce_ksei_df_local(df_raw)

        st.write("Preview (dinormalisasi):")
        st.dataframe(df_ksei.head(30), use_container_width=True)

        if st.button("🚀 Insert ke `ksei_daily` (INSERT IGNORE)"):
            conn, tmp = get_conn()
            try:
                ensure_table_ksei_daily(conn)
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
                CHUNK=5000; total=0
                for i in range(0, len(data), CHUNK):
                    cur.executemany(sql, data[i:i+CHUNK]); total += cur.rowcount
                conn.commit(); cur.close()
                st.success(f"Selesai. Row ksei_daily ditambahkan (non-duplikat): {total}.")
            finally:
                close_conn((conn, tmp))
