
import io
import os
import tempfile
from datetime import datetime
from typing import Dict, Optional, List, Tuple

import pandas as pd
import streamlit as st
import mysql.connector

st.set_page_config(page_title="Upload EOD & Import KSEI", page_icon="📤", layout="wide")

st.title("📤 Upload EOD (CSV) & 📥 Import KSEI")
st.caption("Satu halaman untuk input EOD raw ke `eod_prices_raw` (mendukung **multi-file/batch**) dan import KSEI.")

# -------------------- DB UTILS --------------------
def get_conn():
    """Selalu kembalikan koneksi FRESH (no cache)."""
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

def _to_int_series(s):
    # remove thousand separators / non-digits; keep leading minus
    if getattr(s, "dtype", None) == object:
        s = s.astype(str).str.replace(r"[^0-9\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")

def load_eod_file(uploaded, fallback_source: str) -> Tuple[pd.DataFrame, Dict[str,str], str]:
    """Parse single CSV EOD into canonical schema; returns (df, mapping, src_name)."""
    raw = uploaded.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    mapping = auto_map_columns(df)
    # canonical df
    tmp = pd.DataFrame({
        "Tanggal": df[mapping["date"]].apply(parse_date_any) if "date" in mapping else None,
        "Ticker": df[mapping["ticker"]].astype(str).str.strip() if "ticker" in mapping else None,
        "Open": pd.to_numeric(df[mapping.get("open")], errors="coerce") if "open" in mapping else None,
        "High": pd.to_numeric(df[mapping.get("high")], errors="coerce") if "high" in mapping else None,
        "Low":  pd.to_numeric(df[mapping.get("low")],  errors="coerce") if "low"  in mapping else None,
        "Close":pd.to_numeric(df[mapping.get("close")],errors="coerce") if "close" in mapping else None,
        "Volume": _to_int_series(df[mapping.get("volume")]) if "volume" in mapping else None,
        "OI":     _to_int_series(df[mapping.get("oi")])     if "oi" in mapping else None,
    })
    src_name = fallback_source.strip() or os.path.basename(uploaded.name)
    tmp["SourceFile"] = src_name
    before = len(tmp)
    tmp = tmp.dropna(subset=["Tanggal","Ticker"])
    return tmp, mapping, src_name

# -------------------- HELPERS (KSEI) --------------------
KSEI_ALIASES = {
    "base_symbol": {"base_symbol","symbol","ticker","kode","saham","code"},
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
    if "trade_date" in mapping:
        out["trade_date"] = pd.to_datetime(df[mapping["trade_date"]], errors="coerce").dt.date
    else:
        out["trade_date"] = pd.to_datetime(df.iloc[:,0], errors="coerce").dt.date
    if "base_symbol" in mapping:
        out["base_symbol"] = df[mapping["base_symbol"]].astype(str).str.strip().str.upper()
    else:
        out["base_symbol"] = df.iloc[:,1].astype(str).str.strip().str.upper() if df.shape[1] > 1 else ""

    to_num = lambda s: pd.to_numeric(s, errors="coerce")
    out["foreign_pct"]  = to_num(df[mapping["foreign_pct"]])  if "foreign_pct"  in mapping else None
    out["retail_pct"]   = to_num(df[mapping["retail_pct"]])   if "retail_pct"   in mapping else None
    out["total_volume"] = to_num(df[mapping["total_volume"]]) if "total_volume" in mapping else None
    out["total_value"]  = to_num(df[mapping["total_value"]])  if "total_value"  in mapping else None

    out = out.dropna(subset=["trade_date","base_symbol"])
    out["base_symbol"] = out["base_symbol"].str.replace(r"\s+FF$", "", regex=True).str.strip()
    out = out.drop_duplicates(subset=["base_symbol","trade_date"])
    return out

def parse_balancepos_txt_local(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="ignore")
    first = text.splitlines()[0] if text else ""
    delims = {",": first.count(","), ";": first.count(";"), "\\t": first.count("\\t"), "|": first.count("|")}
    delim = max(delims, key=delims.get) if delims else "|"
    if delim == "\\t":
        delim = "\t"
    df = pd.read_csv(io.StringIO(text), sep=delim)
    df.columns = [c.strip() for c in df.columns]

    # detect totals
    if "Total" in df.columns and "Total.1" in df.columns:
        local_total_col, foreign_total_col = "Total", "Total.1"
    else:
        local_total_col = "Total" if "Total" in df.columns else None
        foreign_total_col = None
        for i, c in enumerate(df.columns):
            if isinstance(c, str) and c.startswith("Foreign "):
                for c2 in df.columns[i:]:
                    if isinstance(c2, str) and c2.strip().lower() == "total":
                        foreign_total_col = c2; break
                break

    out = pd.DataFrame()
    out["trade_date"] = pd.to_datetime(df["Date"] if "Date" in df.columns else df.iloc[:,0], errors="coerce", format="%d-%b-%Y").dt.date
    code_col = "Code" if "Code" in df.columns else ("base_symbol" if "base_symbol" in df.columns else df.columns[1])
    out["base_symbol"] = df[code_col].astype(str).str.strip().str.upper().str.replace(r"\s+FF$", "", regex=True)

    price = pd.to_numeric(df["Price"], errors="coerce") if "Price" in df.columns else None
    local_total = pd.to_numeric(df.get(local_total_col, None), errors="coerce") if local_total_col else None
    foreign_total = pd.to_numeric(df.get(foreign_total_col, None), errors="coerce") if foreign_total_col else None
    if local_total is None:
        local_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Local ")]
        local_total = pd.to_numeric(df[local_cols], errors="coerce").fillna(0).sum(axis=1) if local_cols else None
    if foreign_total is None:
        foreign_cols = [c for c in df.columns if isinstance(c, str) and c.startswith("Foreign ")]
        foreign_total = pd.to_numeric(df[foreign_cols], errors="coerce").fillna(0).sum(axis=1) if foreign_cols else None

    total_volume = None
    if local_total is not None and foreign_total is not None:
        total_volume = local_total.fillna(0) + foreign_total.fillna(0)

    out["foreign_pct"] = None
    out["retail_pct"] = None
    out["total_volume"] = None
    out["total_value"] = None
    if total_volume is not None:
        out["total_volume"] = total_volume
        with pd.option_context('mode.use_inf_as_na', True):
            out["foreign_pct"] = (foreign_total / total_volume * 100).round(2)
            out["retail_pct"]  = (100 - out["foreign_pct"]).round(2)
        if price is not None:
            out["total_value"] = (total_volume * price).astype("Int64")

    out = out.dropna(subset=["trade_date","base_symbol"]).drop_duplicates(subset=["base_symbol","trade_date"])
    for col in ["foreign_pct","retail_pct","total_volume","total_value"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out

# -------------------- UI (Tabs) --------------------
tab1, tab2 = st.tabs(["📤 Upload EOD (CSV) — Multi-file", "📥 Import KSEI (CSV/TXT)"])

with tab1:
    st.subheader("Upload EOD RAW → `eod_prices_raw` (Batch)")
    st.caption("Pilih **banyak file sekaligus**. Setiap file akan di-parse otomatis dan digabung sebelum insert.")
    source_hint = st.text_input("Nama sumber/SourceFile (opsional, override nama file)", value="")
    files = st.file_uploader("Pilih satu/lebih file CSV EOD", type=["csv"], key="eod_uploader_multi", accept_multiple_files=True)

    if files:
        # parse semua file
        combined = []
        maps = []
        per_file_stats = []
        for f in files:
            df_tmp, mapping, src_name = load_eod_file(f, source_hint or f.name)
            maps.append((f.name, mapping))
            before = len(df_tmp)
            df_tmp = df_tmp.dropna(subset=["Tanggal","Ticker"])
            after = len(df_tmp)
            per_file_stats.append((f.name, before, after))
            combined.append(df_tmp)

        if combined:
            all_df = pd.concat(combined, ignore_index=True)
            # tampilkan preview gabungan
            st.write("**Preview gabungan (maks 50 baris):**")
            st.dataframe(all_df.head(50), use_container_width=True)

            # ringkasan mapping per file
            with st.expander("🔎 Pemetaan kolom per file"):
                for name, mp in maps:
                    st.write(f"**{name}** → {mp}")

            # ringkasan baris per file
            with st.expander("📊 Ringkasan validitas per file"):
                summ = pd.DataFrame(per_file_stats, columns=["File","Baris (awal)","Baris valid (Tanggal & Ticker)"])
                st.dataframe(summ, use_container_width=True)

            # siap insert
            if st.button("🚀 INSERT IGNORE semua ke `eod_prices_raw`", type="primary"):
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
                    # chunked executemany untuk menghindari paket terlalu besar
                    CHUNK = 5000
                    total = 0
                    for i in range(0, len(data), CHUNK):
                        cur.executemany(sql, data[i:i+CHUNK])
                        total += cur.rowcount
                    conn.commit(); cur.close()
                    st.success(f"Selesai. Row ditambahkan (non-duplikat/diabaikan oleh PK): {total}.")
                finally:
                    close_conn((conn, ssl_ca_file))
        else:
            st.warning("Tidak ada data valid dari file yang diunggah.")

with tab2:
    st.subheader("Import KSEI → `ksei_daily`")
    st.caption("Menerima CSV/TXT hasil Balancepos. Otomatis normalisasi kolom (base_symbol, trade_date, foreign_pct, retail_pct, total_volume, total_value).")
    up = st.file_uploader("Pilih file KSEI (CSV/TXT)", type=["csv","txt"], key="ksei_uploader")

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
                CHUNK = 5000
                total = 0
                for i in range(0, len(data), CHUNK):
                    cur.executemany(sql, data[i:i+CHUNK])
                    total += cur.rowcount
                conn.commit(); cur.close()
                st.success(f"Selesai. Row ksei_daily ditambahkan (non-duplikat): {total}.")
            finally:
                close_conn((conn, ssl_ca_file))
