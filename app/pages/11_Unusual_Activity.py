# Unusual Activity Scanner (Rolling 20D ex-today)
from datetime import date, timedelta
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from db_utils import get_db_connection, get_db_name, check_secrets

# ── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="🔎 Unusual Activity", page_icon="🔎", layout="wide")
st.title("🔎 Unusual Activity (Rolling 20D ex-today)")

if not check_secrets(show_in_ui=True):
    st.stop()

st.caption(f"DB aktif: **{get_db_name()}**")

THEME = "#3AA6A0"
TABLE_H = 400
HEADER_H = 56

# ── Helpers ─────────────────────────────────────────────────────────────────
def _alive(conn):
    try:
        conn.reconnect(attempts=2, delay=1)
    except Exception:
        pass

def _close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected(): conn.close()
        else:
            conn.close()
    except Exception:
        pass

@st.cache_data(ttl=300)
def _date_bounds():
    con = get_db_connection(); _alive(con)
    try:
        q = "SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM data_harian"
        s = pd.read_sql(q, con)
        mn, mx = s.loc[0, "mn"], s.loc[0, "mx"]
        if pd.isna(mn) or pd.isna(mx):
            today = date.today()
            return today - timedelta(days=30), today
        return mn, mx
    finally:
        _close(con)

def fmt_id(x, dec=0):
    try:
        if x is None or pd.isna(x): return ""
        s = f"{float(x):,.{dec}f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(x)

def fmt_pct(x, dec=2):
    try:
        if x is None or pd.isna(x): return ""
        return fmt_id(float(x)*100, dec) + "%"
    except Exception:
        return str(x)

def df_fmt(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = df.copy()
    for c,(kind,dec) in spec.items():
        if c in df.columns:
            df[c] = df[c].map(lambda v: fmt_id(v,dec) if kind=="num" else fmt_pct(v,dec))
    return df

def to_csv_bytes(df): return df.to_csv(index=False).encode("utf-8")

def esc(s):
    return ("" if s is None else str(s)
            .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;"))

def H(c): return c.replace("_"," ").upper()

def render_table(df_fmt: pd.DataFrame, cols, tooltip_col="nama_perusahaan", height=TABLE_H):
    d = df_fmt[cols].reset_index(drop=True).copy()
    widths = {
        "kode_saham":"14%","nama_perusahaan":"28%","nilai":"18%","volume":"18%",
        "spread_bps":"12%","net_foreign_value":"18%","avg20":"18%","ratio":"12%",
        "zscore":"12%"
    }
    thead = "".join(f'<th style="width:{widths.get(c,"20%")}"><div class="th">{esc(H(c))}</div></th>' for c in cols)
    rows = []
    for i,row in d.iterrows():
        tds=[]
        for c in cols:
            v = "" if pd.isna(row[c]) else esc(row[c])
            if c=="kode_saham":
                tip = esc(df_fmt.loc[i, tooltip_col]) if tooltip_col in df_fmt.columns else ""
                tds.append(f'<td class="code" title="{tip}">{v}</td>')
            elif c in ("nilai","volume","net_foreign_value","avg20","spread_bps","ratio","zscore"):
                tds.append(f'<td class="num">{v}</td>')
            else:
                tds.append(f'<td class="txt">{v}</td>')
        rows.append("<tr>"+"".join(tds)+"</tr>")
    tbody = "\n".join(rows)

    html = f"""
    <style>
      .ua-box {{
        border:1px solid #e9ecef; border-radius:10px; box-shadow:0 1px 2px rgba(0,0,0,.05);
        background:#fff; width:100%; overflow:hidden; box-sizing:border-box;
      }}
      .ua-scroll {{
        height:{height}px; overflow:auto; scrollbar-gutter:stable both-edges;
      }}
      table.ua {{ border-collapse:collapse; width:100%; table-layout:fixed; min-width: 780px; }}
      table.ua th, table.ua td {{ padding:8px 10px; font-size:13px; }}
      thead th {{ position:sticky; top:0; z-index:1; background:{THEME}; color:#fff; font-weight:700;
                 height:{HEADER_H}px; text-align:center; vertical-align:middle; }}
      .th {{ display:flex; align-items:center; justify-content:center; white-space:normal;
             word-break:break-word; line-height:1.15; padding:2px 4px; }}
      td {{ white-space:nowrap; }}
      td.num {{ text-align:right; }}
      td.txt {{ text-align:left;  }}
      td.code{{ text-align:left; font-weight:600; }}
      tr:nth-child(even) {{ background:#f7fbfb; }}
      tr:hover {{ background:#e8f4f3; }}
    </style>
    <div class="ua-box"><div class="ua-scroll">
      <table class="ua">
        <colgroup>{''.join(f'<col style="width:{widths.get(c,"20%")}">' for c in cols)}</colgroup>
        <thead><tr>{thead}</tr></thead>
        <tbody>{tbody}</tbody>
      </table>
    </div></div>
    """
    st_html(html, height=height+HEADER_H, scrolling=False)

# ── Query helpers ───────────────────────────────────────────────────────────
def _sql_block(metric: str):
    """
    metric ∈ {'nilai','volume','abs_nf','spread'}
    Return tuple (select_exprs, where_avg, order)
    """
    if metric == "nilai":
        return ("d.nilai AS metric_now, s.avg20_nilai_excl AS avg20, s.std20_nilai_excl AS std20",
                "s.avg20_nilai_excl", "metric_now / NULLIF(s.avg20_nilai_excl,0) DESC")
    if metric == "volume":
        return ("d.volume AS metric_now, s.avg20_vol_excl AS avg20, s.std20_vol_excl AS std20",
                "s.avg20_vol_excl", "metric_now / NULLIF(s.avg20_vol_excl,0) DESC")
    if metric == "abs_nf":
        return ("ABS(d.net_foreign_value) AS metric_now, s.avg20_abs_nf_excl AS avg20, s.std20_abs_nf_excl AS std20",
                "s.avg20_abs_nf_excl", "metric_now / NULLIF(s.avg20_abs_nf_excl,0) DESC")
    if metric == "spread":
        return ("d.spread_bps AS metric_now, s.avg20_spread_excl AS avg20, s.std20_spread_excl AS std20",
                "s.avg20_spread_excl", "metric_now / NULLIF(s.avg20_spread_excl,0) DESC")
    raise ValueError("Unknown metric")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_unusual(trd: date, metric: str, min_avg20: float, min_ratio: float, min_z: float, topn: int):
    sel, where_avg, orderby = _sql_block(metric)
    con = get_db_connection(); _alive(con)
    try:
        sql = f"""
        SELECT
          d.trade_date,
          d.kode_saham,
          ANY_VALUE(d.nama_perusahaan) AS nama_perusahaan,
          {sel},
          d.nilai, d.volume, d.net_foreign_value, d.spread_bps,
          (metric_now / NULLIF(avg20,0))             AS ratio,
          ((metric_now - avg20)/NULLIF(std20,0))     AS zscore
        FROM v_daily_metrics d
        JOIN v_ua_stats_20d s
          ON s.trade_date = d.trade_date
         AND s.kode_saham = d.kode_saham
        WHERE d.trade_date = %s
          AND {where_avg} IS NOT NULL
          AND {where_avg} >= %s
        HAVING (ratio >= %s OR zscore >= %s)
        ORDER BY {orderby}
        LIMIT %s
        """
        df = pd.read_sql(sql, con, params=[trd, min_avg20, min_ratio, min_z, int(topn)])
        return df
    finally:
        _close(con)

# ── Controls ────────────────────────────────────────────────────────────────
MIN_D, MAX_D = _date_bounds()
left, right = st.columns([1,3])
with left:
    the_date = st.date_input("Tanggal analisa", value=MAX_D, min_value=MIN_D, max_value=MAX_D)
flt1, flt2, flt3, flt4 = st.columns(4)
with flt1:
    min_avg20 = st.number_input("Min ADVT/AVG 20D — Rp", min_value=0, step=100_000_000, value=1_000_000_000)
with flt2:
    min_ratio = st.number_input("Min RATIO (x)", min_value=0.0, step=0.1, value=2.0)
with flt3:
    min_z = st.number_input("Min Z-score", min_value=0.0, step=0.5, value=2.0)
with flt4:
    topn = st.number_input("Top N", min_value=5, max_value=200, step=5, value=30)

st.markdown("---")

# ── Tabs ────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🟩 Nilai (Value) Spike",
    "🟦 Volume Spike",
    "🟧 Net Foreign Spike",
    "🟥 Spread Spike",
])

# 1) NILAI
with tabs[0]:
    df = fetch_unusual(the_date, "nilai", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now":"nilai","avg20":"avg20"})
        df_fmt = df_fmt = df_fmt = df_fmt # just to avoid linter ;)
        df_fmt = df_fmt(df_show,
                        {"nilai":("num",0),"avg20":("num",0),"ratio":("num",2),
                         "zscore":("num",2),"volume":("num",0),"net_foreign_value":("num",0),"spread_bps":("num",2)})
        cols = ["kode_saham","nama_perusahaan","nilai","avg20","ratio","zscore","volume","net_foreign_value","spread_bps"]
        render_table(df_fmt, cols)
        st.download_button("⬇️ Export CSV - Nilai Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_value_{the_date}.csv")

# 2) VOLUME
with tabs[1]:
    df = fetch_unusual(the_date, "volume", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now":"volume","avg20":"avg20"})
        df_fmt2 = df_fmt(df_show,
                         {"volume":("num",0),"avg20":("num",0),"ratio":("num",2),
                          "zscore":("num",2),"nilai":("num",0),"net_foreign_value":("num",0),"spread_bps":("num",2)})
        cols = ["kode_saham","nama_perusahaan","volume","avg20","ratio","zscore","nilai","net_foreign_value","spread_bps"]
        render_table(df_fmt2, cols)
        st.download_button("⬇️ Export CSV - Volume Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_volume_{the_date}.csv")

# 3) NET FOREIGN (ABS)
with tabs[2]:
    df = fetch_unusual(the_date, "abs_nf", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now":"abs_net_foreign","avg20":"avg20"})
        df_show["abs_net_foreign"] = df_show["abs_net_foreign"].astype(float)
        df_fmt3 = df_fmt(df_show,
                         {"abs_net_foreign":("num",0),"avg20":("num",0),"ratio":("num",2),
                          "zscore":("num",2),"nilai":("num",0),"volume":("num",0),"spread_bps":("num",2)})
        df_fmt3 = df_fmt3.rename(columns={"abs_net_foreign":"net_foreign_value"})
        cols = ["kode_saham","nama_perusahaan","net_foreign_value","avg20","ratio","zscore","nilai","volume","spread_bps"]
        render_table(df_fmt3, cols)
        st.download_button("⬇️ Export CSV - Net Foreign Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_netforeign_{the_date}.csv")

# 4) SPREAD
with tabs[3]:
    df = fetch_unusual(the_date, "spread", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now":"spread_bps","avg20":"avg20"})
        df_fmt4 = df_fmt(df_show,
                         {"spread_bps":("num",2),"avg20":("num",2),"ratio":("num",2),
                          "zscore":("num",2),"nilai":("num",0),"volume":("num",0),"net_foreign_value":("num",0)})
        cols = ["kode_saham","nama_perusahaan","spread_bps","avg20","ratio","zscore","nilai","volume","net_foreign_value"]
        render_table(df_fmt4, cols)
        st.download_button("⬇️ Export CSV - Spread Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_spread_{the_date}.csv")
