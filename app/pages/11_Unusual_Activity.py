# app/pages/11_Unusual_Activity.py
# Unusual Activity Scanner (rolling Nd ex-today) with automatic fallback (works on MySQL 5.7/8.0)

from datetime import date, timedelta
from typing import Tuple

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from db_utils import get_db_connection, get_db_name, check_secrets

# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(page_title="🔎 Unusual Activity", page_icon="🔎", layout="wide")
st.title("🔎 Unusual Activity (Rolling Nd ex-today)")

if not check_secrets(show_in_ui=True):
    st.stop()

st.caption(f"DB aktif: **{get_db_name()}**")

# ─────────────────────────────────────────────
# CONSTANTS & HELPERS
# ─────────────────────────────────────────────
THEME_COLOR = "#3AA6A0"
TABLE_HEIGHT = 400
HEADER_HEIGHT = 56

def _alive(conn):
    try: conn.reconnect(attempts=2, delay=1)
    except Exception: pass

def _close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected(): conn.close()
        else:
            conn.close()
    except Exception: pass

@st.cache_data(ttl=300)
def _date_bounds() -> Tuple[date, date]:
    con = get_db_connection(); _alive(con)
    try:
        s = pd.read_sql("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM data_harian", con)
        mn, mx = s.loc[0, "mn"], s.loc[0, "mx"]
        if pd.isna(mn) or pd.isna(mx):
            today = date.today(); return today - timedelta(days=30), today
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

def format_df_id(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = df.copy()
    for c,(kind,dec) in spec.items():
        if c in df.columns:
            df[c] = df[c].map(lambda v: fmt_id(v,dec) if kind=="num" else fmt_pct(v,dec))
    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def esc(s):
    return ("" if s is None else str(s)
            .replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;"))

def H(c:str)->str: return c.replace("_"," ").upper()

def render_table(df_fmt: pd.DataFrame, cols, tooltip_col="nama_perusahaan", height=TABLE_HEIGHT):
    """Header wrap + center, scroll stabil. Dedup kolom agar aman dari duplikasi nama."""
    df_fmt = df_fmt.loc[:, ~df_fmt.columns.duplicated(keep="first")]
    d = df_fmt[cols].reset_index(drop=True).copy()

    widths = {"kode_saham":"14%","nama_perusahaan":"26%","nilai":"16%","volume":"16%",
              "net_foreign_value":"16%","spread_bps":"12%","ratio":"10%","zscore":"10%"}
    # avgN akan tidak ada di dict -> fallback ke 20%
    thead = "".join(
        f'<th style="width:{widths.get(c,"20%")}"><div class="th-wrap">{esc(H(c))}</div></th>'
        for c in cols
    )

    trs = []
    for i,row in d.iterrows():
        tds=[]
        for c in cols:
            val = row[c]
            v = "" if pd.isna(val) else esc(val)
            if c == "kode_saham":
                tip = esc(df_fmt.loc[i, tooltip_col]) if tooltip_col in df_fmt.columns else ""
                tds.append(f'<td class="code" title="{tip}">{v}</td>')
            elif c in ("nilai","volume","net_foreign_value","spread_bps","ratio","zscore") or c.startswith("avg"):
                tds.append(f'<td class="num">{v}</td>')
            else:
                tds.append(f'<td class="txt">{v}</td>')
        trs.append("<tr>"+"".join(tds)+"</tr>")
    tbody = "\n".join(trs)

    html = f"""
    <style>
      .ua-box {{
        border:1px solid #e9ecef; border-radius:10px; box-shadow:0 1px 2px rgba(0,0,0,.05);
        background:#fff; width:100%; overflow:hidden; box-sizing:border-box;
      }}
      .ua-scroll {{ height:{height}px; overflow:auto; scrollbar-gutter:stable both-edges; }}
      table.ua {{ border-collapse:collapse; width:100%; table-layout:fixed; min-width:820px; }}
      table.ua th, table.ua td {{ padding:8px 10px; font-size:13px; }}
      thead th {{ position:sticky; top:0; z-index:1; background:{THEME_COLOR}; color:#fff; font-weight:700;
                 height:{HEADER_HEIGHT}px; text-align:center; vertical-align:middle; }}
      .th-wrap {{ display:flex; align-items:center; justify-content:center; white-space:normal;
                 word-break:break-word; text-align:center; line-height:1.15; padding:2px 4px; }}
      td {{ white-space:nowrap; }} td.num{{text-align:right;}} td.txt{{text-align:left;}} td.code{{font-weight:600;}}
      tr:nth-child(even){{background:#f7fbfb;}} tr:hover{{background:#e8f4f3;}}
      .chips {{ display:flex; gap:.5rem; flex-wrap:wrap; }}
      .chip {{ background:#e8f4f3; padding:.2rem .5rem; border-radius:999px; font-size:.85rem; }}
    </style>
    <div class="ua-box"><div class="ua-scroll">
      <table class="ua">
        <colgroup>{''.join(f'<col style="width:{widths.get(c,"20%")}">' for c in cols)}</colgroup>
        <thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody>
      </table>
    </div></div>
    """
    st_html(html, height=height+HEADER_HEIGHT, scrolling=False)

# ─────────────────────────────────────────────
# SQL BUILDERS (fallback without VIEW/window)
# ─────────────────────────────────────────────
def _metric_plain(metric: str, alias_prefix: str = "d") -> str:
    if metric == "nilai":
        return f"{alias_prefix}.nilai"
    if metric == "volume":
        return f"{alias_prefix}.volume"
    if metric == "abs_nf":
        return f"ABS({alias_prefix}.net_foreign_value)" if alias_prefix=="dvm" else f"ABS({alias_prefix}.foreign_buy - {alias_prefix}.foreign_sell)"
    if metric == "spread":
        return (f"CASE WHEN {alias_prefix}.bid IS NOT NULL AND {alias_prefix}.offer IS NOT NULL "
                f"AND ({alias_prefix}.bid+{alias_prefix}.offer)<>0 "
                f"THEN ({alias_prefix}.offer-{alias_prefix}.bid)/(({alias_prefix}.offer+{alias_prefix}.bid)/2)*10000 END")
    raise ValueError("unknown metric")

@st.cache_data(ttl=300, show_spinner=False)
def fetch_unusual(trd: date, metric: str, win_n: int, min_avg: float, min_ratio: float, min_z: float, topn: int) -> pd.DataFrame:
    """
    1) Coba via VIEW (v_daily_metrics & v_ua_stats_20d) kalau win_n == 20.
    2) Selain itu atau jika gagal → fallback subquery LIMIT win_n (kompatibel 5.7).
    """
    con = get_db_connection(); _alive(con)
    avg_alias = f"avg{win_n}"
    std_alias = f"std{win_n}"

    # 1) VIEW path (cepat, MySQL 8.0) — hanya untuk win_n == 20
    if win_n == 20:
        try:
            met_plain_view = _metric_plain(metric, alias_prefix="dvm")
            met_key = {"nilai":"nilai","volume":"vol","abs_nf":"abs_nf","spread":"spread"}[metric]
            sql_view = f"""
            SELECT
              dvm.trade_date, dvm.kode_saham,
              ANY_VALUE(dvm.nama_perusahaan) AS nama_perusahaan,
              {met_plain_view} AS metric_now,
              dvm.nilai, dvm.volume, dvm.net_foreign_value, dvm.spread_bps,
              s.avg20_{met_key}_excl AS {avg_alias},
              s.std20_{met_key}_excl AS {std_alias},
              ({met_plain_view}/NULLIF(s.avg20_{met_key}_excl,0))             AS ratio,
              (({met_plain_view}-s.avg20_{met_key}_excl)/NULLIF(s.std20_{met_key}_excl,0)) AS zscore
            FROM v_daily_metrics dvm
            JOIN v_ua_stats_20d s ON s.trade_date=dvm.trade_date AND s.kode_saham=dvm.kode_saham
            WHERE dvm.trade_date=%s AND s.avg20_{met_key}_excl IS NOT NULL AND s.avg20_{met_key}_excl >= %s
            HAVING (ratio >= %s OR zscore >= %s)
            ORDER BY ratio DESC
            LIMIT %s
            """
            df = pd.read_sql(sql_view, con, params=[trd, min_avg, min_ratio, min_z, int(topn)])
            _close(con); return df
        except Exception:
            pass

    # 2) Fallback: subquery avg/std Nd (kompatibel 5.7)
    try:
        met_plain = _metric_plain(metric, alias_prefix="d")
        inner = _metric_plain(metric, alias_prefix="m")
        avgstd = f"""
          (SELECT AVG(vv) FROM (
             SELECT {inner} AS vv FROM data_harian m
             WHERE m.kode_saham=d.kode_saham AND m.trade_date<d.trade_date
             ORDER BY m.trade_date DESC LIMIT {int(win_n)}
          ) t) AS {avg_alias},
          (SELECT STDDEV_POP(vv) FROM (
             SELECT {inner} AS vv FROM data_harian m
             WHERE m.kode_saham=d.kode_saham AND m.trade_date<d.trade_date
             ORDER BY m.trade_date DESC LIMIT {int(win_n)}
          ) t2) AS {std_alias}
        """
        sql_fb = f"""
        SELECT
          d.trade_date, d.kode_saham, d.nama_perusahaan,
          {met_plain} AS metric_now,
          d.nilai, d.volume,
          (d.foreign_buy-d.foreign_sell) AS net_foreign_value,
          CASE WHEN d.bid IS NOT NULL AND d.offer IS NOT NULL AND (d.bid+d.offer)<>0
               THEN (d.offer-d.bid)/((d.offer+d.bid)/2)*10000 END AS spread_bps,
          {avgstd},
          ({met_plain}/NULLIF({avg_alias},0))         AS ratio,
          (({met_plain}-{avg_alias})/NULLIF({std_alias},0)) AS zscore
        FROM data_harian d
        WHERE d.trade_date=%s
        HAVING {avg_alias} IS NOT NULL AND {avg_alias} >= %s AND (ratio >= %s OR zscore >= %s)
        ORDER BY ratio DESC
        LIMIT %s
        """
        df = pd.read_sql(sql_fb, con, params=[trd, min_avg, min_ratio, min_z, int(topn)])
        _close(con); return df
    except Exception as e:
        _close(con); st.error(f"Query UA gagal: {type(e).__name__}: {e}")
        return pd.DataFrame()

# ─────────────────────────────────────────────
# SUMMARY / HINTS
# ─────────────────────────────────────────────
def render_summary(df: pd.DataFrame, metric: str):
    if df.empty: return
    n = len(df)
    max_ratio = float(df["ratio"].astype(float).max())
    max_z = float(df["zscore"].astype(float).max())
    if metric == "nilai":
        total = float(df["nilai"].astype(float).sum()); label="Total Nilai (Rp)"; total_str=fmt_id(total,0)
    elif metric == "volume":
        total = float(df["volume"].astype(float).sum()); label="Total Volume"; total_str=fmt_id(total,0)
    elif metric == "abs_nf":
        total = float(df["net_foreign_value"].abs().astype(float).sum()); label="Total |Net Foreign| (Rp)"; total_str=fmt_id(total,0)
    else:
        total = float(df["spread_bps"].astype(float).mean()); label="Rata-rata Spread (bps)"; total_str=fmt_id(total,2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Emiten", f"{n}")
    c2.metric(label, total_str)
    c3.metric("Max Ratio (x)", fmt_id(max_ratio, 2))
    c4.metric("Max Z-score", fmt_id(max_z, 2))

def render_empty_hint(metric: str, win_n: int):
    bullets = [
        "Threshold **Min Ratio** dan/atau **Min Z-score** terlalu tinggi.",
        f"Filter **Min AVG {win_n}D** terlalu besar sehingga emiten illiquid tersaring.",
        "Data untuk tanggal ini belum lengkap/tersedia di tabel `data_harian`.",
        f"Saham belum punya **{win_n} hari historis** sebelum tanggal ini → avg & std jadi `NULL`.",
    ]
    if metric == "spread":
        bullets.append("Kolom **bid/offer** kosong untuk banyak saham pada tanggal tsb.")
    st.info("Tidak ada spike sesuai filter.")
    st.write("**Kemungkinan penyebab:**")
    st.write("\n".join(f"- {b}" for b in bullets))
    st.write("**Saran cepat:** turunkan `Min Ratio`/`Min Z-score` atau kecilkan `Min AVG`.")

with st.expander("📘 Cara membaca & catatan", expanded=False):
    st.markdown("""
**Definisi singkat**
- **RATIO** = nilai metrik hari ini ÷ **AVG Nd** (rata-rata N hari *sebelum* hari ini).
- **Z-SCORE** = (nilai metrik hari ini − **AVG Nd**) ÷ **STD Nd**.
- **Rolling window** bisa dipilih: **5, 7, 10, 20, 30** hari (ex-today).
- Input **Min AVG** menyaring saham dengan rata-rata Nd kecil (illiquid).

**Tips**
- Mulai dengan `Min Ratio = 2` dan `Min Z-score = 2`, lalu sesuaikan.
- Jika tab kosong, kecilkan threshold, ganti window (mis. 10 → 7), atau pilih tanggal lain.
    """)

# ─────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────
MIN_D, MAX_D = _date_bounds()
lh1, lh2 = st.columns([1,1])
with lh1:
    the_date = st.date_input("Tanggal analisa", value=MAX_D, min_value=MIN_D, max_value=MAX_D)
with lh2:
    win_n = st.selectbox("Rolling window (hari)", options=[30,20,10,7,5], index=1)  # default 20

avg_label_text = f"Min AVG {win_n}D — Rp"  # label dinamis

c1,c2,c3,c4 = st.columns(4)
with c1: min_avg = st.number_input(avg_label_text, min_value=0, step=100_000_000, value=1_000_000_000)
with c2: min_ratio = st.number_input("Min Ratio (x)", min_value=0.0, step=0.1, value=2.0)
with c3: min_z = st.number_input("Min Z-score", min_value=0.0, step=0.5, value=2.0)
with c4: topn = st.number_input("Top N", min_value=5, max_value=200, step=5, value=30)

st.markdown("---")

tabs = st.tabs(["🟩 Nilai (Value) Spike", "🟦 Volume Spike", "🟧 Net Foreign Spike", "🟥 Spread Spike"])

# helper untuk nama kolom AVG dinamis
def avg_col_name(win_n: int) -> str:
    return f"avg{win_n}"

# ─────────────────────────────────────────────
# TAB 1: VALUE
# ─────────────────────────────────────────────
with tabs[0]:
    df_raw = fetch_unusual(the_date, "nilai", win_n, min_avg, min_ratio, min_z, topn)
    if df_raw.empty:
        render_empty_hint("nilai", win_n)
    else:
        avg_col = avg_col_name(win_n)
        df_show = df_raw.drop(columns=["nilai"], errors="ignore").rename(columns={"metric_now":"nilai"})
        render_summary(df_show, "nilai")
        fmt = {
            "nilai":("num",0), avg_col:("num",0), "ratio":("num",2), "zscore":("num",2),
            "volume":("num",0),"net_foreign_value":("num",0),"spread_bps":("num",2)
        }
        df_fmt = format_df_id(df_show, fmt)
        cols = ["kode_saham","nama_perusahaan","nilai",avg_col,"ratio","zscore",
                "volume","net_foreign_value","spread_bps"]
        render_table(df_fmt, cols)
        st.download_button("⬇️ Export CSV - Nilai Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_value_{the_date}_n{win_n}.csv")

# ─────────────────────────────────────────────
# TAB 2: VOLUME
# ─────────────────────────────────────────────
with tabs[1]:
    df_raw = fetch_unusual(the_date, "volume", win_n, min_avg, min_ratio, min_z, topn)
    if df_raw.empty:
        render_empty_hint("volume", win_n)
    else:
        avg_col = avg_col_name(win_n)
        df_show = df_raw.drop(columns=["volume"], errors="ignore").rename(columns={"metric_now":"volume"})
        render_summary(df_show, "volume")
        fmt = {
            "volume":("num",0), avg_col:("num",0), "ratio":("num",2), "zscore":("num",2),
            "nilai":("num",0),"net_foreign_value":("num",0),"spread_bps":("num",2)
        }
        df_fmt = format_df_id(df_show, fmt)
        cols = ["kode_saham","nama_perusahaan","volume",avg_col,"ratio","zscore",
                "nilai","net_foreign_value","spread_bps"]
        render_table(df_fmt, cols)
        st.download_button("⬇️ Export CSV - Volume Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_volume_{the_date}_n{win_n}.csv")

# ─────────────────────────────────────────────
# TAB 3: NET FOREIGN (ABS)
# ─────────────────────────────────────────────
with tabs[2]:
    df_raw = fetch_unusual(the_date, "abs_nf", win_n, min_avg, min_ratio, min_z, topn)
    if df_raw.empty:
        render_empty_hint("abs_nf", win_n)
    else:
        avg_col = avg_col_name(win_n)
        df_show = df_raw.drop(columns=["net_foreign_value"], errors="ignore").rename(columns={"metric_now":"net_foreign_value"})
        render_summary(df_show, "abs_nf")
        fmt = {
            "net_foreign_value":("num",0), avg_col:("num",0), "ratio":("num",2), "zscore":("num",2),
            "nilai":("num",0),"volume":("num",0),"spread_bps":("num",2)
        }
        df_fmt = format_df_id(df_show, fmt)
        cols = ["kode_saham","nama_perusahaan","net_foreign_value",avg_col,"ratio","zscore",
                "nilai","volume","spread_bps"]
        render_table(df_fmt, cols)
        st.download_button("⬇️ Export CSV - Net Foreign Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_netforeign_{the_date}_n{win_n}.csv")

# ─────────────────────────────────────────────
# TAB 4: SPREAD
# ─────────────────────────────────────────────
with tabs[3]:
    df_raw = fetch_unusual(the_date, "spread", win_n, min_avg, min_ratio, min_z, topn)
    if df_raw.empty:
        render_empty_hint("spread", win_n)
    else:
        avg_col = avg_col_name(win_n)
        df_show = df_raw.drop(columns=["spread_bps"], errors="ignore").rename(columns={"metric_now":"spread_bps"})
        render_summary(df_show, "spread")
        fmt = {
            "spread_bps":("num",2), avg_col:("num",2), "ratio":("num",2), "zscore":("num",2),
            "nilai":("num",0),"volume":("num",0),"net_foreign_value":("num",0)
        }
        df_fmt = format_df_id(df_show, fmt)
        cols = ["kode_saham","nama_perusahaan","spread_bps",avg_col,"ratio","zscore",
                "nilai","volume","net_foreign_value"]
        render_table(df_fmt, cols)
        st.download_button("⬇️ Export CSV - Spread Spike", data=to_csv_bytes(df_show),
                           file_name=f"ua_spread_{the_date}_n{win_n}.csv")

st.markdown("---")
st.caption(
    "Engine otomatis: jika VIEW/window function tersedia akan dipakai (untuk N=20), "
    "selain itu fallback ke subquery rolling N-day (kompatibel MySQL 5.7+). "
    "Pastikan index (trade_date, kode_saham) dan (kode_saham, trade_date) dibuat agar cepat."
)
