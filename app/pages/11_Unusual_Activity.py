# app/pages/11_Unusual_Activity.py
# Unusual Activity Scanner (rolling 20D ex-today) with automatic fallback (no VIEW required)

from datetime import date, timedelta
from typing import Tuple, Optional

import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html

from db_utils import get_db_connection, get_db_name, check_secrets


# ─────────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────────
st.set_page_config(page_title="🔎 Unusual Activity", page_icon="🔎", layout="wide")
st.title("🔎 Unusual Activity (Rolling 20D ex-today)")

if not check_secrets(show_in_ui=True):
    st.stop()

st.caption(f"DB aktif: **{get_db_name()}**")


# ─────────────────────────────────────────────
# CONSTANTS & SMALL HELPERS
# ─────────────────────────────────────────────
THEME_COLOR = "#3AA6A0"
TABLE_HEIGHT = 400
HEADER_HEIGHT = 56


def _alive(conn):
    """Best-effort reconnect (for pooled mysql connector)."""
    try:
        conn.reconnect(attempts=2, delay=1)
    except Exception:
        pass


def _close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected():
                conn.close()
        else:
            conn.close()
    except Exception:
        pass


@st.cache_data(ttl=300)
def _date_bounds() -> Tuple[date, date]:
    con = get_db_connection()
    _alive(con)
    try:
        s = pd.read_sql("SELECT MIN(trade_date) mn, MAX(trade_date) mx FROM data_harian", con)
        mn, mx = s.loc[0, "mn"], s.loc[0, "mx"]
        if pd.isna(mn) or pd.isna(mx):
            today = date.today()
            return today - timedelta(days=30), today
        return mn, mx
    finally:
        _close(con)


def fmt_id(x, dec=0):
    """Indonesian number format: dot thousands, comma decimals."""
    try:
        if x is None or pd.isna(x):
            return ""
        s = f"{float(x):,.{dec}f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return str(x)


def fmt_pct(x, dec=2):
    try:
        if x is None or pd.isna(x):
            return ""
        return fmt_id(float(x) * 100, dec) + "%"
    except Exception:
        return str(x)


def format_df_id(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Apply Indonesian formatting according to spec = {col: ('num'|'pct', decimals)}."""
    df = df.copy()
    for c, (kind, dec) in spec.items():
        if c in df.columns:
            if kind == "num":
                df[c] = df[c].map(lambda v: fmt_id(v, dec))
            elif kind == "pct":
                df[c] = df[c].map(lambda v: fmt_pct(v, dec))
    return df


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def esc(s):
    return (
        ""
        if s is None
        else str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def H(c: str) -> str:
    """Humanize header: underscore -> space, uppercase."""
    return c.replace("_", " ").upper()


def render_table(df_fmt: pd.DataFrame, cols, tooltip_col="nama_perusahaan", height=TABLE_HEIGHT):
    """Responsive, consistent table: header wrap + center, stable scroll, horizontal scroll allowed."""
    d = df_fmt[cols].reset_index(drop=True).copy()

    # Widths add to ~100% to avoid jitter (can be tuned if needed)
    widths = {
        "kode_saham": "14%",
        "nama_perusahaan": "26%",
        "nilai": "16%",
        "volume": "16%",
        "net_foreign_value": "16%",
        "spread_bps": "12%",
        "avg20": "16%",
        "ratio": "10%",
        "zscore": "10%",
    }

    thead = "".join(
        f'<th style="width:{widths.get(c, "20%")}"><div class="th-wrap">{esc(H(c))}</div></th>'
        for c in cols
    )

    rows = []
    for i, row in d.iterrows():
        tds = []
        for c in cols:
            v = "" if pd.isna(row[c]) else esc(row[c])
            if c == "kode_saham":
                tip = esc(df_fmt.loc[i, tooltip_col]) if tooltip_col in df_fmt.columns else ""
                tds.append(f'<td class="code" title="{tip}">{v}</td>')
            elif c in (
                "nilai",
                "volume",
                "net_foreign_value",
                "spread_bps",
                "avg20",
                "ratio",
                "zscore",
            ):
                tds.append(f'<td class="num">{v}</td>')
            else:
                tds.append(f'<td class="txt">{v}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
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
      table.ua {{ border-collapse:collapse; width:100%; table-layout:fixed; min-width: 820px; }}
      table.ua th, table.ua td {{ padding:8px 10px; font-size:13px; }}
      thead th {{
        position:sticky; top:0; z-index:1; background:{THEME_COLOR}; color:#fff; font-weight:700;
        height:{HEADER_HEIGHT}px; text-align:center; vertical-align:middle;
      }}
      .th-wrap {{
        display:flex; align-items:center; justify-content:center;
        white-space:normal; word-break:break-word; text-align:center; line-height:1.15; padding:2px 4px;
      }}
      td {{ white-space:nowrap; }}
      td.num {{ text-align:right; }}
      td.txt {{ text-align:left;  }}
      td.code{{ text-align:left; font-weight:600; }}
      tr:nth-child(even) {{ background:#f7fbfb; }}
      tr:hover {{ background:#e8f4f3; }}
    </style>
    <div class="ua-box"><div class="ua-scroll">
      <table class="ua">
        <colgroup>
          {''.join(f'<col style="width:{widths.get(c, "20%")}">' for c in cols)}
        </colgroup>
        <thead><tr>{thead}</tr></thead>
        <tbody>{tbody}</tbody>
      </table>
    </div></div>
    """
    st_html(html, height=height + HEADER_HEIGHT, scrolling=False)


# ─────────────────────────────────────────────
# SQL BUILDERS (fallback without VIEW/window)
# ─────────────────────────────────────────────
def _metric_now_expr(alias: str, metric: str) -> str:
    """Return expression for today's metric value."""
    if metric == "nilai":
        base = "d.nilai"
    elif metric == "volume":
        base = "d.volume"
    elif metric == "abs_nf":
        base = "ABS(d.foreign_buy - d.foreign_sell)"
    elif metric == "spread":
        base = (
            "CASE WHEN d.bid IS NOT NULL AND d.offer IS NOT NULL AND (d.bid+d.offer)<>0 "
            "THEN (d.offer-d.bid)/((d.offer+d.bid)/2)*10000 END"
        )
    else:
        raise ValueError("unknown metric")
    return f"{base} AS {alias}"


def _avgstd_20d_sub(metric: str) -> str:
    """Return two subqueries (avg20, std20) using last 20 rows BEFORE today (ex-today)."""
    if metric == "nilai":
        inner = "m.nilai"
    elif metric == "volume":
        inner = "m.volume"
    elif metric == "abs_nf":
        inner = "ABS(m.foreign_buy - m.foreign_sell)"
    elif metric == "spread":
        inner = (
            "CASE WHEN m.bid IS NOT NULL AND m.offer IS NOT NULL AND (m.bid+m.offer)<>0 "
            "THEN (m.offer-m.bid)/((m.offer+m.bid)/2)*10000 END"
        )
    else:
        raise ValueError("unknown metric")

    return f"""
      (SELECT AVG(vv) FROM (
         SELECT {inner} AS vv
         FROM data_harian m
         WHERE m.kode_saham=d.kode_saham AND m.trade_date < d.trade_date
         ORDER BY m.trade_date DESC LIMIT 20
      ) t) AS avg20,
      (SELECT STDDEV_POP(vv) FROM (
         SELECT {inner} AS vv
         FROM data_harian m
         WHERE m.kode_saham=d.kode_saham AND m.trade_date < d.trade_date
         ORDER BY m.trade_date DESC LIMIT 20
      ) t2) AS std20
    """


@st.cache_data(ttl=300, show_spinner=False)
def fetch_unusual(trd: date, metric: str, min_avg20: float, min_ratio: float, min_z: float, topn: int) -> pd.DataFrame:
    """
    Try faster VIEW path (if v_ua_stats_20d & v_daily_metrics exist, MySQL 8.0+),
    else fallback to portable subquery method (works on MySQL 5.7).
    """
    con = get_db_connection()
    _alive(con)

    # 1) Attempt VIEW path
    try:
        sel_now = {
            "nilai": "d.nilai AS metric_now",
            "volume": "d.volume AS metric_now",
            "abs_nf": "ABS(d.net_foreign_value) AS metric_now",
            "spread": "d.spread_bps AS metric_now",
        }[metric]

        met_key = {"nilai": "nilai", "volume": "vol", "abs_nf": "abs_nf", "spread": "spread"}[metric]

        sql_view = f"""
        SELECT
          d.trade_date, d.kode_saham,
          ANY_VALUE(d.nama_perusahaan) AS nama_perusahaan,
          {sel_now},
          d.nilai, d.volume, d.net_foreign_value, d.spread_bps,
          s.avg20_{met_key}_excl AS avg20,
          s.std20_{met_key}_excl AS std20
        FROM v_daily_metrics d
        JOIN v_ua_stats_20d s
          ON s.trade_date=d.trade_date AND s.kode_saham=d.kode_saham
        WHERE d.trade_date=%s AND s.avg20_{met_key}_excl IS NOT NULL AND s.avg20_{met_key}_excl >= %s
        HAVING (metric_now/NULLIF(avg20,0)) >= %s OR ((metric_now-avg20)/NULLIF(std20,0)) >= %s
        ORDER BY (metric_now/NULLIF(avg20,0)) DESC
        LIMIT %s
        """
        df = pd.read_sql(sql_view, con, params=[trd, min_avg20, min_ratio, min_z, int(topn)])
        _close(con)
        return df
    except Exception:
        # Fall through to portable version
        pass

    # 2) Fallback portable path (no VIEW, no window functions)
    try:
        metric_now = _metric_now_expr("metric_now", metric)
        avgstd = _avgstd_20d_sub(metric)
        sql_fb = f"""
        SELECT
          d.trade_date, d.kode_saham, d.nama_perusahaan,
          {metric_now},
          d.nilai, d.volume,
          (d.foreign_buy - d.foreign_sell) AS net_foreign_value,
          CASE WHEN d.bid IS NOT NULL AND d.offer IS NOT NULL AND (d.bid+d.offer)<>0
               THEN (d.offer-d.bid)/((d.offer+d.bid)/2)*10000 END AS spread_bps,
          {avgstd}
        FROM data_harian d
        WHERE d.trade_date=%s
        HAVING avg20 IS NOT NULL AND avg20 >= %s
           AND ( (metric_now/NULLIF(avg20,0)) >= %s OR ((metric_now-avg20)/NULLIF(std20,0)) >= %s )
        ORDER BY (metric_now/NULLIF(avg20,0)) DESC
        LIMIT %s
        """
        df = pd.read_sql(sql_fb, con, params=[trd, min_avg20, min_ratio, min_z, int(topn)])
        _close(con)
        return df
    except Exception as e:
        _close(con)
        # show real error to help debugging on UI
        st.error(f"Query UA gagal: {type(e).__name__}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────
MIN_D, MAX_D = _date_bounds()
left, right = st.columns([1, 3])
with left:
    the_date = st.date_input("Tanggal analisa", value=MAX_D, min_value=MIN_D, max_value=MAX_D)

flt1, flt2, flt3, flt4 = st.columns(4)
with flt1:
    min_avg20 = st.number_input("Min ADVT/AVG 20D — Rp", min_value=0, step=100_000_000, value=1_000_000_000)
with flt2:
    min_ratio = st.number_input("Min Ratio (x)", min_value=0.0, step=0.1, value=2.0)
with flt3:
    min_z = st.number_input("Min Z-score", min_value=0.0, step=0.5, value=2.0)
with flt4:
    topn = st.number_input("Top N", min_value=5, max_value=200, step=5, value=30)

st.markdown("---")

tabs = st.tabs(
    ["🟩 Nilai (Value) Spike", "🟦 Volume Spike", "🟧 Net Foreign Spike", "🟥 Spread Spike"]
)

# ─────────────────────────────────────────────
# TAB 1: VALUE
# ─────────────────────────────────────────────
with tabs[0]:
    df = fetch_unusual(the_date, "nilai", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now": "nilai"})
        df_fmt = format_df_id(
            df_show,
            {
                "nilai": ("num", 0),
                "avg20": ("num", 0),
                "ratio": ("num", 2),
                "zscore": ("num", 2),
                "volume": ("num", 0),
                "net_foreign_value": ("num", 0),
                "spread_bps": ("num", 2),
            },
        )
        cols = [
            "kode_saham",
            "nama_perusahaan",
            "nilai",
            "avg20",
            "ratio",
            "zscore",
            "volume",
            "net_foreign_value",
            "spread_bps",
        ]
        render_table(df_fmt, cols)
        st.download_button(
            "⬇️ Export CSV - Nilai Spike",
            data=to_csv_bytes(df_show),
            file_name=f"ua_value_{the_date}.csv",
        )

# ─────────────────────────────────────────────
# TAB 2: VOLUME
# ─────────────────────────────────────────────
with tabs[1]:
    df = fetch_unusual(the_date, "volume", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now": "volume"})
        df_fmt = format_df_id(
            df_show,
            {
                "volume": ("num", 0),
                "avg20": ("num", 0),
                "ratio": ("num", 2),
                "zscore": ("num", 2),
                "nilai": ("num", 0),
                "net_foreign_value": ("num", 0),
                "spread_bps": ("num", 2),
            },
        )
        cols = [
            "kode_saham",
            "nama_perusahaan",
            "volume",
            "avg20",
            "ratio",
            "zscore",
            "nilai",
            "net_foreign_value",
            "spread_bps",
        ]
        render_table(df_fmt, cols)
        st.download_button(
            "⬇️ Export CSV - Volume Spike",
            data=to_csv_bytes(df_show),
            file_name=f"ua_volume_{the_date}.csv",
        )

# ─────────────────────────────────────────────
# TAB 3: NET FOREIGN (ABS)
# ─────────────────────────────────────────────
with tabs[2]:
    df = fetch_unusual(the_date, "abs_nf", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now": "net_foreign_value"})
        # metric pada query adalah ABS(net_foreign); tetap beri nama yang jelas.
        df_fmt = format_df_id(
            df_show,
            {
                "net_foreign_value": ("num", 0),
                "avg20": ("num", 0),
                "ratio": ("num", 2),
                "zscore": ("num", 2),
                "nilai": ("num", 0),
                "volume": ("num", 0),
                "spread_bps": ("num", 2),
            },
        )
        cols = [
            "kode_saham",
            "nama_perusahaan",
            "net_foreign_value",
            "avg20",
            "ratio",
            "zscore",
            "nilai",
            "volume",
            "spread_bps",
        ]
        render_table(df_fmt, cols)
        st.download_button(
            "⬇️ Export CSV - Net Foreign Spike",
            data=to_csv_bytes(df_show),
            file_name=f"ua_netforeign_{the_date}.csv",
        )

# ─────────────────────────────────────────────
# TAB 4: SPREAD
# ─────────────────────────────────────────────
with tabs[3]:
    df = fetch_unusual(the_date, "spread", min_avg20, min_ratio, min_z, topn)
    if df.empty:
        st.info("Tidak ada spike sesuai filter.")
    else:
        df_show = df.rename(columns={"metric_now": "spread_bps"})
        df_fmt = format_df_id(
            df_show,
            {
                "spread_bps": ("num", 2),
                "avg20": ("num", 2),
                "ratio": ("num", 2),
                "zscore": ("num", 2),
                "nilai": ("num", 0),
                "volume": ("num", 0),
                "net_foreign_value": ("num", 0),
            },
        )
        cols = [
            "kode_saham",
            "nama_perusahaan",
            "spread_bps",
            "avg20",
            "ratio",
            "zscore",
            "nilai",
            "volume",
            "net_foreign_value",
        ]
        render_table(df_fmt, cols)
        st.download_button(
            "⬇️ Export CSV - Spread Spike",
            data=to_csv_bytes(df_show),
            file_name=f"ua_spread_{the_date}.csv",
        )

st.markdown("---")
st.caption(
    "Fallback otomatis: jika VIEW / window function tersedia akan dipakai; "
    "kalau tidak, halaman memakai subquery 20D sehingga kompatibel di MySQL 5.7+. "
    "Pastikan index (trade_date, kode_saham) & (kode_saham, trade_date) sudah dibuat untuk performa."
)
