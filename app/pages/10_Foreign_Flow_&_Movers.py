# app/pages/10_Foreign_Flow_&_Movers.py
# Foreign Flow & Top Movers Dashboard (FAST MODE + Quick Range + Symmetric UI)

from datetime import date, timedelta
from typing import Tuple, Optional
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from db_utils import get_db_connection, get_db_name, check_secrets

# ─────────────────────────── UI SETUP ───────────────────────────
st.set_page_config(page_title="📊 Foreign Flow & Movers", page_icon="📊", layout="wide")
st.title("📊 Foreign Flow & Top Movers")

with st.expander("ℹ️ Cara pakai & definisi", expanded=False):
    st.markdown(
        """
**Foreign Flow:** `net_foreign = foreign_buy - foreign_sell` (akumulasi di rentang tanggal yang dipilih).

**Mode cepat (default)** → pakai **rata-rata nilai transaksi pada periode terpilih** (proxy ADVT).  
**Mode akurat** → pakai **ADVT20** (rata-rata nilai 20 hari).

**Spread (bps)** = `(offer - bid) / ((offer + bid)/2) * 10,000`.
        """
    )

if not check_secrets(show_in_ui=True):
    st.stop()

st.caption(f"DB aktif: **{get_db_name()}**")

# ───────────────────── Helper koneksi aman ─────────────────────
def _ensure_alive(conn):
    try: conn.reconnect(attempts=2, delay=1)
    except Exception: pass

def _safe_close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected(): conn.close()
        else:
            conn.close()
    except Exception: pass

# ──────────────────────── DB PREP (views) ───────────────────────
DDL_V_DAILY_FAST = """
CREATE OR REPLACE VIEW v_daily_fast AS
SELECT
  trade_date,
  kode_saham,
  nama_perusahaan,
  nilai,
  (foreign_buy - foreign_sell) AS net_foreign_value,
  CASE WHEN bid IS NOT NULL AND offer IS NOT NULL AND (bid+offer) <> 0
       THEN (offer - bid)/((offer + bid)/2) * 10000 END AS spread_bps,
  sebelumnya, penutupan, volume, bid, offer, frekuensi
FROM data_harian;
"""

DDL_V_DAILY_METRICS = """
CREATE OR REPLACE VIEW v_daily_metrics AS
SELECT
  trade_date,
  kode_saham,
  nama_perusahaan,
  sebelumnya       AS prev_close,
  penutupan        AS close_price,
  CASE WHEN sebelumnya IS NOT NULL AND sebelumnya <> 0
       THEN (penutupan/sebelumnya) - 1 END                 AS ret_1d,
  volume,
  nilai,
  frekuensi,
  foreign_buy,
  foreign_sell,
  (foreign_buy - foreign_sell)                              AS net_foreign_value,
  CASE WHEN nilai IS NOT NULL AND nilai <> 0
       THEN (foreign_buy - foreign_sell)/nilai END          AS net_foreign_ratio,
  bid,
  offer,
  CASE WHEN bid IS NOT NULL AND offer IS NOT NULL AND (bid+offer) <> 0
       THEN (offer - bid)/((offer + bid)/2) * 10000 END     AS spread_bps
FROM data_harian;
"""

DDL_V_ROLLING_20D = """
CREATE OR REPLACE VIEW v_rolling_20d AS
SELECT
  d.*,
  AVG(d.nilai)   OVER (PARTITION BY d.kode_saham ORDER BY d.trade_date
                       ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) AS advt_20
FROM v_daily_metrics d;
"""

def ensure_views(conn):
    cur = conn.cursor()
    try:
        cur.execute(DDL_V_DAILY_FAST)
        cur.execute(DDL_V_DAILY_METRICS)
        cur.execute(DDL_V_ROLLING_20D)
        conn.commit()
    except Exception as e:
        st.warning("Gagal membuat VIEW (mungkin hak akses terbatas). Mode cepat tetap jalan.")
        st.caption(f"Detail: {e}")
    finally:
        try: cur.close()
        except Exception: pass

def get_date_bounds(conn) -> Tuple[date, date]:
    q = "SELECT MIN(trade_date), MAX(trade_date) FROM data_harian"
    cur = conn.cursor()
    cur.execute(q)
    row = cur.fetchone()
    cur.close()
    if not row or not row[0] or not row[1]:
        today = date.today()
        return today - timedelta(days=30), today
    return row[0], row[1]

# ───────────────────────── Helpers tampilan ─────────────────────────
THEME_COLOR     = "#3AA6A0"
TABLE_HEIGHT    = 430     # tinggi konten tabel (tanpa header)
HEADER_HEIGHT   = 56      # header fixed height → wrap + center tetap simetris

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
        return fmt_id(float(x) * 100, dec) + "%"
    except Exception:
        return str(x)

def df_format_id(df: pd.DataFrame, formats: dict) -> pd.DataFrame:
    df = df.copy()
    for col, (kind, dec) in formats.items():
        if col in df.columns:
            if kind == "num": df[col] = df[col].map(lambda v: fmt_id(v, dec))
            elif kind == "pct": df[col] = df[col].map(lambda v: fmt_pct(v, dec))
    return df

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def html_escape(s):
    return ("" if s is None else str(s)
            .replace("&","&amp;").replace("<","&lt;")
            .replace(">","&gt;").replace('"',"&quot;"))

def humanize_header(c: str) -> str:
    return c.replace("_", " ").upper()

def render_table_html(df_fmt: pd.DataFrame, cols_to_show, tooltip_col="nama_perusahaan", height=TABLE_HEIGHT):
    """
    Tabel HTML simetris:
      - header wrap + center (uppercase, underscore→spasi), tinggi fixed
      - scrollbar horizontal dihilangkan; gunakan gutter stabil agar lebar tidak berubah
      - tooltip perusahaan saat hover kolom kode_saham
      - kolom fixed width → kiri/kanan identik
    """
    d = df_fmt.copy()
    d = d[cols_to_show].reset_index(drop=True)

    # Width persis 100% → 18/22/22/28/10 untuk ranking
    widths = {
        "kode_saham": "18%",
        "cum_net_foreign": "22%",
        "total_value": "22%",
        "avg_liq": "28%",
        "avg_spread_bps": "10%",
        # movers (5 kolom)
        "ret_1d": "18%", "nilai": "22%", "volume": "22%", "spread_bps": "12%",
    }

    ths = "".join(
        f'<th style="width:{widths.get(c,"20%")}"><div class="th-wrap">{html_escape(humanize_header(c))}</div></th>'
        for c in cols_to_show
    )

    trs = []
    for i, row in d.iterrows():
        tds = []
        for c in cols_to_show:
            val = "" if pd.isna(row[c]) else html_escape(row[c])
            if c == "kode_saham":
                tip = ""
                if tooltip_col in df_fmt.columns and i < len(df_fmt):
                    tip = html_escape(df_fmt.loc[i, tooltip_col])
                tds.append(f'<td class="code" title="{tip}">{val}</td>')
            else:
                cls = "num" if c in ("cum_net_foreign","total_value","avg_liq","avg_spread_bps","nilai","volume","spread_bps") else "txt"
                tds.append(f'<td class="{cls}">{val}</td>')
        trs.append("<tr>" + "".join(tds) + "</tr>")
    body = "\n".join(trs)

    html = f"""
    <style>
      .tbl-outer {{
        border:1px solid #e9ecef; border-radius:10px;
        box-shadow:0 1px 2px rgba(0,0,0,0.05);
        background:#fff; width:100%; overflow:hidden;
        box-sizing:border-box;
      }}
      .tbl-scroll {{
        height:{height}px;
        overflow-y:auto; overflow-x:hidden;
        /* Kunci space scrollbar supaya kiri/kanan tidak berubah lebarnya */
        scrollbar-gutter: stable both-edges;
      }}
      table.tbl {{ border-collapse:collapse; width:100%; table-layout:fixed; }}
      table.tbl th, table.tbl td {{ padding:8px 10px; font-size:13px; }}
      table.tbl thead th {{
        position:sticky; top:0; z-index:1;
        background:{THEME_COLOR}; color:#fff; font-weight:700;
        text-align:center; height:{HEADER_HEIGHT}px; vertical-align:middle;
      }}
      /* Header wrap di SPASI (tidak pecah huruf) + center */
      table.tbl th .th-wrap {{
        display:flex; align-items:center; justify-content:center;
        white-space:normal;           /* bisa multi-line */
        word-break:keep-all;          /* jangan pecah huruf */
        overflow-wrap:anywhere;       /* pecah di spasi/titik */
        text-align:center; line-height:1.15; padding:2px 4px;
      }}
      table.tbl td {{ white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
      table.tbl td.num {{ text-align:right; }}
      table.tbl td.txt {{ text-align:left;  }}
      table.tbl td.code{{ text-align:left; font-weight:600; }}
      table.tbl tr:nth-child(even) {{ background:#f7fbfb; }}
      table.tbl tr:hover {{ background:#e8f4f3; }}
    </style>
    <div class="tbl-outer">
      <div class="tbl-scroll">
        <table class="tbl">
          <colgroup>
            {''.join(f'<col style="width:{widths.get(c,"20%")}">' for c in cols_to_show)}
          </colgroup>
          <thead><tr>{ths}</tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
    </div>
    """
    st_html(html, height=height + HEADER_HEIGHT, scrolling=False)

# ───────────────────────── Data fetchers ─────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def fetch_net_foreign_rank_fast(start: date, end: date, min_avg_value: Optional[float], max_spread: Optional[float], topn: int) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        _ensure_alive(conn); ensure_views(conn)
        sql = """
        SELECT
          kode_saham,
          ANY_VALUE(nama_perusahaan)                AS nama_perusahaan,
          SUM(net_foreign_value)                    AS cum_net_foreign,
          SUM(nilai)                                AS total_value,
          AVG(nilai)                                AS avg_value_period,
          AVG(spread_bps)                           AS avg_spread_bps
        FROM v_daily_fast
        WHERE trade_date BETWEEN %s AND %s
        GROUP BY kode_saham
        HAVING 1=1
        """
        params = [start, end]
        if min_avg_value and min_avg_value > 0:
            sql += " AND AVG(nilai) >= %s"; params.append(min_avg_value)
        if max_spread and max_spread > 0:
            sql += " AND (AVG(spread_bps) <= %s OR AVG(spread_bps) IS NULL)"; params.append(max_spread)
        sql += " ORDER BY cum_net_foreign DESC LIMIT %s"; params.append(int(topn))
        return pd.read_sql(sql, conn, params=params)
    finally:
        _safe_close(conn)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_net_foreign_rank_accurate(start: date, end: date, min_advt: Optional[float], max_spread: Optional[float], topn: int) -> pd.DataFrame:
    conn = get_db_connection()
    try:
        _ensure_alive(conn); ensure_views(conn)
        sql = """
        SELECT
          d.kode_saham,
          ANY_VALUE(d.nama_perusahaan)                AS nama_perusahaan,
          SUM(d.net_foreign_value)                    AS cum_net_foreign,
          SUM(d.nilai)                                AS total_value,
          AVG(r.advt_20)                              AS avg_advt_20,
          AVG(d.spread_bps)                           AS avg_spread_bps
        FROM v_daily_metrics d
        LEFT JOIN v_rolling_20d r
               ON r.trade_date = d.trade_date AND r.kode_saham = d.kode_saham
        WHERE d.trade_date BETWEEN %s AND %s
        GROUP BY d.kode_saham
        HAVING 1=1
        """
        params = [start, end]
        if min_advt and min_advt > 0:
            sql += " AND AVG(r.advt_20) >= %s"; params.append(min_advt)
        if max_spread and max_spread > 0:
            sql += " AND (AVG(d.spread_bps) <= %s OR AVG(d.spread_bps) IS NULL)"; params.append(max_spread)
        sql += " ORDER BY cum_net_foreign DESC LIMIT %s"; params.append(int(topn))
        return pd.read_sql(sql, conn, params=params)
    finally:
        _safe_close(conn)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_movers_fast(trade_dt: date, min_value_today: Optional[float], max_spread: Optional[float], topn: int):
    conn = get_db_connection()
    try:
        _ensure_alive(conn); ensure_views(conn)
        sql = """
        SELECT
          kode_saham, nama_perusahaan,
          CASE WHEN sebelumnya IS NOT NULL AND sebelumnya <> 0
               THEN (penutupan/sebelumnya) - 1 END AS ret_1d,
          nilai, volume, spread_bps
        FROM v_daily_fast
        WHERE trade_date = %s
        """
        params = [trade_dt]
        if min_value_today and min_value_today > 0:
            sql += " AND nilai >= %s"; params.append(min_value_today)
        if max_spread and max_spread > 0:
            sql += " AND (spread_bps <= %s OR spread_bps IS NULL)"; params.append(max_spread)
        df_all = pd.read_sql(sql, conn, params=params)
        df_gainer = df_all.sort_values(by="ret_1d", ascending=False).head(int(topn)).reset_index(drop=True)
        df_loser  = df_all.sort_values(by="ret_1d", ascending=True ).head(int(topn)).reset_index(drop=True)
        return df_gainer, df_loser
    finally:
        _safe_close(conn)

# ───────────────────────────── Range & Filters ─────────────────────────────
conn_for_bounds = get_db_connection()
_ensure_alive(conn_for_bounds); ensure_views(conn_for_bounds)
min_d, max_d = get_date_bounds(conn_for_bounds)
_safe_close(conn_for_bounds)

quick = st.radio(
    "Rentang cepat",
    ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "Semua", "Custom"],
    horizontal=True,
)

def quick_to_range(q: str) -> Tuple[date, date]:
    end = max_d
    if q == "1 Minggu":  return (max_d - timedelta(days=6), end)
    if q == "1 Bulan":   return (max_d - timedelta(days=30), end)
    if q == "3 Bulan":   return (max_d - timedelta(days=91), end)
    if q == "6 Bulan":   return (max_d - timedelta(days=182), end)
    if q == "1 Tahun":   return (max_d - timedelta(days=365), end)
    if q == "Semua":     return (min_d, end)
    return None, None

if quick != "Custom":
    start_date, end_date = quick_to_range(quick)
else:
    colA, colB = st.columns([1,1])
    with colA:
        start_date = st.date_input("Tanggal Mulai", value=max(min_d, max_d - timedelta(days=29)), min_value=min_d, max_value=max_d, key="start_custom")
    with colB:
        end_date   = st.date_input("Tanggal Akhir", value=max_d, min_value=min_d, max_value=max_d, key="end_custom")

if start_date > end_date:
    st.error("Tanggal Mulai tidak boleh lebih besar dari Tanggal Akhir.")
    st.stop()

top_bar = st.columns([1,1,1,1])
with top_bar[0]:
    fast_mode = st.toggle("⚡ Mode cepat", value=True, help="Avg nilai periode sebagai proxy ADVT.")
with top_bar[1]:
    if fast_mode:
        min_liq = st.number_input("Min **Avg Nilai (periode)** — Rp", min_value=0, step=100_000_000, value=1_000_000_000)
    else:
        min_liq = st.number_input("Min **ADVT20** — Rp", min_value=0, step=100_000_000, value=1_000_000_000)
with top_bar[2]:
    max_spread_bps = st.number_input("Max Spread (bps)", min_value=0, step=5, value=50)
with top_bar[3]:
    topn = st.number_input("Top N", min_value=5, max_value=100, step=5, value=20)

st.markdown("---")

# ───────────────────────────── Foreign Flow ─────────────────────────────
st.subheader("🟩 Top Net **Buy Asing** (Akumulasi)")

if fast_mode:
    df_rank = fetch_net_foreign_rank_fast(start_date, end_date, min_liq, max_spread_bps, topn*2)
    df_rank = df_rank.rename(columns={"avg_value_period":"avg_liq"})
else:
    df_rank = fetch_net_foreign_rank_accurate(start_date, end_date, min_liq, max_spread_bps, topn*2)
    df_rank = df_rank.rename(columns={"avg_advt_20":"avg_liq"})

df_buy  = df_rank.sort_values("cum_net_foreign", ascending=False).head(topn).reset_index(drop=True)
df_sell = df_rank.sort_values("cum_net_foreign", ascending=True ).head(topn).reset_index(drop=True)

fmt_cols_rank = {
    "cum_net_foreign": ("num", 0),
    "total_value":     ("num", 0),
    "avg_liq":         ("num", 0),
    "avg_spread_bps":  ("num", 2),
}
df_buy_fmt  = df_format_id(df_buy,  fmt_cols_rank)
df_sell_fmt = df_format_id(df_sell, fmt_cols_rank)

cols_rank = ["kode_saham","cum_net_foreign","total_value","avg_liq","avg_spread_bps"]

c1, c2 = st.columns(2, gap="large")
with c1:
    render_table_html(df_buy_fmt, cols_rank)
    st.download_button("⬇️ Export CSV - Top Net Buy",
                       data=to_csv_bytes(df_buy),
                       file_name=f"net_buy_{start_date}_{end_date}.csv")

with c2:
    st.subheader("🟥 Top Net **Sell Asing** (Akumulasi)")
    render_table_html(df_sell_fmt, cols_rank)
    st.download_button("⬇️ Export CSV - Top Net Sell",
                       data=to_csv_bytes(df_sell),
                       file_name=f"net_sell_{start_date}_{end_date}.csv")

st.markdown("---")

# ───────────────────────────── Movers 1D ─────────────────────────────
st.subheader("⚡ Top Movers (1D)")
mcol1, _ = st.columns([1,3])
with mcol1:
    movers_date = st.date_input("Tanggal untuk Movers (1D)", value=end_date, min_value=min_d, max_value=max_d, key="movers_date")

min_value_today = st.number_input("Min Nilai (hari itu) — Rp", min_value=0, step=100_000_000, value=500_000_000)

gainers, losers = fetch_top_movers_fast(movers_date, min_value_today, max_spread_bps, topn)

fmt_cols_mov = {
    "ret_1d": ("pct", 2),
    "nilai":  ("num", 0),
    "volume": ("num", 0),
    "spread_bps": ("num", 2),
}
gainers_fmt = df_format_id(gainers, fmt_cols_mov)
losers_fmt  = df_format_id(losers,  fmt_cols_mov)

cols_movers = ["kode_saham","ret_1d","nilai","volume","spread_bps"]

mc1, mc2 = st.columns(2, gap="large")
with mc1:
    st.markdown(f"**Top Gainer — {movers_date}**")
    render_table_html(gainers_fmt, cols_movers)
    st.download_button("⬇️ Export CSV - Top Gainer",
                       data=to_csv_bytes(gainers),
                       file_name=f"top_gainer_{movers_date}.csv")

with mc2:
    st.markdown(f"**Top Loser — {movers_date}**")
    render_table_html(losers_fmt, cols_movers)
    st.download_button("⬇️ Export CSV - Top Loser",
                       data=to_csv_bytes(losers),
                       file_name=f"top_loser_{movers_date}.csv")

st.markdown("---")
st.caption("Header kembali wrap & center. Scrollbar vertical stabil (scrollbar-gutter), sehingga tabel kiri/kanan benar-benar simetris.")
