# app/pages/10_Foreign_Flow_&_Movers.py

from datetime import date, timedelta
from typing import Tuple, Optional
import pandas as pd
import streamlit as st
from streamlit.components.v1 import html as st_html
from db_utils import get_db_connection, get_db_name, check_secrets

st.set_page_config(page_title="📊 Foreign Flow & Movers", page_icon="📊", layout="wide")
st.title("📊 Foreign Flow & Top Movers")

with st.expander("ℹ️ Cara pakai & definisi", expanded=False):
    st.markdown("""
**Foreign Flow:** `net_foreign = foreign_buy - foreign_sell`.

**Mode cepat** → rata2 nilai pada periode (proxy ADVT).  
**Mode akurat** → ADVT20 (window 20 hari).

**Spread (bps)** = `(offer - bid) / ((offer + bid)/2) * 10,000`.
""")

if not check_secrets(show_in_ui=True):
    st.stop()
st.caption(f"DB aktif: **{get_db_name()}**")

def _ensure_alive(conn):
    try: conn.reconnect(attempts=2, delay=1)
    except Exception: pass

def _safe_close(conn):
    try:
        if hasattr(conn, "is_connected"):
            if conn.is_connected(): conn.close()
        else: conn.close()
    except Exception: pass

DDL_V_DAILY_FAST = """
CREATE OR REPLACE VIEW v_daily_fast AS
SELECT trade_date, kode_saham, nama_perusahaan, nilai,
       (foreign_buy-foreign_sell) AS net_foreign_value,
       CASE WHEN bid IS NOT NULL AND offer IS NOT NULL AND (bid+offer)<>0
            THEN (offer-bid)/((offer+bid)/2)*10000 END AS spread_bps,
       sebelumnya, penutupan, volume, bid, offer, frekuensi
FROM data_harian;
"""
DDL_V_DAILY_METRICS = """
CREATE OR REPLACE VIEW v_daily_metrics AS
SELECT trade_date, kode_saham, nama_perusahaan,
       sebelumnya AS prev_close, penutupan AS close_price,
       CASE WHEN sebelumnya IS NOT NULL AND sebelumnya<>0
            THEN (penutupan/sebelumnya)-1 END AS ret_1d,
       volume, nilai, frekuensi, foreign_buy, foreign_sell,
       (foreign_buy-foreign_sell) AS net_foreign_value,
       CASE WHEN nilai IS NOT NULL AND nilai<>0
            THEN (foreign_buy-foreign_sell)/nilai END AS net_foreign_ratio,
       bid, offer,
       CASE WHEN bid IS NOT NULL AND offer IS NOT NULL AND (bid+offer)<>0
            THEN (offer-bid)/((offer+bid)/2)*10000 END AS spread_bps
FROM data_harian;
"""
DDL_V_ROLLING_20D = """
CREATE OR REPLACE VIEW v_rolling_20d AS
SELECT d.*, AVG(d.nilai) OVER (PARTITION BY d.kode_saham
                               ORDER BY d.trade_date
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
    except Exception:
        pass
    finally:
        try: cur.close()
        except Exception: pass

def get_date_bounds(conn) -> Tuple[date, date]:
    cur = conn.cursor(); cur.execute("SELECT MIN(trade_date), MAX(trade_date) FROM data_harian")
    mn,mx = cur.fetchone(); cur.close()
    if not mn or not mx:
        today = date.today(); return today-timedelta(days=30), today
    return mn, mx

THEME_COLOR   = "#3AA6A0"
TABLE_H       = 430
HEADER_H      = 56

def fmt_id(x, dec=0):
    try:
        if x is None or pd.isna(x): return ""
        s = f"{float(x):,.{dec}f}"
        return s.replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception: return str(x)

def fmt_pct(x, dec=2):
    try:
        if x is None or pd.isna(x): return ""
        return fmt_id(float(x)*100, dec) + "%"
    except Exception: return str(x)

def df_format_id(df, formats):
    df = df.copy()
    for c,(kind,dec) in formats.items():
        if c in df.columns:
            df[c] = df[c].map(lambda v: fmt_id(v,dec) if kind=="num" else fmt_pct(v,dec))
    return df

def to_csv_bytes(df): return df.to_csv(index=False).encode("utf-8")
def esc(s): 
    return ("" if s is None else str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;"))
def H(c): return c.replace("_"," ").upper()

def render_table_html(df_fmt: pd.DataFrame, cols, tooltip_col="nama_perusahaan", height=TABLE_H):
    d = df_fmt[cols].reset_index(drop=True).copy()

    widths = {
        "kode_saham":"18%","cum_net_foreign":"22%","total_value":"22%","avg_liq":"28%","avg_spread_bps":"10%",
        "ret_1d":"18%","nilai":"22%","volume":"22%","spread_bps":"12%",
    }

    thead = "".join(f'<th style="width:{widths.get(c,"20%")}"><div class="th">{esc(H(c))}</div></th>' for c in cols)

    rows = []
    for i,row in d.iterrows():
        tds=[]
        for c in cols:
            v = "" if pd.isna(row[c]) else esc(row[c])
            if c=="kode_saham":
                tip = esc(df_fmt.loc[i,tooltip_col]) if tooltip_col in df_fmt.columns else ""
                tds.append(f'<td class="code" title="{tip}">{v}</td>')
            else:
                tds.append(f'<td class="num">{v}</td>' if c in ("cum_net_foreign","total_value","avg_liq","avg_spread_bps","nilai","volume","spread_bps") else f'<td class="txt">{v}</td>')
        rows.append("<tr>"+"".join(tds)+"</tr>")
    tbody = "\n".join(rows)

    html = f"""
    <style>
      .box {{
        border:1px solid #e9ecef; border-radius:10px; box-shadow:0 1px 2px rgba(0,0,0,.05);
        background:#fff; width:100%; overflow:hidden; box-sizing:border-box;
      }}
      .scroll {{
        height:{height}px; overflow-y:auto; overflow-x:auto;   /* <— IZINKAN SCROLL X, SUPAYA NILAI TIDAK TERPOTONG */
        scrollbar-gutter: stable both-edges;
      }}
      table.tb {{ border-collapse:collapse; width:100%; table-layout:fixed; min-width: 720px; }}
      table.tb th, table.tb td {{ padding:8px 10px; font-size:13px; }}
      thead th {{
        position:sticky; top:0; z-index:1; background:{THEME_COLOR}; color:#fff; font-weight:700;
        height:{HEADER_H}px; text-align:center; vertical-align:middle;
      }}
      .th {{
        display:flex; align-items:center; justify-content:center;
        white-space:normal; word-break:break-word; text-align:center; line-height:1.15; padding:2px 4px;
      }}
      td {{ white-space:nowrap; }}            /* <— JANGAN POTONG NILAI */
      td.num {{ text-align:right; }}
      td.txt {{ text-align:left;  }}
      td.code{{ text-align:left; font-weight:600; }}
      tr:nth-child(even) {{ background:#f7fbfb; }}
      tr:hover {{ background:#e8f4f3; }}
    </style>
    <div class="box">
      <div class="scroll">
        <table class="tb">
          <colgroup>{''.join(f'<col style="width:{widths.get(c,"20%")}">' for c in cols)}</colgroup>
          <thead><tr>{thead}</tr></thead>
          <tbody>{tbody}</tbody>
        </table>
      </div>
    </div>
    """
    st_html(html, height=height+HEADER_H, scrolling=False)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_net_foreign_rank_fast(start: date, end: date, min_avg_value: Optional[float], max_spread: Optional[float], topn: int):
    conn = get_db_connection()
    try:
        _ensure_alive(conn); ensure_views(conn)
        sql = """
        SELECT kode_saham, ANY_VALUE(nama_perusahaan) AS nama_perusahaan,
               SUM(net_foreign_value) AS cum_net_foreign,
               SUM(nilai) AS total_value, AVG(nilai) AS avg_value_period,
               AVG(spread_bps) AS avg_spread_bps
        FROM v_daily_fast
        WHERE trade_date BETWEEN %s AND %s
        GROUP BY kode_saham
        HAVING 1=1
        """
        p=[start,end]
        if min_avg_value and min_avg_value>0: sql+=" AND AVG(nilai)>=%s"; p.append(min_avg_value)
        if max_spread and max_spread>0:       sql+=" AND (AVG(spread_bps)<=%s OR AVG(spread_bps) IS NULL)"; p.append(max_spread)
        sql+=" ORDER BY cum_net_foreign DESC LIMIT %s"; p.append(int(topn))
        return pd.read_sql(sql, conn, params=p)
    finally:
        _safe_close(conn)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_net_foreign_rank_accurate(start: date, end: date, min_advt: Optional[float], max_spread: Optional[float], topn: int):
    conn = get_db_connection()
    try:
        _ensure_alive(conn); ensure_views(conn)
        sql = """
        SELECT d.kode_saham, ANY_VALUE(d.nama_perusahaan) AS nama_perusahaan,
               SUM(d.net_foreign_value) AS cum_net_foreign,
               SUM(d.nilai) AS total_value, AVG(r.advt_20) AS avg_advt_20,
               AVG(d.spread_bps) AS avg_spread_bps
        FROM v_daily_metrics d
        LEFT JOIN v_rolling_20d r ON r.trade_date=d.trade_date AND r.kode_saham=d.kode_saham
        WHERE d.trade_date BETWEEN %s AND %s
        GROUP BY d.kode_saham
        HAVING 1=1
        """
        p=[start,end]
        if min_advt and min_advt>0: sql+=" AND AVG(r.advt_20)>=%s"; p.append(min_advt)
        if max_spread and max_spread>0: sql+=" AND (AVG(d.spread_bps)<=%s OR AVG(d.spread_bps) IS NULL)"; p.append(max_spread)
        sql+=" ORDER BY cum_net_foreign DESC LIMIT %s"; p.append(int(topn))
        return pd.read_sql(sql, conn, params=p)
    finally:
        _safe_close(conn)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_top_movers_fast(dt: date, min_value_today: Optional[float], max_spread: Optional[float], topn: int):
    conn = get_db_connection()
    try:
        _ensure_alive(conn); ensure_views(conn)
        sql = """
        SELECT kode_saham, nama_perusahaan,
               CASE WHEN sebelumnya IS NOT NULL AND sebelumnya<>0
                    THEN (penutupan/sebelumnya)-1 END AS ret_1d,
               nilai, volume, spread_bps
        FROM v_daily_fast
        WHERE trade_date=%s
        """
        p=[dt]
        if min_value_today and min_value_today>0: sql+=" AND nilai>=%s"; p.append(min_value_today)
        if max_spread and max_spread>0: sql+=" AND (spread_bps<=%s OR spread_bps IS NULL)"; p.append(max_spread)
        df_all = pd.read_sql(sql, conn, params=p)
        return (df_all.sort_values("ret_1d", ascending=False).head(int(topn)).reset_index(drop=True),
                df_all.sort_values("ret_1d", ascending=True ).head(int(topn)).reset_index(drop=True))
    finally:
        _safe_close(conn)

# Range/filters
conn0=get_db_connection(); _ensure_alive(conn0); ensure_views(conn0)
MIN_D, MAX_D = get_date_bounds(conn0); _safe_close(conn0)

quick = st.radio("Rentang cepat",
                 ["1 Minggu","1 Bulan","3 Bulan","6 Bulan","1 Tahun","Semua","Custom"],
                 horizontal=True)

def quick_to_range(q:str)->Tuple[date,date]:
    e = MAX_D
    return {
        "1 Minggu":(MAX_D-timedelta(days=6),e),
        "1 Bulan": (MAX_D-timedelta(days=30),e),
        "3 Bulan": (MAX_D-timedelta(days=91),e),
        "6 Bulan": (MAX_D-timedelta(days=182),e),
        "1 Tahun": (MAX_D-timedelta(days=365),e),
        "Semua":   (MIN_D,e)
    }.get(q,(None,None))

if quick!="Custom":
    start_date,end_date = quick_to_range(quick)
else:
    cA,cB = st.columns(2)
    with cA: start_date = st.date_input("Tanggal Mulai", value=max(MIN_D, MAX_D-timedelta(days=29)), min_value=MIN_D, max_value=MAX_D)
    with cB: end_date   = st.date_input("Tanggal Akhir", value=MAX_D, min_value=MIN_D, max_value=MAX_D)

if start_date>end_date:
    st.error("Tanggal Mulai tidak boleh lebih besar dari Tanggal Akhir.")
    st.stop()

c = st.columns(4)
with c[0]: fast_mode = st.toggle("⚡ Mode cepat", value=True)
with c[1]:
    min_liq = st.number_input("Min **Avg Nilai (periode/ADVT20)** — Rp", min_value=0, step=100_000_000, value=1_000_000_000)
with c[2]: max_spread_bps = st.number_input("Max Spread (bps)", min_value=0, step=5, value=50)
with c[3]: topn = st.number_input("Top N", min_value=5, max_value=100, step=5, value=20)

st.markdown("---")

# Foreign Flow
st.subheader("🟩 Top Net **Buy Asing** (Akumulasi)")
if fast_mode:
    df_rank = fetch_net_foreign_rank_fast(start_date, end_date, min_liq, max_spread_bps, topn*2)
    df_rank = df_rank.rename(columns={"avg_value_period":"avg_liq"})
else:
    df_rank = fetch_net_foreign_rank_accurate(start_date, end_date, min_liq, max_spread_bps, topn*2)
    df_rank = df_rank.rename(columns={"avg_advt_20":"avg_liq"})

df_buy  = df_rank.sort_values("cum_net_foreign", ascending=False).head(topn).reset_index(drop=True)
df_sell = df_rank.sort_values("cum_net_foreign", ascending=True ).head(topn).reset_index(drop=True)

fmt_rank = {"cum_net_foreign":("num",0),"total_value":("num",0),"avg_liq":("num",0),"avg_spread_bps":("num",2)}
df_buy_fmt  = df_format_id(df_buy,  fmt_rank)
df_sell_fmt = df_format_id(df_sell, fmt_rank)
cols_rank = ["kode_saham","cum_net_foreign","total_value","avg_liq","avg_spread_bps"]

c1,c2 = st.columns(2, gap="large")
with c1:
    render_table_html(df_buy_fmt, cols_rank)
    st.download_button("⬇️ Export CSV - Top Net Buy", data=to_csv_bytes(df_buy),
                       file_name=f"net_buy_{start_date}_{end_date}.csv")
with c2:
    st.subheader("🟥 Top Net **Sell Asing** (Akumulasi)")
    render_table_html(df_sell_fmt, cols_rank)
    st.download_button("⬇️ Export CSV - Top Net Sell", data=to_csv_bytes(df_sell),
                       file_name=f"net_sell_{start_date}_{end_date}.csv")

st.markdown("---")

# Movers
st.subheader("⚡ Top Movers (1D)")
dcol,_ = st.columns([1,3])
with dcol: movers_date = st.date_input("Tanggal untuk Movers (1D)", value=end_date, min_value=MIN_D, max_value=MAX_D)
min_value_today = st.number_input("Min Nilai (hari itu) — Rp", min_value=0, step=100_000_000, value=500_000_000)

gainers, losers = fetch_top_movers_fast(movers_date, min_value_today, max_spread_bps, topn)

fmt_mov = {"ret_1d":("pct",2),"nilai":("num",0),"volume":("num",0),"spread_bps":("num",2)}
gainers_fmt = df_format_id(gainers, fmt_mov)
losers_fmt  = df_format_id(losers,  fmt_mov)
cols_mov = ["kode_saham","ret_1d","nilai","volume","spread_bps"]

mc1,mc2 = st.columns(2, gap="large")
with mc1:
    st.markdown(f"**Top Gainer — {movers_date}**")
    render_table_html(gainers_fmt, cols_mov)
    st.download_button("⬇️ Export CSV - Top Gainer", data=to_csv_bytes(gainers),
                       file_name=f"top_gainer_{movers_date}.csv")
with mc2:
    st.markdown(f"**Top Loser — {movers_date}**")
    render_table_html(losers_fmt, cols_mov)
    st.download_button("⬇️ Export CSV - Top Loser", data=to_csv_bytes(losers),
                       file_name=f"top_loser_{movers_date}.csv")

st.caption("Nilai tidak lagi terpotong (scroll X diaktifkan bila perlu). Header wrap & center, tinggi header dikunci agar kiri/kanan simetris.")
