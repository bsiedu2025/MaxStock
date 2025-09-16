# app/pages/3_Konsol_Database.py
# -*- coding: utf-8 -*-
import streamlit as st
from db_utils import get_db_connection, execute_query, get_table_list, get_db_name

st.set_page_config(page_title="Konsol Database", page_icon="🗄️", layout="wide")

st.title("Konsol Database 🗄️")
st.caption(f"Database aktif: **{get_db_name()}**")

with st.expander("ℹ️ Petunjuk", expanded=False):
    st.markdown("""
- Gunakan query **SELECT** untuk membaca data (ditampilkan sebagai tabel).
- Untuk **DDL/DML** (CREATE/ALTER/INSERT/UPDATE/DELETE), centang opsi *Perintah DDL/DML*.
- Klik **Jalankan** untuk mengeksekusi.
    """)

query = st.text_area("SQL", height=180, placeholder="SELECT * FROM stock_prices_history LIMIT 50;")
is_dml = st.checkbox("Perintah DDL/DML (akan di-COMMIT)", value=False)

col1, col2 = st.columns([1, 2])
with col1:
    if st.button("Jalankan", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("SQL masih kosong.")
        else:
            if is_dml:
                ok, err = execute_query(query, is_dml_ddl=True)
                if err:
                    st.error(err)
                else:
                    st.success(f"Perintah berhasil. Rowcount: {ok}")
            else:
                rows, err = execute_query(query, fetch_all=True)
                if err:
                    st.error(err)
                else:
                    st.dataframe(rows if rows else [])

with col2:
    st.write("**Daftar Tabel**")
    tabs = get_table_list()
    if not tabs:
        st.info("Belum ada tabel.")
    else:
        st.code("\n".join(tabs))
