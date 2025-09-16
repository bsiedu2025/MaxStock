import streamlit as st
import psycopg2
import os

@st.cache_resource
def get_connection():
    try:
        conn = psycopg2.connect(st.secrets["connections"]["supabase"]["database_url"])
        return conn
    except Exception as e:
        st.error(f"Gagal terhubung ke database: {e}")
        return None

def fetch_data(query):
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                data = cur.fetchall()
            return data
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menjalankan query: {e}")
            return None
        finally:
            if conn:
                conn.close()
    return None

def execute_query(query):
    conn = get_connection()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(query)
                conn.commit()
            return True
        except Exception as e:
            st.error(f"Terjadi kesalahan saat mengeksekusi query: {e}")
            return False
        finally:
            if conn:
                conn.close()
    return False