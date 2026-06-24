import streamlit as st
import pandas as pd
import numpy as np

# ==================================
# KONFIGURASI HALAMAN
# ==================================

st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================
# HEADER
# ==================================

st.title("📊 Data Mining Dashboard")
st.markdown("Dashboard Analisis Data Menggunakan Streamlit")

st.divider()

# ==================================
# SIDEBAR
# ==================================

st.sidebar.header("⚙️ Menu")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

# ==================================
# LOAD DATA
# ==================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ==============================
    # KPI
    # ==============================

    st.subheader("📌 Ringkasan Dataset")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Jumlah Baris",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "Jumlah Kolom",
        f"{df.shape[1]}"
    )

    col3.metric(
        "Missing Value",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Duplikat",
        int(df.duplicated().sum())
    )

    st.divider()

    # ==============================
    # PREVIEW DATA
    # ==============================

    st.subheader("📋 Preview Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    # ==============================
    # TIPE DATA
    # ==============================

    st.subheader("📝 Informasi Kolom")

    info_df = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.astype(str),
        "Missing Value": df.isnull().sum()
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    # ==============================
    # MISSING VALUE
    # ==============================

    st.subheader("🔍 Analisis Missing Value")

    missing = pd.DataFrame({
        "Kolom": df.columns,
        "Jumlah Missing": df.isnull().sum(),
        "Persentase (%)":
        round(
            (df.isnull().sum()/len(df))*100,
            2
        )
    })

    st.dataframe(
        missing,
        use_container_width=True
    )

    # ==============================
    # STATISTIK DESKRIPTIF
    # ==============================

    st.subheader("📈 Statistik Deskriptif")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )

    # ==============================
    # NUMERIC COLUMNS
    # ==============================

    numeric_cols = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    # ==============================
    # VISUALISASI
    # ==============================

    if len(numeric_cols) > 0:

        st.subheader("📊 Visualisasi")

        selected_col = st.selectbox(
            "Pilih Kolom Numerik",
            numeric_cols
        )

        tab1, tab2 = st.tabs([
            "Bar Chart",
            "Line Chart"
        ])

        with tab1:
            st.bar_chart(
                df[selected_col]
            )

        with tab2:
            st.line_chart(
                df[selected_col]
            )

    # ==============================
    # KORELASI
    # ==============================

    if len(numeric_cols) >= 2:

        st.subheader("🔗 Korelasi")

        corr = df[numeric_cols].corr()

        st.dataframe(
            corr,
            use_container_width=True
        )

    # ==============================
    # FILTER DATA
    # ==============================

    st.subheader("🎯 Filter Dataset")

    selected_rows = st.slider(
        "Jumlah Data Ditampilkan",
        5,
        len(df),
        min(20, len(df))
    )

    st.dataframe(
        df.head(selected_rows),
        use_container_width=True
    )

    # ==============================
    # DOWNLOAD
    # ==============================

    st.subheader("⬇️ Download Dataset")

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hasil_data_mining.csv",
        mime="text/csv"
    )

else:

    st.info(
        "📁 Silakan upload dataset CSV terlebih dahulu."
    )

    st.markdown("""
    ### Format yang Didukung

    - CSV
    - Data Penjualan
    - Data Mahasiswa
    - Data Keuangan
    - Data Kesehatan
    - Dataset Data Mining lainnya
    """)
