import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Data Mining",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Data Mining")

# Upload file
uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil diupload!")

    # Preview Data
    st.subheader("📋 Preview Dataset")
    st.dataframe(df)

    # Informasi Dataset
    st.subheader("ℹ️ Informasi Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Missing Value", int(df.isnull().sum().sum()))

    # Statistik Deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.dataframe(df.describe())

    # Pilih kolom numerik
    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns

    if len(numeric_cols) > 0:

        st.subheader("📊 Grafik Batang")

        selected_col = st.selectbox(
            "Pilih Kolom",
            numeric_cols
        )

        st.bar_chart(df[selected_col])

        st.subheader("📈 Grafik Garis")

        st.line_chart(df[selected_col])

    else:
        st.warning("Tidak ada kolom numerik.")

    # Korelasi
    if len(numeric_cols) >= 2:

        st.subheader("🔍 Korelasi")

        corr = df[numeric_cols].corr()

        st.dataframe(corr)

    # Download Data
    st.subheader("⬇️ Download Dataset")

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hasil.csv",
        mime="text/csv"
    )

else:
    st.info("Silakan upload dataset CSV terlebih dahulu.")
