import streamlit as st
import pandas as pd

# Konfigurasi halaman
st.set_page_config(
    page_title="Dashboard Data Mining",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Dashboard Data Mining")

uploaded_file = st.file_uploader(
    "Upload File CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil diupload")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

    st.subheader("Preview Data")

    st.dataframe(df)

    st.subheader("Statistik Deskriptif")

    st.dataframe(df.describe())

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Pilih Kolom",
            numeric_cols
        )

        st.bar_chart(df[selected_col])

        st.line_chart(df[selected_col])

else:

    st.info(
        "Silakan upload file CSV"
    )
