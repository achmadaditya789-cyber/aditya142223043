import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Mining Dashboard")

uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset berhasil diupload")

        # KPI
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Missing", int(df.isnull().sum().sum()))
        col4.metric("Duplicate", int(df.duplicated().sum()))

        st.divider()

        # Dataset
        st.subheader("📋 Dataset")

        st.dataframe(
            df,
            use_container_width=True
        )

        # Statistik
        st.subheader("📈 Statistik Deskriptif")

        st.dataframe(
            df.describe(include="all")
        )

        # Kolom Numerik
        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) > 0:

            st.subheader("📊 Visualisasi")

            selected_col = st.selectbox(
                "Pilih Kolom",
                numeric_cols
            )

            fig = px.histogram(
                df,
                x=selected_col,
                title=f"Distribusi {selected_col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        # Korelasi
        if len(numeric_cols) >= 2:

            st.subheader("🔗 Korelasi")

            corr = df[numeric_cols].corr()

            fig_corr = px.imshow(
                corr,
                aspect="auto"
            )

            st.plotly_chart(
                fig_corr,
                use_container_width=True
            )

        # Download
        csv = df.to_csv(index=False)

        st.download_button(
            "⬇ Download CSV",
            csv,
            "hasil_analisis.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(
            f"Terjadi error: {e}"
        )

else:

    st.info(
        "Silakan upload dataset CSV terlebih dahulu."
    )
