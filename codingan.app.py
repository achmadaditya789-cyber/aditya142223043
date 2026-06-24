import streamlit as st
import pandas as pd
import plotly.express as px

# =====================================
# KONFIGURASI HALAMAN
# =====================================

st.set_page_config(
    page_title="Dashboard Data Mining",
    page_icon="📊",
    layout="wide"
)

# =====================================
# HEADER
# =====================================

st.title("📊 Dashboard Data Mining")
st.markdown("Analisis Data Interaktif Menggunakan Streamlit")

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV",
    type=["csv"]
)

# =====================================
# LOAD DATA
# =====================================

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil diupload")

    # =====================================
    # KPI
    # =====================================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Jumlah Baris",
        df.shape[0]
    )

    col2.metric(
        "Jumlah Kolom",
        df.shape[1]
    )

    col3.metric(
        "Missing Value",
        int(df.isnull().sum().sum())
    )

    st.divider()

    # =====================================
    # DATASET
    # =====================================

    st.subheader("📋 Dataset")

    st.dataframe(
        df,
        use_container_width=True
    )

    # =====================================
    # STATISTIK
    # =====================================

    st.subheader("📈 Statistik Deskriptif")

    st.dataframe(
        df.describe(include="all"),
        use_container_width=True
    )

    # =====================================
    # PILIH KOLOM NUMERIK
    # =====================================

    numeric_cols = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Pilih Kolom Numerik",
            numeric_cols
        )

        # Histogram
        st.subheader("📊 Distribusi Data")

        fig = px.histogram(
            df,
            x=selected_col,
            nbins=20,
            title=f"Distribusi {selected_col}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # Boxplot
        st.subheader("📦 Deteksi Outlier")

        fig2 = px.box(
            df,
            y=selected_col,
            title=f"Boxplot {selected_col}"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # =====================================
    # KORELASI
    # =====================================

    if len(numeric_cols) >= 2:

        st.subheader("🔗 Korelasi")

        corr = df[numeric_cols].corr()

        fig3 = px.imshow(
            corr,
            text_auto=True,
            title="Correlation Matrix"
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    # =====================================
    # DOWNLOAD
    # =====================================

    st.subheader("⬇ Download Dataset")

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hasil_analisis.csv",
        mime="text/csv"
    )

else:

    st.info(
        "Silakan upload dataset CSV terlebih dahulu."
    )
