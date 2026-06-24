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

st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Pilih File CSV",
    type=["csv"]
)

# =====================================
# LOAD DATA
# =====================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # ===============================
        # KPI
        # ===============================

        st.success("✅ Dataset berhasil diupload")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Jumlah Baris",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Jumlah Kolom",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Value",
                int(df.isnull().sum().sum())
            )

        st.divider()

        # ===============================
        # DATASET
        # ===============================

        st.subheader("📋 Preview Dataset")

        st.dataframe(
            df,
            use_container_width=True
        )

        # ===============================
        # STATISTIK DESKRIPTIF
        # ===============================

        st.subheader("📈 Statistik Deskriptif")

        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

        # ===============================
        # KOLOM NUMERIK
        # ===============================

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_cols) > 0:

            st.subheader("📊 Visualisasi Data")

            selected_col = st.selectbox(
                "Pilih Kolom Numerik",
                numeric_cols
            )

            # Histogram

            fig_hist = px.histogram(
                df,
                x=selected_col,
                nbins=20,
                title=f"Distribusi {selected_col}"
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )

            # Boxplot

            fig_box = px.box(
                df,
                y=selected_col,
                title=f"Deteksi Outlier - {selected_col}"
            )

            st.plotly_chart(
                fig_box,
                use_container_width=True
            )

        # ===============================
        # KORELASI
        # ===============================

        if len(numeric_cols) >= 2:

            st.subheader("🔗 Correlation Matrix")

            corr = df[numeric_cols].corr()

            fig_corr = px.imshow(
                corr,
                color_continuous_scale="Blues",
                title="Correlation Matrix"
            )

            st.plotly_chart(
                fig_corr,
                use_container_width=True
            )

        # ===============================
        # DOWNLOAD DATA
        # ===============================

        st.subheader("⬇️ Download Dataset")

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="hasil_analisis.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Terjadi kesalahan: {e}"
        )

else:

    st.info(
        "📁 Silakan upload dataset CSV terlebih dahulu."
    )
