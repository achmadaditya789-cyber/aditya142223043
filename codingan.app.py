import streamlit as st
import pandas as pd

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Professional Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

# =====================================
# HEADER
# =====================================

st.title("📊 Professional Data Mining Dashboard")
st.markdown("### Analisis Dataset Menggunakan Streamlit")

st.divider()

# =====================================
# SIDEBAR
# =====================================

st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File CSV",
    type=["csv"]
)

# =====================================
# LOAD DATA
# =====================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        # ==============================
        # KPI
        # ==============================

        st.subheader("📌 Dashboard Overview")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Jumlah Baris",
            f"{df.shape[0]:,}"
        )

        col2.metric(
            "Jumlah Kolom",
            df.shape[1]
        )

        col3.metric(
            "Missing Value",
            int(df.isnull().sum().sum())
        )

        col4.metric(
            "Data Duplikat",
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
        # INFO KOLOM
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
        # STATISTIK
        # ==============================

        st.subheader("📈 Statistik Deskriptif")

        st.dataframe(
            df.describe(include="all"),
            use_container_width=True
        )

        # ==============================
        # FILTER DATA
        # ==============================

        st.subheader("🎯 Filter Dataset")

        jumlah_data = st.slider(
            "Jumlah Baris Ditampilkan",
            5,
            len(df),
            min(20, len(df))
        )

        st.dataframe(
            df.head(jumlah_data),
            use_container_width=True
        )

        # ==============================
        # KOLOM NUMERIK
        # ==============================

        numeric_cols = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numeric_cols) > 0:

            st.subheader("📊 Visualisasi Data")

            selected_col = st.selectbox(
                "Pilih Kolom Numerik",
                numeric_cols
            )

            tab1, tab2 = st.tabs([
                "Bar Chart",
                "Line Chart"
            ])

            with tab1:
                st.bar_chart(df[selected_col])

            with tab2:
                st.line_chart(df[selected_col])

        # ==============================
        # KORELASI
        # ==============================

        if len(numeric_cols) >= 2:

            st.subheader("🔗 Korelasi Data")

            corr = df[numeric_cols].corr()

            st.dataframe(
                corr,
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
            file_name="hasil_analisis.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Terjadi Error: {e}"
        )

else:

    st.info(
        "📁 Silakan upload dataset CSV terlebih dahulu."
    )

    st.markdown("""
    ### Fitur Dashboard

    ✅ Upload Dataset CSV  
    ✅ KPI Dashboard  
    ✅ Informasi Kolom  
    ✅ Statistik Deskriptif  
    ✅ Filter Data  
    ✅ Visualisasi Data  
    ✅ Korelasi Data  
    ✅ Download Dataset
    """)
