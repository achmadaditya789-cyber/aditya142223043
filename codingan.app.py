import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Mining Dashboard")
st.write("Dashboard Analisis Data Menggunakan Streamlit")

# ==========================
# UPLOAD FILE
# ==========================
uploaded_file = st.sidebar.file_uploader(
    "Upload Dataset CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success("Dataset berhasil diupload!")

    # ==========================
    # PREVIEW DATA
    # ==========================
    st.header("📋 Preview Dataset")

    st.dataframe(df)

    # ==========================
    # INFORMASI DATA
    # ==========================
    st.header("ℹ️ Informasi Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Jumlah Baris", df.shape[0])
    col2.metric("Jumlah Kolom", df.shape[1])
    col3.metric("Missing Value", df.isnull().sum().sum())

    # ==========================
    # DATA CLEANING
    # ==========================
    st.header("🧹 Data Cleaning")

    if st.button("Hapus Missing Value"):
        df = df.dropna()
        st.success("Missing value berhasil dihapus")

    st.dataframe(df)

    # ==========================
    # STATISTIK DESKRIPTIF
    # ==========================
    st.header("📈 Statistik Deskriptif")

    st.dataframe(df.describe())

    # ==========================
    # VISUALISASI
    # ==========================
    st.header("📊 Visualisasi Data")

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if len(numeric_cols) > 0:

        selected_col = st.selectbox(
            "Pilih Kolom Numerik",
            numeric_cols
        )

        fig, ax = plt.subplots()

        ax.hist(df[selected_col], bins=15)

        ax.set_title(
            f"Distribusi {selected_col}"
        )

        st.pyplot(fig)

    # ==========================
    # KORELASI
    # ==========================
    st.header("🔍 Korelasi Data")

    if len(numeric_cols) >= 2:

        corr = df[numeric_cols].corr()

        st.dataframe(corr)

        fig2, ax2 = plt.subplots()

        im = ax2.imshow(corr)

        plt.colorbar(im)

        ax2.set_xticks(range(len(corr.columns)))
        ax2.set_xticklabels(
            corr.columns,
            rotation=45
        )

        ax2.set_yticks(range(len(corr.columns)))
        ax2.set_yticklabels(corr.columns)

        st.pyplot(fig2)

    # ==========================
    # K-MEANS CLUSTERING
    # ==========================
    st.header("🤖 K-Means Clustering")

    if len(numeric_cols) >= 2:

        col_x = st.selectbox(
            "Pilih Feature X",
            numeric_cols,
            key="x"
        )

        col_y = st.selectbox(
            "Pilih Feature Y",
            numeric_cols,
            key="y"
        )

        jumlah_cluster = st.slider(
            "Jumlah Cluster",
            2,
            10,
            3
        )

        data_cluster = df[[col_x, col_y]]

        scaler = StandardScaler()

        scaled = scaler.fit_transform(
            data_cluster
        )

        kmeans = KMeans(
            n_clusters=jumlah_cluster,
            random_state=42,
            n_init=10
        )

        cluster = kmeans.fit_predict(
            scaled
        )

        df["Cluster"] = cluster

        fig3, ax3 = plt.subplots()

        scatter = ax3.scatter(
            df[col_x],
            df[col_y],
            c=df["Cluster"]
        )

        ax3.set_xlabel(col_x)
        ax3.set_ylabel(col_y)
        ax3.set_title(
            "Hasil Clustering"
        )

        st.pyplot(fig3)

        st.dataframe(df)

    else:
        st.warning(
            "Minimal diperlukan 2 kolom numerik untuk clustering"
        )

    # ==========================
    # DOWNLOAD HASIL
    # ==========================
    st.header("⬇ Download Data")

    csv = df.to_csv(index=False)

    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="hasil_data_mining.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Silakan upload dataset CSV terlebih dahulu."
    )
