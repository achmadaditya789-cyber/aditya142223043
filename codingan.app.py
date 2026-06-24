import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Professional Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.main {
    padding-top: 1rem;
}
.metric-card {
    background-color: #f5f7fa;
    padding: 15px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================

st.title("📊 Professional Data Mining Dashboard")
st.markdown("### Analisis Dataset Interaktif Menggunakan Streamlit")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.header("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Pilih file CSV",
    type=["csv"]
)

# ==================================================
# LOAD DATA
# ==================================================

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        # ==========================================
        # DATA CLEANING
        # ==========================================

        df = df.drop_duplicates()

        # ==========================================
        # KPI SECTION
        # ==========================================

        st.subheader("📌 Dashboard Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Rows",
                f"{df.shape[0]:,}"
            )

        with col2:
            st.metric(
                "Total Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        with col4:
            st.metric(
                "Duplicate Rows",
                int(df.duplicated().sum())
            )

        st.divider()

        # ==========================================
        # TABS
        # ==========================================

        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Dataset",
            "📈 Statistics",
            "📊 Visualization",
            "🤖 Clustering"
        ])

        # ==========================================
        # TAB DATASET
        # ==========================================

        with tab1:

            st.subheader("Dataset Preview")

            st.dataframe(
                df,
                use_container_width=True
            )

            info_df = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum()
            })

            st.subheader("Column Information")

            st.dataframe(
                info_df,
                use_container_width=True
            )

        # ==========================================
        # TAB STATISTICS
        # ==========================================

        with tab2:

            st.subheader("Descriptive Statistics")

            st.dataframe(
                df.describe(include="all"),
                use_container_width=True
            )

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(numeric_cols) >= 2:

                corr = df[numeric_cols].corr()

                fig_corr = px.imshow(
                    corr,
                    text_auto=".2f",
                    title="Correlation Matrix"
                )

                st.plotly_chart(
                    fig_corr,
                    use_container_width=True
                )

        # ==========================================
        # TAB VISUALIZATION
        # ==========================================

        with tab3:

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(numeric_cols) > 0:

                selected_col = st.selectbox(
                    "Select Numeric Column",
                    numeric_cols
                )

                fig_hist = px.histogram(
                    df,
                    x=selected_col,
                    nbins=30,
                    title=f"Distribution of {selected_col}"
                )

                st.plotly_chart(
                    fig_hist,
                    use_container_width=True
                )

                fig_box = px.box(
                    df,
                    y=selected_col,
                    title=f"Outlier Detection ({selected_col})"
                )

                st.plotly_chart(
                    fig_box,
                    use_container_width=True
                )

        # ==========================================
        # TAB CLUSTERING
        # ==========================================

        with tab4:

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if len(numeric_cols) >= 2:

                st.subheader("K-Means Clustering")

                x_col = st.selectbox(
                    "Feature X",
                    numeric_cols,
                    key="x"
                )

                y_col = st.selectbox(
                    "Feature Y",
                    numeric_cols,
                    key="y"
                )

                n_cluster = st.slider(
                    "Jumlah Cluster",
                    2,
                    10,
                    3
                )

                cluster_data = df[
                    [x_col, y_col]
                ].dropna()

                scaler = StandardScaler()

                scaled_data = scaler.fit_transform(
                    cluster_data
                )

                model = KMeans(
                    n_clusters=n_cluster,
                    random_state=42,
                    n_init=10
                )

                labels = model.fit_predict(
                    scaled_data
                )

                cluster_data["Cluster"] = labels

                fig_cluster = px.scatter(
                    cluster_data,
                    x=x_col,
                    y=y_col,
                    color=cluster_data["Cluster"].astype(str),
                    title="K-Means Clustering Result"
                )

                st.plotly_chart(
                    fig_cluster,
                    use_container_width=True
                )

        # ==========================================
        # DOWNLOAD
        # ==========================================

        st.divider()

        st.subheader("⬇ Export Data")

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name="hasil_analisis.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(f"Error: {e}")

else:

    st.info(
        "Silakan upload dataset CSV untuk memulai analisis."
    )

    st.markdown("""
    ### Fitur Dashboard
    
    ✅ Upload Dataset CSV  
    ✅ Data Cleaning  
    ✅ KPI Dashboard  
    ✅ Statistik Deskriptif  
    ✅ Correlation Matrix  
    ✅ Histogram Interaktif  
    ✅ Outlier Detection  
    ✅ K-Means Clustering  
    ✅ Export Hasil Analisis  
    """)
