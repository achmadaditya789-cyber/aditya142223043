import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ====================================
# PAGE CONFIG
# ====================================

st.set_page_config(
    page_title="Advanced Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

# ====================================
# HEADER
# ====================================

st.title("🚀 Advanced Data Mining Dashboard")
st.markdown(
    "Analisis Dataset Secara Interaktif Menggunakan Python dan Streamlit"
)

# ====================================
# SIDEBAR
# ====================================

st.sidebar.title("📁 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=["csv"]
)

# ====================================
# LOAD DATA
# ====================================

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # ====================================
    # DATA CLEANING
    # ====================================

    df = df.drop_duplicates()

    # ====================================
    # KPI
    # ====================================

    st.subheader("📌 Dashboard Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Rows",
        f"{df.shape[0]:,}"
    )

    col2.metric(
        "Columns",
        df.shape[1]
    )

    col3.metric(
        "Missing",
        int(df.isnull().sum().sum())
    )

    col4.metric(
        "Duplicates",
        int(df.duplicated().sum())
    )

    st.divider()

    # ====================================
    # DATASET
    # ====================================

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dataset",
        "Statistics",
        "Visualization",
        "Machine Learning"
    ])

    # ====================================
    # DATASET TAB
    # ====================================

    with tab1:

        st.subheader("📋 Dataset")

        st.dataframe(
            df,
            use_container_width=True
        )

        st.subheader("Data Types")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str),
            "Missing": df.isnull().sum()
        })

        st.dataframe(
            info_df,
            use_container_width=True
        )

    # ====================================
    # STATISTICS TAB
    # ====================================

    with tab2:

        st.subheader("📈 Statistics")

        st.dataframe(
            df.describe(),
            use_container_width=True
        )

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns

        if len(numeric_cols) > 1:

            corr = df[numeric_cols].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                title="Correlation Matrix"
            )

            st.plotly_chart(
                fig_corr,
                use_container_width=True
            )

    # ====================================
    # VISUALIZATION TAB
    # ====================================

    with tab3:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) > 0:

            col = st.selectbox(
                "Select Numeric Column",
                numeric_cols
            )

            fig_hist = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}"
            )

            st.plotly_chart(
                fig_hist,
                use_container_width=True
            )

            fig_box = px.box(
                df,
                y=col,
                title=f"Outlier Detection ({col})"
            )

            st.plotly_chart(
                fig_box,
                use_container_width=True
            )

    # ====================================
    # MACHINE LEARNING TAB
    # ====================================

    with tab4:

        numeric_cols = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if len(numeric_cols) >= 2:

            st.subheader("🤖 K-Means Clustering")

            x_col = st.selectbox(
                "Feature X",
                numeric_cols
            )

            y_col = st.selectbox(
                "Feature Y",
                numeric_cols,
                index=1
            )

            cluster = st.slider(
                "Jumlah Cluster",
                2,
                10,
                3
            )

            data = df[[x_col, y_col]].dropna()

            scaler = StandardScaler()

            scaled = scaler.fit_transform(
                data
            )

            model = KMeans(
                n_clusters=cluster,
                random_state=42,
                n_init=10
            )

            labels = model.fit_predict(
                scaled
            )

            data["Cluster"] = labels

            fig_cluster = px.scatter(
                data,
                x=x_col,
                y=y_col,
                color=data["Cluster"].astype(str),
                title="K-Means Clustering"
            )

            st.plotly_chart(
                fig_cluster,
                use_container_width=True
            )

    # ====================================
    # DOWNLOAD
    # ====================================

    st.divider()

    csv = df.to_csv(index=False)

    st.download_button(
        "⬇ Download Dataset",
        csv,
        "hasil_analisis.csv",
        "text/csv"
    )

else:

    st.info(
        "Silakan upload dataset CSV untuk memulai analisis."
    )

    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        use_container_width=True
    )
