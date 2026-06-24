import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# =====================================
# Konfigurasi Halaman
# =====================================

st.set_page_config(
    page_title="Data Mining Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Mining Sales Dashboard")
st.markdown("Dashboard Analisis Penjualan Menggunakan Streamlit")

# =====================================
# Load Data
# =====================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/sales_data.csv")
    return df

df = load_data()

# =====================================
# Data Cleaning
# =====================================

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

df["TotalSales"] = df["Price"] * df["Quantity"]

# =====================================
# Sidebar Filter
# =====================================

st.sidebar.header("Filter Data")

category_filter = st.sidebar.multiselect(
    "Pilih Kategori",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

city_filter = st.sidebar.multiselect(
    "Pilih Kota",
    options=df["City"].unique(),
    default=df["City"].unique()
)

filtered_df = df[
    (df["Category"].isin(category_filter))
    &
    (df["City"].isin(city_filter))
]

# =====================================
# KPI
# =====================================

total_revenue = filtered_df["TotalSales"].sum()
total_orders = len(filtered_df)
avg_sales = filtered_df["TotalSales"].mean()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "💰 Total Revenue",
        f"${total_revenue:,.0f}"
    )

with col2:
    st.metric(
        "📦 Total Orders",
        total_orders
    )

with col3:
    st.metric(
        "📈 Average Sales",
        f"${avg_sales:,.2f}"
    )

# =====================================
# Dataset
# =====================================

st.subheader("📋 Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True
)

# =====================================
# Statistik Deskriptif
# =====================================

st.subheader("📊 Statistik Deskriptif")

st.dataframe(
    filtered_df.describe(),
    use_container_width=True
)

# =====================================
# Sales per Category
# =====================================

st.subheader("📈 Sales per Category")

category_sales = (
    filtered_df
    .groupby("Category")["TotalSales"]
    .sum()
    .reset_index()
)

fig1, ax1 = plt.subplots(figsize=(6,4))

ax1.bar(
    category_sales["Category"],
    category_sales["TotalSales"]
)

ax1.set_xlabel("Category")
ax1.set_ylabel("Revenue")
ax1.set_title("Revenue by Category")

st.pyplot(fig1)

# =====================================
# Sales per City
# =====================================

st.subheader("🏙️ Sales per City")

city_sales = (
    filtered_df
    .groupby("City")["TotalSales"]
    .sum()
    .reset_index()
)

fig2, ax2 = plt.subplots(figsize=(6,6))

ax2.pie(
    city_sales["TotalSales"],
    labels=city_sales["City"],
    autopct="%1.1f%%"
)

ax2.set_title("Revenue Distribution by City")

st.pyplot(fig2)

# =====================================
# Clustering K-Means
# =====================================

st.subheader("🤖 Customer/Product Clustering")

cluster_df = filtered_df.copy()

X = cluster_df[["Price", "Quantity"]]

if len(cluster_df) >= 3:

    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )

    cluster_df["Cluster"] = kmeans.fit_predict(X)

    fig3, ax3 = plt.subplots(figsize=(8,5))

    scatter = ax3.scatter(
        cluster_df["Price"],
        cluster_df["Quantity"],
        c=cluster_df["Cluster"]
    )

    ax3.set_xlabel("Price")
    ax3.set_ylabel("Quantity")
    ax3.set_title("K-Means Clustering")

    st.pyplot(fig3)

    st.dataframe(
        cluster_df,
        use_container_width=True
    )

else:
    st.warning(
        "Data tidak cukup untuk clustering."
    )

# =====================================
# Download Dataset
# =====================================

st.subheader("⬇️ Download Data")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="sales_analysis.csv",
    mime="text/csv"
)

# =====================================
# Footer
# =====================================

st.markdown("---")
st.markdown(
    "Dibuat dengan ❤️ menggunakan Streamlit dan Python"
)