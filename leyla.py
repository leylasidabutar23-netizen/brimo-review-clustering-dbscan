import streamlit as st
import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import Normalizer
import collections
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import plotly.express as px
from sklearn.decomposition import PCA
# ===============================
# INITIALIZATION (WAJIB DI ATAS)
# ===============================
if 'run_id' not in st.session_state:

    st.session_state.run_id = 0
# ===============================
# PAGE CONFIG
# ===============================
# Mengganti judul utama
st.title("Klasterisasi Ulasan Pengguna Aplikasi BRImo Menggunakan Metode DBSCAN")
st.markdown("""

<style>
    /* 1. Latar belakang biru sampai ke paling atas */
    [data-testid="stAppViewContainer"] {

        background-color: #eaf2ff !important;
    }
    header[data-testid="stHeader"] {
        background-color: rgba(0,0,0,0) !important;
    }
    /* 2. Menghilangkan padding default agar judul bisa naik ke atas */
    [data-testid="stMainViewContainer"] [data-testid="stVerticalBlock"] > div:first-child {
        padding-top: 0rem !important;
    }

    /* 3. Membuat Judul di Tengah dan Naik ke Atas */
    h1 {
        color: #1e40af !important;
        text-align: center; /* Membuat judul di tengah */
        margin-top: -50px !important; /* Menaikkan judul (atur angka ini sesuai selera) */
        padding-bottom: 20px;
    }
    /* 4. Mewarnai teks lainnya */
    h2, h3, h4, h5, h6, p, label, span {
        color: #1e40af !important;
    }
    /* 5. Sidebar tetap biru */
    [data-testid="stSidebar"] {
        background-color: #d1e3ff !important;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    /* 1. Latar belakang & Header */
    [data-testid="stAppViewContainer"] { background-color: #eaf2ff !important; }
    header[data-testid="stHeader"] { background-color: rgba(0,0,0,0) !important; }
    /* 2. Judul Utama */
    h1 {
        color: #1e40af !important;
        text-align: center;
        margin-top: -50px !important;
        padding-bottom: 20px;
    }

    /* 3. Sidebar Warna Biru */

    [data-testid="stSidebar"] { background-color: #d1e3ff !important; }

    /* 4. PERBAIKAN TOMBOL: Pendek & Teks Pas */
    div.stButton > button {
        height: 2.2rem !important;
        padding: 0px 5px !important;
        font-size: 12px !important; /* Ukuran teks agak kecil supaya muat */
        width: 100% !important;
        border-radius: 8px !important;
    }
    /* 5. Merapatkan jarak antar widget di sidebar */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 0.4rem !important;
    }
</style>
""", unsafe_allow_html=True)
# ===============================
# NLTK & STEMMER
# ===============================
nltk.download("stopwords")
stop_words = set(stopwords.words("indonesian"))
custom_stopwords = {
    "yg", "ya", "yah", "aja", "nih", "deh", "dong", "sih",
    "dll", "dsb", "lah", "kok", "loh", "nya",
    "gak", "ga", "gk", "nggak", "tdk", "kn", "skrng",
    "ok", "oke", "muas", "itu", "ini",
    "terima", "kasih", "alhamdulillah", "banget", "bgt", "brimo","bri"
}

stop_words = stop_words.union(custom_stopwords)
stemmer = StemmerFactory().create_stemmer()
# ===============================
# NORMALISASI KATA (WAJIB)
# ===============================
normalisasi = {
    "muas": "puas",
    "nyusahin": "susah",
    "menyusahkan": "susah",
    "kesulitan": "sulit",
    "tdk": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak"

}

def preprocess(text):
    text = str(text).lower()
    # 1. Normalisasi kata
    for k, v in normalisasi.items():
        text = re.sub(rf"\b{k}\b", v, text)
    # 2. Gabung Negasi
    text = re.sub(r"tidak\s+bisa", "tidak_bisa", text)
    text = re.sub(r"tidak\s+mudah", "tidak_mudah", text)
    text = re.sub(r"tidak\s+login", "tidak_login", text)
    text = re.sub(r"gagal\s+login", "gagal_login", text)
    # 3. Hilangkan URL & Karakter berulang
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    # 4. Tokenisasi + Stopword
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    # 5. Stemming
    stemmed_text = stemmer.stem(" ".join(tokens))
    # 6. Filter panjang kata (min 3 huruf) & panjang ulasan (min 3 kata)
    final_tokens = [w for w in stemmed_text.split() if len(w) >= 3]
    # 🔥 Jika ulasan terlalu pendek setelah diproses, kembalikan None
    if len(final_tokens) < 3:
        return None
    return " ".join(final_tokens)
# ===============================
# SESSION STATE
# ===============================
for key in ["df", "X", "X_dbscan", "eps", "vectorizer"]:
    if key not in st.session_state:
        st.session_state[key] = None
# ==============================
# SIDEBAR
# ===============================
with st.sidebar:
    st.title("📁 MENU")
    # PERBAIKAN 1: key harus masuk ke dalam kurung fungsi
    file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key=f"uploader_{st.session_state.run_id}"
    )
    st.markdown("---")
    # Tombol Preprocessing & TF-IDF 
    col_proc1, col_proc2 = st.columns(2)
    with col_proc1:
         btn_pre = st.button("🧹 Preprocessing", use_container_width=True, key=f"btn_pre_{st.session_state.run_id}")
    with col_proc2:
         btn_tfidf = st.button("🔢 TF-IDF", use_container_width=True, key=f"btn_tfidf_{st.session_state.run_id}")

    st.markdown("---")
    # PERBAIKAN 2: Tambahkan koma setelah parameter key
    min_pts = st.number_input(
        "MinPts (disarankan ≥4)",
        min_value=4,
        max_value=50,
        key=f"min_pts_{st.session_state.run_id}", 
        value=5,
        step=1,
        format="%d"
    )
    # Tombol Analisis
    c1, c2, c3 = st.columns(3)
    with c1:
        btn_kdist = st.button("📈 K-Dist", key=f"btn_kdist_{st.session_state.run_id}")
    with c2:
        btn_dbscan = st.button("🚀 DBSCAN", key=f"btn_dbscan_{st.session_state.run_id}")
    with c3:
        btn_vis = st.button("📊 Visualisasi", key=f"btn_vis_{st.session_state.run_id}")
    st.markdown("---")
    col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
    with col_tengah:
        if st.button("🔄 Reset Aplikasi", key=f"reset_final_{st.session_state.run_id}"):
            # Hapus semua session state
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            # Tingkatkan run_id untuk reset widget uploader
            st.session_state['run_id'] = st.session_state.get('run_id', 0) + 1
            st.cache_data.clear()
            st.rerun()
# ===============================
# LOAD DATA (BACA SEKALI SAJA)
# ===============================
def load_csv_flexible(file):
    encodings = ["utf-8", "latin1", "cp1252", "ISO-8859-1"]
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, sep=";", encoding=enc, header=0, on_bad_lines="skip")
        except Exception:
            continue
    file.seek(0)
    return pd.read_csv(file, sep=";", encoding="latin1", header=0)
if file is not None and st.session_state.df is None:
    df = load_csv_flexible(file)
    # rapikan kolom
    df.columns = df.columns.str.lower().str.strip()
    df = df.loc[:, ~df.columns.str.contains("^unnamed")]
    # ===============================
    # PISAHKAN TANGGAL & REVIEW
    # ===============================
    if "review" in df.columns:
        split_data = df["review"].astype(str).str.split(";", n=1, expand=True)
        if split_data.shape[1] == 2:
            df["tanggal"] = pd.to_datetime(
                split_data[0],
                format="%d/%m/%Y %H:%M",
                errors="coerce"
            )
            df["review"] = split_data[1]
    st.session_state.df = df
    for col in df.columns:
     if "review" in col or "ulasan" in col or "content" in col:
        df.rename(columns={col: "review"}, inplace=True)
    if "tanggal" in df.columns:
        df["tanggal"] = pd.to_datetime(df["tanggal"], errors="coerce")
    st.session_state.df = df
# ===============================
# PREVIEW
# ===============================
if st.session_state.df is not None:
    if "review" in st.session_state.df.columns:
        cols = ["review"]
        if "tanggal" in st.session_state.df.columns:
            cols.append("tanggal")
        st.dataframe(
            st.session_state.df[cols],
            use_container_width=True
        )
    else:
        st.error(
            f"❌ Kolom 'review' tidak ditemukan. "
            f"Kolom tersedia: {st.session_state.df.columns.tolist()}"
        )
else:
    st.warning("⚠️ Silakan upload file terlebih dahulu")
# ===============================
# PREPROCESS
# ===============================
if btn_pre:
    if st.session_state.df is None:
        st.error("Upload data dulu")
        st.stop()
    df = st.session_state.df.copy()
    df = df.dropna(subset=["review"])
    df["review_clean"] = (
        df["review"]
        .astype(str)  
        .apply(preprocess)
    )
    df = df[df["review_clean"].str.split().str.len() >= 3]
    df.reset_index(drop=True, inplace=True)
    df["doc_id"] = df.index
    st.session_state.df = df
    st.success(f"Preprocessing selesai!")
    st.dataframe(
    df[["review", "review_clean"]],
    use_container_width=True,
    height=500
    )

# ======================================
# BUTTON TF-IDF
# ======================================
if btn_tfidf:

    # ---------- VALIDASI ----------
    if st.session_state.df is None:
        st.error("❌ Jalankan Preprocessing terlebih dahulu")
        st.stop()

    # ---------- AMBIL DATA ----------
    df = st.session_state.df.copy()

    # ---------- TF-IDF ----------
    vectorizer = TfidfVectorizer(
        min_df=5,
        max_df=0.6,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2"
    )

    X_tfidf = vectorizer.fit_transform(df["review_clean"])

    # ---------- DATAFRAME TF-IDF ----------
    tfidf_df = pd.DataFrame(
        X_tfidf.toarray(),
        columns=vectorizer.get_feature_names_out()
    )
    tfidf_df.insert(0, "Doc", range(1, len(tfidf_df) + 1))
    st.session_state.tfidf_df = tfidf_df

    # ---------- REDUKSI DIMENSI ----------
    svd = TruncatedSVD(
        n_components=5,
        random_state=42
    )
    X_reduced = svd.fit_transform(X_tfidf)

    # ---------- NORMALISASI ----------
    normalizer = Normalizer()
    X_norm = normalizer.fit_transform(X_reduced)
# Simpan hasil SVD + Normalisasi ke dataframe
    svd_df = pd.DataFrame(
    X_norm,
    columns=[f"SVD{i+1}" for i in range(X_norm.shape[1])]
)

    svd_df.insert(0, "Doc", range(1, len(svd_df)+1))

    st.session_state.svd_df = svd_df

# Tombol download
    csv_svd = svd_df.to_csv(index=False, sep=";").encode("utf-8")
    st.download_button(
    "⬇ Download Hasil SVD",
    data=csv_svd,
    file_name="hasil_svd.csv",
    mime="text/csv"
)

    # SIMPAN DBSCAN INPUT
    st.session_state.X_dbscan = X_norm
    st.session_state.vectorizer = vectorizer

    st.success(
        f"✅ TF-IDF berhasil | Dokumen: {X_norm.shape[0]} | Dimensi: {X_norm.shape[1]}"
    )

# ======================================
# TAMPILKAN MATRKS TF-IDF 
# ======================================
if st.session_state.get("tfidf_df") is not None:
    st.subheader("📊 Matriks TF-IDF")

    st.write(
        f"Dokumen: {st.session_state.tfidf_df.shape[0]} | "
        f"Term: {st.session_state.tfidf_df.shape[1]}"
    )
    tfidf_show = st.session_state.tfidf_df.reset_index(drop=True)
    st.dataframe(
        tfidf_show,
        use_container_width=True,
        hide_index=True,
        height=650
    )

# ===============================
# K-DISTANCE 
# ===============================
if btn_kdist:

    if st.session_state.X_dbscan is None:
        st.error("❌ Jalankan TF-IDF terlebih dahulu")
        st.stop()

    X = st.session_state.X_dbscan

    neigh = NearestNeighbors(n_neighbors=min_pts)
    neigh.fit(X)

    distances, _ = neigh.kneighbors(X)

    # ambil jarak tetangga ke-minPts
    k_dist = np.sort(distances[:, min_pts - 1])
    eps_suggested = np.percentile(k_dist, 70)
    st.session_state.eps = eps_suggested

    # plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_dist)
    ax.axhline(
        eps_suggested,
        linestyle="--",
        color="red",
        label=f"eps = {eps_suggested:.4f}"
    )
    ax.set_title("K-Distance Plot (DBSCAN)")
    ax.set_xlabel("Data (sorted)")
    ax.set_ylabel("Distance")
    ax.legend()

    st.pyplot(fig)
    st.info(
        f"Parameter DBSCAN → eps = {eps_suggested:.4f}, "
        f"minPts = {min_pts}, "
 
    )

# ===============================
# DBSCAN 
# ===============================
if btn_dbscan:
    if st.session_state.df is None:
        st.error("❌ Data belum tersedia")
        st.stop()

    if st.session_state.X_dbscan is None or st.session_state.eps is None:
        st.error("⚠️ Jalankan TF-IDF dan K-Distance dulu")
        st.stop()
    df_dbscan = st.session_state.df.copy()
    X = st.session_state.X_dbscan

    # ===============================
    # MODEL DBSCAN 
    # ===============================
    model = DBSCAN(
        eps=st.session_state.eps,
        min_samples=min_pts,
        metric="euclidean"
    )

    clusters = model.fit_predict(X)
    df_dbscan["cluster"] = clusters

    # ===============================
    # METRIK DASAR
    # ===============================
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = list(clusters).count(-1)

    st.subheader("📌 Hasil DBSCAN ")
    col1, col2 = st.columns(2)
    col1.metric("Jumlah Cluster", n_clusters)
    col2.metric("Jumlah Noise", n_noise)

    # ===============================
    # SILHOUETTE (TANPA NOISE)
    # ===============================
    mask = clusters != -1
    if n_clusters > 1 and np.any(mask):
        score = silhouette_score(X[mask], clusters[mask])
        st.metric("✨ Silhouette Score", f"{score:.4f}")
    st.session_state.df = df_dbscan
    st.dataframe(
    df_dbscan[["review", "cluster"]],
    use_container_width=True,
    height=500  
)

    st.success("🚀 DBSCAN selesai — cluster rapat & stabil.")
# ======================================
# VISUALISASI HASIL CLUSTERING
# ======================================
if btn_vis:
    st.subheader("📊 Analisis Klasterisasi & Persebaran Data")

    if "df" not in st.session_state or st.session_state.df is None:
        st.error("Data belum tersedia / Jalankan DBSCAN terlebih dahulu")
        st.stop()

    if st.session_state.get("X_dbscan") is None:
        st.error("Jalankan DBSCAN terlebih dahulu.")
        st.stop()

    df_final = st.session_state.df.copy()
    tfidf_df = st.session_state.get("tfidf_df")
    X = st.session_state.X_dbscan

    # ==================================================
    # 1. VISUALISASI PCA 2D
    # ==================================================
    st.write("### 📍 Titik Persebaran Klaster (PCA 2D)")

    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X)

    df_scatter = pd.DataFrame(pca_coords, columns=["x", "y"])
    df_scatter["cluster"] = df_final["cluster"].astype(str)
    df_scatter["review"] = df_final["review"].astype(str).str.wrap(35)

    fig_scatter = px.scatter(
        df_scatter,
        x="x",
        y="y",
        color="cluster",
        hover_data=["review"],
        title="Peta Kedekatan Ulasan (PCA 2D)",
        template="plotly_white"
    )

    st.plotly_chart(fig_scatter, use_container_width=True)
    st.info("💡 Titik berdekatan menunjukkan ulasan memiliki kemiripan kata.")

    # ==================================================
    # 2. DETAIL KLASTER & PENENTUAN SENTIMEN
    # ==================================================
    st.divider()
    st.subheader("📌 Detail Klaster & Kata Kunci")

    def get_top_keywords(sub_df, tfidf_matrix, top_n=10):
        if tfidf_matrix is None:
            return []
        idx = sub_df.index
        existing_idx = [i for i in idx if i in tfidf_matrix.index]
        if not existing_idx:
            return []
        mean_tfidf = tfidf_matrix.loc[existing_idx].mean(axis=0)

        stop_umum = {
            "aplikasi", "brimo", "bri", "bank", "yang", "Doc", "ini",
            "saya", "utk", "bisa", "sudah", "cepat aman",
            "hari", "malam", "pokok", "tingkat","perlu",
            "pas", "dgn","moga","kena","klo","udah"
        }

        mean_tfidf = mean_tfidf.drop(
            labels=[w for w in stop_umum if w in mean_tfidf.index],
            errors="ignore"
        )

        return mean_tfidf.sort_values(ascending=False).head(top_n).index.tolist()

    # ==============================
    # Daftar Kata Sentimen
    # ==============================
    positive_words = ["mudah", "cepat", "bagus", "mantap", "membantu", "lancar", "praktis"]
    negative_words = ["error", "gagal", "lambat", "sulit", "ganggu", "kecewa", "buruk", "ribet"]

    def tentukan_sentimen(kw):
        pos_score = sum(1 for w in kw if w in positive_words)
        neg_score = sum(1 for w in kw if w in negative_words)

        if neg_score > pos_score:
            return "Negatif"
        elif pos_score > neg_score:
            return "Positif"
        else:
            return "Netral"
    cluster_sentimen = {}
    for cl in sorted(df_final["cluster"].unique()):
        if cl == -1:
            cluster_sentimen[cl] = "Noise"
            continue

        sub = df_final[df_final["cluster"] == cl]
        kw = get_top_keywords(sub, tfidf_df)

        label = tentukan_sentimen(kw)
        cluster_sentimen[cl] = label

        with st.expander(f"🔷 Cluster {cl} ({label}) - {len(sub)} ulasan"):
            st.write(f"🔑 **Top Keywords:** {', '.join(kw)}")
            st.dataframe(sub[["tanggal", "review"]], use_container_width=True)

    # ==================================================
    # 3. GRAFIK TREN SENTIMEN BULANAN
    # ==================================================
    st.divider()
    st.subheader("📈 Tren Sentimen Bulanan")

    df_final["tanggal"] = pd.to_datetime(df_final["tanggal"], errors='coerce')
    df_final["bulan_tahun"] = df_final["tanggal"].dt.strftime('%Y-%m')

    df_final["sentimen"] = df_final["cluster"].map(cluster_sentimen)

    target_months = ["2025-11", "2025-12", "2026-01"]

    df_resmi = df_final[df_final["sentimen"].isin(["Positif", "Negatif", "Netral"])]
    df_plot = df_resmi.groupby(["bulan_tahun", "sentimen"]).size().reset_index(name="jumlah")

    dummy_rows = []
    for m in target_months:
        for s in ["Positif", "Negatif", "Netral"]:
            exists = df_plot[(df_plot["bulan_tahun"] == m) & (df_plot["sentimen"] == s)]
            if exists.empty:
                dummy_rows.append({"bulan_tahun": m, "sentimen": s, "jumlah": 0})

    df_final_plot = pd.concat([df_plot, pd.DataFrame(dummy_rows)], ignore_index=True)
    df_final_plot = df_final_plot[df_final_plot["bulan_tahun"].isin(target_months)]
    df_final_plot = df_final_plot.sort_values(["bulan_tahun", "sentimen"])

    fig_bar = px.bar(
        df_final_plot,
        x="bulan_tahun",
        y="jumlah",
        color="sentimen",
        barmode="group",
        color_discrete_map={
            "Positif": "#2ecc71",
            "Negatif": "#e74c3c",
            "Netral": "#95a5a6"
        },
        category_orders={"sentimen": ["Positif", "Negatif", "Netral"]}
    )

    fig_bar.update_xaxes(type='category', title="Periode (Bulan)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ==================================================
    # 4. TABEL REKAP
    # ==================================================
    st.subheader("📝 Rekapitulasi Akhir")

    rekap = df_final_plot.pivot(
        index="bulan_tahun",
        columns="sentimen",
        values="jumlah"
    ).fillna(0).astype(int)

    st.table(rekap[["Positif", "Negatif", "Netral"]])

