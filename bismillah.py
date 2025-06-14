import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
# from IPython.display import display
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA



st.set_page_config(layout="wide")

import streamlit as st

st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap" rel="stylesheet">

""", unsafe_allow_html=True)

st.markdown(
    """
    <div style="height : 70px">

    </div>


""",unsafe_allow_html=True
)

st.markdown("""
<style>

div.stButton > button:first-child {
    background-color: white;
    color: red;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    border: 2px solid red;
}
div.stButton > button:hover {
    background-color: red;
    color : white;
}
/* Ubah max-width dari uploader container */
div[data-testid="stFileUploader"] > label {
    width: 600px;
}

/* Ubah elemen uploader-nya sendiri */
div[data-testid="stFileUploader"] > div {
    width: 600px;
    background-color: #f7f7f7;
    padding: 10px;
    border-radius: 8px;
}
.block-container {
    padding: 0 !important;
}
.navbar-container {
    position: sticky;
    # top: 0;
    z-index: 1000;
    display: flex;
    justify-content: space-between;
    background-color: #F2F8FD;
    border-radius: 999px;
    display: flex;
    align-items: center;
    font-family: 'Poppins', sans-serif;
    box-shadow: 0 0 0 12px #f0f7ff;
    margin-right : 40px;
    margin-left : 40px
    
}

.nav-brand {
    background-color: #0D2438;
    color: white;
    font-weight: bold;
    padding: 10px 30px;
    border-radius: 999px;
    font-size: 16px;
    margin-right: 40px;
    text-decoration: none;
}

.nav-link {
    margin-right: 20px;
    text-decoration: none;
    color: #333;
    font-weight: 500;
    font-size: 14px;
    text-decoration: none;
}

.nav-link:hover {
    text-decoration: none;
}
            
.problem-box {
    background-color: #0D2438;
    color: white;
    padding: 30px;
    border-radius: 16px;
    font-family: 'Segoe UI', sans-serif;
    margin: 40px 40px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    text-align : justify;
}

.problem-box h3 {
    margin-top: 0;
    font-size: 24px;
    font-weight: bold;
}

.problem-box p {
    font-size: 15px;
    line-height: 1.6;
    color: #f1f1f1;
}
.problem-box1 {
    background-color: #0D2438;
    color: white;
    padding: 10px 30px ;
    
    border-radius: 16px;
    font-family: 'Segoe UI', sans-serif;
    margin: 40px 40px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    text-align : justify;
    -webkit-box-shadow: 1px 9px 10px -8px #747474; 
    box-shadow: 1px 9px 10px -8px #747474;
}

.problem-box1 h3 {
    margin-top: 0;
    font-size: 24px;
    font-weight: 700px;
}

.problem-box1 p {
    font-size: 15px;
    line-height: 1.6;
    color: #f1f1f1;
}
.problem-box2 {
    background-color: #ffdada;
    border: 3px solid red;
    color: red;
    padding: 10px 30px ;
    
    border-radius: 16px;
    font-family: 'Segoe UI', sans-serif;
    margin: 40px 40px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    text-align : justify;
    -webkit-box-shadow: 1px 9px 10px -8px #747474; 
    box-shadow: 1px 9px 10px -8px #747474;
}

.problem-box2 h3 {
    margin-top: 0;
    font-size: 14px;
    font-weight: 700px;
}

.problem-box2 p {
    font-size: 15px;
    line-height: 1.6;
    color: #f1f1f1;
}

.section {
      margin-bottom: 60px;
    }
            

    .grid {
      display: flex;
      flex-wrap: wrap;
      gap: 20px;
    }

    .card {
      background-color: #0c1c3f;
      color: white;
      padding: 20px;
      border-radius: 12px;
      flex: 1 1 calc(33.333% - 20px);
      box-sizing: border-box;
      min-width: 280px;
      box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    .card h3 {
      font-size: 1.1rem;
      margin-bottom: 10px;
    }

    @media (max-width: 768px) {
      .card {
        flex: 1 1 100%;
      }
    }
</style>

<div class="navbar-container">
        <a class="nav-brand" style="text-decoration: none;color:white" href="#">MINIKIWIR</a>
    <div >
        <a class="nav-link" style="text-decoration: none; color:black" href="#home">Home</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#upload">Upload File</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#bu">Bussiness Understanding</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#du">Data Understanding</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#dp">Data Preparation</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#km">K-Means</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#silhouette">Silhouette</a>
        <a class="nav-link" style="text-decoration: none; color:black" href="#ad">Analisis Data</a>
    </div>
    
</div>
""", unsafe_allow_html=True)

st.markdown(
    """
     <div class="title" style="font-size : 34px; padding-right : 10px; padding-left: 10px; font-weight: 900; text-align: center; margin-top : 50px">
        <p>
            SEGMENTASI UMKM MENGGUNAKAN METODE K-MEANS  CLUSTERING  UNTUK PERUMUSAN STRATEGI  PEMBINAAN  BERDASARKAN PREFERENSI MARKETPLACE
        </p>
    </div>
    <div class="title" style="font-size : 18px; padding-right : 40px; padding-left: 40px; font-weight: 900; text-align: center; margin-top : 20px">
        <p>
        Website ini dirancang untuk membantu proses segmentasi UMKM berdasarkan preferensi marketplace yang mereka gunakan. Tujuan utama dari analisis ini adalah untuk mendukung strategi pembinaan yang lebih terarah dan berbasis data menggunakan algoritma K-Means Clustering.
        </p>
    </div> """

    ,unsafe_allow_html=True

)

components.html(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

    <style>
    .custom-carousel img {
        height: 300px;
        object-fit: cover;
        border-radius: 16px;
    }

    .custom-carousel .carousel-inner {
        border-radius: 16px;
        overflow: hidden;
    }
    </style>
    <div class="px-5">
        <div id="carouselExampleSlidesOnly" class="carousel slide custom-carousel" data-bs-ride="carousel" data-bs-interval="3000">
            <div class="carousel-inner">
                <div class="carousel-item active">
                    <img src="https://images.unsplash.com/photo-1598063414123-d8fd7fb018b2?q=80&w=2070&auto=format&fit=crop" class="d-block w-100" alt="...">
                </div>
                <div class="carousel-item">
                    <img src="https://images.unsplash.com/photo-1694884443156-11383e30fa4b?q=80&w=2070&auto=format&fit=crop" class="d-block w-100" alt="...">
                </div>
                <div class="carousel-item">
                    <img src="https://images.unsplash.com/photo-1639846921405-7f6e6996b0c8?q=80&w=1931&auto=format&fit=crop" class="d-block w-100" alt="...">
                </div>
            </div>
        </div>
    </div>
    """,
    height=300,  # Tinggi iframe agar tidak terpotong
    width=1920
)

st.markdown("""

    <div class="problem-box">
        <div id="bu"></div>
        <h2>Permasalahan</h2>
        <p>
            UMKM di Indonesia menghadapi berbagai tantangan yang menghambat pertumbuhannya, seperti keterbatasan akses pasar,
            kesulitan memperoleh pembiayaan, serta strategi pemasaran yang belum optimal. Dampak pandemi COVID-19 juga
            memperburuk kondisi ini, menyebabkan banyak UMKM kesulitan dalam beradaptasi dengan perubahan pasar.
            Untuk itu, dibutuhkan pendekatan yang lebih terarah dalam mengidentifikasi segmen-segmen UMKM yang membutuhkan
            intervensi spesifik. Salah satu solusi yang dapat diterapkan adalah metode K-Means clustering, yang dapat membantu
            mengelompokkan UMKM berdasarkan karakteristik tertentu, seperti volume penjualan atau aktivitas digital.
        </p>
    </div>
    <div class="section" style="margin-left : 40px; margin-right : 40px">
    <h2>Tujuan</h2>
    <div class="grid">
      <div class="card">
        <h3>Clustering UMKM dengan K-Means</h3>
        <p>Untuk menganalisis penerapan metode K-Means clustering pada UMKM dalam mengelompokkan berdasarkan karakteristik yang diperlukan.</p>
      </div>
      <div class="card">
        <h3>Optimasi cluster via Silhouette Score</h3>
        <p>Untuk menilai hasil evaluasi penggunaan metode Silhouette Coefficient (SC) dalam menentukan jumlah cluster yang optimal pada pengelompokan UMKM.</p>
      </div>
      <div class="card">
        <h3>Segmentasi UMKM yang tepat sasaran</h3>
        <p>Untuk memahami bagaimana label cluster terbentuk berdasarkan jumlah cluster yang optimal, agar segmentasi UMKM lebih jelas dan tepat sasaran.</p>
      </div>
    </div>
  </div>

    <div class="section" style="margin-left : 40px; margin-right : 40px">
        <h2>Manfaat</h2>
        <div class="grid">
        <div class="card">
            <h3>Strategi pembinaan terarah</h3>
            <p>Memberikan panduan bagi pemerintah dan lembaga pembinaan UMKM untuk merancang strategi pembinaan yang lebih terarah sesuai dengan kebutuhan masing-masing kelompok UMKM.</p>
        </div>
        <div class="card">
            <h3>Peningkatan daya saing UMKM</h3>
            <p>Menyediakan solusi bagi UMKM untuk meningkatkan daya saing dengan membuka peluang akses pasar yang lebih optimal dan adaptif terhadap perubahan pasca-pandemi.</p>
        </div>
        <div class="card">
            <h3>Literatur segmentasi UMKM</h3>
            <p>Menyediakan kontribusi pada literatur tentang penggunaan metode data mining dalam segmentasi UMKM, khususnya melalui K-Means clustering dan Silhouette Coefficient.</p>
        </div>
        <div class="card">
            <h3>Keputusan berbasis data</h3>
            <p>Memberikan dasar objektif untuk pengambilan keputusan dalam pembinaan UMKM, sehingga strategi yang diterapkan lebih tepat dan efisien.</p>
        </div>
        </div>
    </div>

    <div class="problem-box">
        <h2>Metode Usulan</h2>
        <p>
            Metode evaluasi yang umum digunakan dalam menentukan jumlah cluster yang optimal adalah Silhouette Coefficient. Dengan menggunakan Silhouette Coefficient, kita dapat mengetahui sejauh mana objek dalam suatu cluster memiliki kemiripan yang lebih tinggi dengan objek dalam cluster yang sama dibandingkan dengan objek dalam cluster lain. Oleh karena itu, khususnya metode K-Means clustering dengan metode evaluasi Silhouette Coefficient, UMKM dapat dikelompokkan dan permasalahannya meliputi: sejauh mana penerapan metode K-Means, bagaimana hasil evaluasi menggunakan metode Silhouette Coefficient, serta bagaimana label hasil cluster terbentuk berdasarkan jumlah cluster yang optimal.
            <br>Berdasarkan penelitian sebelumnya, Metode K-Means dipilih karena sederhana dan efisiensi dalam pengelompokkan sehingga mudah diaplikasikan diberbagai bidang (Jurnal 1 nanti dibenerin di word), selain itu juga penggunaan Metode Sillhouette Coeficient (SC) digunakan untuk menentukan jumlah cluster yang paling optimal saat data preparation (Jurnal 2 dibenerin di world)
        </p>
    </div>
    <div id="upload"></div>
    <h2 id="upload" style="margin-top : 50px;margin-left:40px">Upload File</h2>
    """,unsafe_allow_html=True

)
col1, col2, col3 = st.columns([1, 34, 1])

with col2:
    uploaded_file = st.file_uploader("", type=["csv"])

st.markdown(
    """
    <div id="du">

    </div> """, unsafe_allow_html=True
)
st.markdown(
    """
    <div class="problem-box1">
        <h3>Data Understanding</h3>
    </div>""",
    unsafe_allow_html=True
)
baris = 0
kolom = 0
jumlah_numerik = 0
jumlah_kategori = 0
df = None
if uploaded_file is not None:
    
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5 :
        df = pd.read_csv(uploaded_file)
        baris = df.shape[0]
        kolom = df.shape[1]
        numerik_cols = df.select_dtypes(include=['number']).columns
        jumlah_numerik = len(numerik_cols)
        category_cols = df.select_dtypes(include=['object']).columns
        jumlah_kategori = len(category_cols)


if df is not None:
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5:
        st.markdown("""
            <style>
                .notif-container {
                    display: flex;
                    align-items: center;
                    background-color: #f9f9f9;
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    font-family: 'Segoe UI', sans-serif;
                    width: 100%;
                    margin-top: 20px;
                    margin-bottom : 20px;
                }

                .notif-icon {
                    background-color: #28a745; /* Warna hijau */
                    color: white;
                    width: 200px;
                    height: 200px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .notif-icon svg {
                    width: 48px;
                    height: 48px;
                }

                .notif-content {
                    padding: 20px;
                    flex: 1;
                }

                .notif-title {
                    font-size: 24px;
                    font-weight: bold;
                    color: #333;
                    margin-bottom: 12px;
                }

                .notif-text {
                    color: #999;
                    font-size: 16px;
                    line-height: 1.6;
                }

                .notif-close {
                    color: #ccc;
                    font-weight: bold;
                    padding: 20px;
                    font-size: 13px;
                }
            </style>
            """, unsafe_allow_html=True)
        
        st.markdown(
            f"""
                <div class="notif-container">
                <div class="notif-icon">
                    <svg viewBox="0 0 24 24" fill="none">
                        <circle cx="12" cy="12" r="10" stroke="white" stroke-width="2"/>
                        <polyline points="8 12.5 11 15.5 16 9.5" stroke="white" stroke-width="2" fill="none"/>
                    </svg>
                </div>
                <div class="notif-content">
                    <div class="notif-title">Data Tersedia</div>
                    <div class="notif-text">
                        Jumlah Data : {baris}<br>
                        Jumlah variabel : {kolom}<br>
                        Jumlah Kolom Numerik : {jumlah_numerik}<br>
                        Jumlah Kolom Kategori : {jumlah_kategori}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True
        )

        st.text("\n\nOverview Data          : ")
        st.dataframe(df)

if df is None:
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5:
        import streamlit as st

        st.markdown("""
        <style>
            .notif-container {
                display: flex;
                align-items: center;
                background-color: #f9f9f9;
                border-radius: 12px;
                overflow: hidden;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                font-family: 'Segoe UI', sans-serif;
                width: 100%;
                margin-top: 20px;
            }

            .notif-icon {
                background-color: #f55757;
                color: white;
                width: 200px;
                height: 200px;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .notif-icon svg {
                width: 48px;
                height: 48px;
            }

            .notif-content {
                padding: 20px;
                flex: 1;
            }

            .notif-title {
                font-size: 24px;
                font-weight: bold;
                color: #333;
                margin-bottom: 12px;
            }

            .notif-text {
                color: #999;
                font-size: 16px;
                line-height: 1.6;
            }

            .notif-close {
                color: #ccc;
                font-weight: bold;
                padding: 20px;
                font-size: 13px;
            }
        </style>

        <div class="notif-container">
            <div class="notif-icon">
                <svg viewBox="0 0 24 24" fill="none">
                    <circle cx="12" cy="12" r="10" stroke="white" stroke-width="2"/>
                    <line x1="9" y1="9" x2="15" y2="15" stroke="white" stroke-width="2"/>
                    <line x1="15" y1="9" x2="9" y2="15" stroke="white" stroke-width="2"/>
                </svg>
            </div>
            <div class="notif-content">
                <div class="notif-title">Tidak Ada Data</div>
                <div class="notif-text">
                    Jumlah Data : 0<br>
                    Jumlah variabel : 0<br>
                    Jumlah Kolom Numerik : 0<br>
                    Jumlah Kolom Kategori : 0
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# DATA PREPARATIONNNNNN ====================
st.markdown(
    """
    <div class="problem-box1">
        <h3>Data Preparation</h3>
    </div>""",
    unsafe_allow_html=True
)
after_drop_missing_df = None
count_missing_rows = None

if df is not None:
# rubah invalid values pada kolom numerik
    col = ['jenis_usaha', 'marketplace', 'status_legalitas','nama_usaha', 'id_umkm']

    for i in df.columns :
        if i not in col :
            df[i] = pd.to_numeric(df[i], errors='coerce').astype('Int64')

    # rubah invalid values pada kolom kategorikal
    col = ['jenis_usaha', 'marketplace', 'status_legalitas']


    for i in col :
        df[i] = df[i].replace('unknown', np.nan)

    missing_rows = df[df.isnull().any(axis=1)]
    count_missing_rows = len(missing_rows)
    print(count_missing_rows)
    
    after_drop_missing_df = df.dropna()

dataset_ori_clean = None
if count_missing_rows is not None:
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5:
        st.text(f"Jumlah Baris Invalid dan Missing Values                   : {count_missing_rows}")
        st.text(f"Jumlah Baris setelah Drop Missing dan Invalid Values      : {after_drop_missing_df.shape[0]}")
        dataset_ori_clean = after_drop_missing_df.copy()

if count_missing_rows is None:
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5:
        st.text(f"Jumlah Baris Invalid dan Missing Values                   :0")
        st.text(f"Jumlah Baris setelah Drop Missing dan Invalid Values      : 0")


# ================================================= TRANSFORMASI DATA ============================================
st.markdown(
    """
    <h4 style="margin-top : 0;margin-left:40px">a. Transformasi Data</h4>

""",unsafe_allow_html=True
)

st.markdown(
    """
    <h6 style="margin-top : 0;margin-left:70px">    1. Hasil Encode + Normalisasi Data</h6>

""",unsafe_allow_html=True
)
normalisasi_df = None

if after_drop_missing_df is not None : 
    after_drop_missing_df['jenis_usaha'] = after_drop_missing_df['jenis_usaha'].astype('category')
    after_drop_missing_df['marketplace'] = after_drop_missing_df['marketplace'].astype('category')
    after_drop_missing_df['status_legalitas'] = after_drop_missing_df['status_legalitas'].astype('category')
    after_drop_missing_df['status_legalitas'] = after_drop_missing_df['status_legalitas'].map({'Terdaftar': 1, 'Belum Terdaftar': 0})
    # after_drop_missing_df.head(5)
    # dataset_new = dataset.copy()
    after_drop_missing_df['tahun_berdiri'] = after_drop_missing_df['tahun_berdiri'].astype('category')
    num_cols = after_drop_missing_df.select_dtypes(include='number').columns
    after_drop_missing_df_new = after_drop_missing_df.select_dtypes(include='number')
    after_drop_missing_df[num_cols] = (after_drop_missing_df[num_cols] - after_drop_missing_df[num_cols].min()) / (after_drop_missing_df[num_cols].max() - after_drop_missing_df[num_cols].min())
    normalisasi_df = True

if normalisasi_df is not None :
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5:
        st.dataframe(after_drop_missing_df)
        
if normalisasi_df is None:
    col4, col5, col6 = st.columns([1, 1, 1])
    with col5:
        st.markdown("""
            <style>
            .custom-btn {
                background-color: transparent;
                color: red;
                border: 2px solid red;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                transition: all 0.3s ease;
            }

            .custom-btn:hover {
                background-color: red;
                color: white;
            }
            </style>

            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data</button>
            </a>
            """, unsafe_allow_html=True
        )


# ======================================== normalisasi =================================

# ===================================== FEATURE ENGINEERING ==============================
st.markdown(
    """
    <h4 style="margin-top : 0;margin-left:40px">b. Features Engineering</h4>

""",unsafe_allow_html=True
)
# dataset_new['jumlah_karyawan'] = dataset_new['tenaga_kerja_laki_laki'] + dataset_new['tenaga_kerja_perempuan']
# dataset_new.drop(columns=['tenaga_kerja_laki_laki', 'tenaga_kerja_perempuan'], inplace=True)
if after_drop_missing_df is not None : 
    after_drop_missing_df['jumlah_karyawan'] = after_drop_missing_df['tenaga_kerja_laki_laki'] + after_drop_missing_df['tenaga_kerja_perempuan']
    after_drop_missing_df.drop(columns=['tenaga_kerja_laki_laki', 'tenaga_kerja_perempuan'], inplace=True)
    col4, col5, col6 = st.columns([1, 34, 1])
    with col5:
        st.dataframe(after_drop_missing_df)
# after_drop_missing_df.head(3)
if after_drop_missing_df is None :
    col4, col5, col6 = st.columns([1, 1, 1])
    with col5:
        st.markdown(
            """
            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data</button>
            </a>
            """, unsafe_allow_html=True
        )

# =======================================================================================





# ===================================== CORRELATION MATRIX ==============================

st.markdown(
    """
    <h4 style="margin-top : 0;margin-left:40px">c. Correlation Matrix</h4>

""",unsafe_allow_html=True
)
st.markdown(
    """
    <h6 style="margin-top : 0;margin-left:70px">1. Before Delete Correlation Column</h6>

""",unsafe_allow_html=True
)

if after_drop_missing_df is not None:
    col4, col5, col6 = st.columns([1, 5, 1])
    with col5:
        df_numeric = after_drop_missing_df.select_dtypes(include=['number'])

        # Hitung correlation matrix
        corr_matrix = df_numeric.corr()


        # Tampilkan correlation matrix dengan heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

        # drop kolom dengan std lebih rendah
        print(after_drop_missing_df[['omset', 'laba']].std())
        after_drop_missing_df.drop(columns=['omset'], inplace=True)

        print(after_drop_missing_df[['jumlah_pelanggan', 'kapasitas_produksi']].std())
        after_drop_missing_df.drop(columns=['jumlah_pelanggan'], inplace=True)

        print(after_drop_missing_df[['biaya_karyawan', 'jumlah_karyawan']].std())
        after_drop_missing_df.drop(columns=['jumlah_karyawan'], inplace=True)

if after_drop_missing_df is None:
    col10, col11, col12 = st.columns([1, 1, 1])
    with col11:
        st.markdown(
            """
            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data</button>
            </a>
            """, unsafe_allow_html=True
        )

st.markdown(
    """
    <h6 style="margin-top : 0;margin-left:70px">2. After Delete Correlation Column</h6>

""",unsafe_allow_html=True
)

if after_drop_missing_df is not None:
    col4, col5, col6 = st.columns([1, 5, 1])
    with col5:
        after_drop_missing_df['tahun_berdiri'] = after_drop_missing_df['tahun_berdiri'].astype('int')
        df_numeric = after_drop_missing_df.select_dtypes(include=['number'])

        # Hitung correlation matrix
        corr_matrix = df_numeric.corr()


        # Tampilkan correlation matrix dengan heatmap
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)


if after_drop_missing_df is None:
    col10, col11, col12 = st.columns([1, 1, 1])
    with col11:
        st.markdown(
            """
            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data</button>
            </a>
            """, unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="problem-box1">
        <h3>Modeling</h3>
    </div>""",
    unsafe_allow_html=True
)
score = None

if after_drop_missing_df is not None:
    col10, col11, col12 = st.columns([1, 34, 1])
    with col11:
        k = st.slider("slide")
        if st.button("Mulai Kluster") :
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(df_numeric)
            score = silhouette_score(df_numeric, labels)

if after_drop_missing_df is None:
    col10, col11, col12 = st.columns([1, 1, 1])
    with col11:
        st.markdown(
            """
            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data</button>
            </a>
            """, unsafe_allow_html=True
        )

st.markdown(
    """
    <div class="problem-box1">
        <h3>Evaluation</h3>
    </div>""",
    unsafe_allow_html=True
)

if score is not None:
    col10, col11, col12 = st.columns([1, 34, 1])
    with col11:
        st.text(f"Score Silhouette Coeffecient : {score}")

if score is None:
    col10, col11, col12 = st.columns([1, 1, 1])
    with col11:
        st.markdown(
            """
            <style>
            .custom-btn {
                background-color: transparent;
                color: red;
                border: 2px solid red;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                transition: all 0.3s ease;
            }

            .custom-btn:hover {
                background-color: red;
                color: white;
            }
            </style>
            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data</button>
            </a>
            """, unsafe_allow_html=True
        )



st.markdown(
    """
    <div class="problem-box1">
        <h3>Hasil</h3>
    </div>""",
    unsafe_allow_html=True
)

unique = None
if score is not None:
    col10, col11, col12 = st.columns([1, 5, 1])
    with col11:
        X = df_numeric.drop(columns=['cluster'], errors='ignore')

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)

        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(X)

        # Tambahkan kolom cluster
        # df_numeric['cluster'] = clusters
        cluster_series = pd.Series(clusters, index=df_numeric.index, name='cluster')
        # dataset['cluster'] = clusters
        # dataset_new['cluster'] = clusters

        # Hitung dan transform centroid ke ruang PCA
        centroids = kmeans.cluster_centers_
        centroids_pca = pca.transform(centroids)

        # Buat figure
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7, label='Data')
        ax.scatter(centroids_pca[:, 0], centroids_pca[:, 1], c='red', marker='X', s=200, label='Centroid')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        ax.set_title('Visualisasi Cluster (PCA)')
        ax.legend()
        ax.grid(True)

        # Tambahkan colorbar (manual karena colorbar default tidak langsung bisa di Streamlit)
        from matplotlib.cm import ScalarMappable
        cbar = fig.colorbar(ScalarMappable(cmap='viridis'), ax=ax, label='Cluster')

        # Tampilkan ke Streamlit
        st.pyplot(fig)
    st.markdown(
        """
        <h6 style="margin-top : 0;margin-left:70px">Overview Dataset : </h6>

    """,unsafe_allow_html=True
    )
    
    col10, col11, col12 = st.columns([1, 5, 1])
    with col11:
        dataset_ori_clean['cluster'] = cluster_series+1
        unique = list(dataset_ori_clean['cluster'].unique())
        st.dataframe(dataset_ori_clean)
    st.badge("Overview Tiap Cluster : ", color="green")
    col13, col14, col15 = st.columns([1, 28, 1])
    with col14:
        for label in sorted(unique):
            with st.container():
                st.markdown(
                    f"""
                        <div style="display:inline-block; background-color:#122B3F; color:white; padding:12px 24px; font-size:16px; font-weight:bold; border-radius:32px; cursor:pointer; text-align:center; margin-bottom : 16px">
                            Kluster {label}
                        </div>
                    """, unsafe_allow_html=True
                )
                df_label = dataset_ori_clean[dataset_ori_clean['cluster'] == label]
                st.dataframe(df_label)


if score is None:
    col10, col11, col12 = st.columns([1, 1, 1])
    with col11:
        st.markdown(
            """
            <style>
            .custom-btn {
                background-color: transparent;
                color: red;
                border: 2px solid red;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                cursor: pointer;
                width: 100%;
                transition: all 0.3s ease;
            }

            .custom-btn:hover {
                background-color: red;
                color: white;
            }
            </style>
            <a href="#upload" style="text-decoration: none;">
                <button class="custom-btn">Belum Ada Data 2</button>
            </a>
            """, unsafe_allow_html=True
        )


st.markdown(
    """
    <div class="problem-box1" >
        <h3>Hasil Analisa</h3>
    </div>""",
    unsafe_allow_html=True
)
if score is not None:
    col10, col11, col12 = st.columns([1, 5, 1])
    with col11:

        def get_kategori_umkm(omset_rata2):
            if omset_rata2 <= 500_000_000:
                return "UMKM Kecil"
            else:
                return "UMKM Menengah"

        # Inisialisasi list untuk menyimpan data summary per cluster
        cluster_summary = []

        # Loop setiap cluster unik
        for cl in sorted(dataset_ori_clean['cluster'].unique()):
            data_cl = dataset_ori_clean[dataset_ori_clean['cluster'] == cl]

            omset_min = int(data_cl['omset'].min())
            omset_max = int(data_cl['omset'].max())
            tenaga_perempuan_min = int(data_cl['tenaga_kerja_perempuan'].min())
            tenaga_perempuan_max = int(data_cl['tenaga_kerja_perempuan'].max())
            tenaga_laki_min = int(data_cl['tenaga_kerja_laki_laki'].min())
            tenaga_laki_max = int(data_cl['tenaga_kerja_laki_laki'].max())
            aset_min = int(data_cl['aset'].min())
            aset_max = int(data_cl['aset'].max())
            kap_prod_min = int(data_cl['kapasitas_produksi'].min())
            kap_prod_max = int(data_cl['kapasitas_produksi'].max())
            tahun_min = int(data_cl['tahun_berdiri'].min())
            tahun_max = int(data_cl['tahun_berdiri'].max())
            biaya_karyawan_min = int(data_cl['biaya_karyawan'].min())
            biaya_karyawan_max = int(data_cl['biaya_karyawan'].max())
            jumlah_pelanggan_min = int(data_cl['jumlah_pelanggan'].min())
            jumlah_pelanggan_max = int(data_cl['jumlah_pelanggan'].max())

            tenaga_kerja_perempuan_mean = data_cl['tenaga_kerja_perempuan'].mean()
            tenaga_kerja_laki_mean = data_cl['tenaga_kerja_laki_laki'].mean()
            omset_mean = data_cl['omset'].mean()
            aset_mean = data_cl['aset'].mean()
            kapasitas_produksi_mean = data_cl['kapasitas_produksi'].mean()
            biaya_karyawan_mean = data_cl['biaya_karyawan'].mean()
            laba_mean = data_cl['laba'].mean()
            
            # Format angka pakai format lokal
            omset_range = f"{omset_min:,.0f} – {omset_max:,.0f}".replace(",", ".")
            aset_range = f"{aset_min:,.0f} – {aset_max:,.0f}".replace(",", ".")
            
            kap_prod_range = f"{kap_prod_min:,.0f} – {kap_prod_max:,.0f}".replace(",", ".")
            tenaga_laki_range = f"{tenaga_laki_min:,.0f} – {tenaga_laki_max:,.0f}".replace(",", ".")
            tenaga_perempuan_range = f"{tenaga_perempuan_min:,.0f} – {tenaga_perempuan_max:,.0f}".replace(",", ".")
            jumlah_pelanggan_range = f"{jumlah_pelanggan_min:,.0f} – {jumlah_pelanggan_max:,.0f}".replace(",", ".")
            biaya_karyawan_range = f"{biaya_karyawan_min:,.0f} – {biaya_karyawan_max:,.0f}".replace(",", ".")
            

            kategori = get_kategori_umkm(omset_mean)

            cluster_summary.append({
                "Cluster": f"Cluster {cl}",
                "omset Rata-rata": round(omset_mean, 2),
                "Omset Range": omset_range,
                "Aset Rata-rata": round(aset_mean, 2),
                "Aset Range": aset_range,
                "tenaga_kerja_laki Rata-rata": round(tenaga_kerja_laki_mean, 2),
                "tenaga kerja Range": tenaga_laki_range,

                "Kategori UMKM": kategori
            })
        # Buat dataframe ringkasan
        df_summary = pd.DataFrame(cluster_summary)
        # Tampilkan di UI
        st.dataframe(df_summary, use_container_width=True)
if score is None:
    col10, col11, col12 = st.columns([1, 5, 1])
    with col11:
        st.badge("error", color="green")




# =========================== DATA PREPARATON =================================

st.markdown(
    """
    <div style="height : 10px">

    </div>


""",unsafe_allow_html=True
)

