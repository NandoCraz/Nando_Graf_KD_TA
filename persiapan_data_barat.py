import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

def cleaning(path_data_mentah="data_for_analisis.csv", path_output="data_barat.csv"):
    # --- 1. Baca data mentah ---
    data = pd.read_csv(path_data_mentah)
    print("Shape data mentah:", data.shape)

    # --- 2. Agregasi kolom nilai AN ---
    # nilai_AN = rata-rata dari 14 subskor literasi & numerasi
    kolom_nilai = [
        'LIT', 'LIT_INF', 'LIT_SAS', 'LIT_L1', 'LIT_L2', 'LIT_L3',
        'NUM', 'NUM_ALJ', 'NUM_GEO', 'NUM_BIL', 'NUM_DAT', 'NUM_L1', 'NUM_L2', 'NUM_L3'
    ]
    data['nilai_AN'] = data[kolom_nilai].mean(axis=1)

    # --- 3. Drop kolom yang tidak terpakai ---
    kolom_di_drop = [
        'SES_siswa',
        'LIT', 'LIT_INF', 'LIT_SAS', 'LIT_L1', 'LIT_L2', 'LIT_L3',
        'NUM', 'NUM_ALJ', 'NUM_GEO', 'NUM_BIL', 'NUM_DAT', 'NUM_L1', 'NUM_L2', 'NUM_L3',
        'AAKN', 'AAMN', 'AAPR', 'ABKR', 'ABMK', 'AMKA', 'AMKB', 'AMKG', 'AMTA', 'AMTB',
        'GRKP', 'GRPB', 'GRPR', 'KGKI', 'KGMB', 'KMPE', 'KMPR', 'KRBB', 'KRMI', 'KRMK',
        'NKAE', 'NKPI', 'NKRE',
        'ACH', 'AIN', 'AKC', 'BCP', 'BUL', 'CAF', 'DIB', 'DIC', 'DIE', 'ECP', 'ENP',
        'EQC', 'EQR', 'ESA', 'ESV', 'ISB', 'ITB', 'KKG', 'OCC', 'PBR', 'PBU', 'PCP',
        'PKG', 'PLIT', 'PMU', 'PNUM', 'POT', 'PSA', 'PSV', 'RKG', 'RPI', 'SAF', 'SBU', 'SKG',
        'SMK_GRMAG', 'SMK_GTINS', 'SMK_INDS', 'SMK_PBEL', 'SMK_PIM', 'SMK_SARPR', 'SMK_TEFA',
        'TAS', 'TOC', 'TOR', 'TSC', 'WEL',
        'SES_sekolah', 'IASP',
        'pendidikan_sederajat', 'jenis_sek', 'sts_sek', 'kd_kokab'
    ]
    data = data.drop(columns=kolom_di_drop)
    print("Shape setelah reduksi fitur:", data.shape)

    # --- 4. Tangani data hilang (listwise deletion) ---
    print("\nJumlah NaN sebelum dibersihkan:")
    print(data.isna().sum())

    data.dropna(inplace=True)

    print("\nJumlah NaN setelah dibersihkan:")
    print(data.isna().sum())
    print(f"\nTotal data bersih (nasional): {data['kd_siswa_an'].count()}")

    # --- 5. Filter Indonesia Barat ---
    data_barat = data[data['wilayah_bagian'] == 'Indonesia Barat'].copy()
    print(f"Data Indonesia Barat: {data_barat['kd_siswa_an'].count()}")

    # --- 6. Bentuk label kelas dari nilai_AN ---
    nilai_AN_min = data_barat['nilai_AN'].min()
    nilai_AN_max = data_barat['nilai_AN'].max()
    interval = (nilai_AN_max - nilai_AN_min) / 3

    def klasifikasi_nilai_AN(x):
        if x < nilai_AN_min + interval:
            return 'rendah'
        elif x < nilai_AN_min + 2 * interval:
            return 'cukup'
        else:
            return 'tinggi'

    data_barat['kelas_nilai_AN'] = data_barat['nilai_AN'].apply(klasifikasi_nilai_AN)

    # --- 7. Finalisasi: buang kolom nilai_AN mentah, simpan ---
    data_barat = data_barat.drop('nilai_AN', axis=1)
    data_barat.to_csv(path_output)
    print(f"\nData final disimpan ke: {path_output}, shape: {data_barat.shape}")

    return data_barat

def encoding(path_data_barat="data_barat.csv"):
    data_for_encod = pd.read_csv(path_data_barat)
    data_for_encod = data_for_encod.drop('Unnamed: 0', axis=1)

    kolom_nominal = [
        'kurikulum', 'daerah_khusus', 'jenis_wilayah', 'status_wilayah',
        'ketersediaan_internet', 'ketersediaan_listrik',
        'Apakah kamu perempuan atau laki-laki?',
        'Apa pekerjaan ayahmu?', 'Apa pekerjaan ibumu?',
        'Kamar tidur untuk dirimu sendiri (tidak berbagi dengan saudara atau orang lain)',
        'Kepemilikan Meja belajar di rumah',
        'Tempat belajar yang tenang di rumah',
        'Komputer, laptop, atau tablet yang bisa kamu pakai untuk keperluan belajar atau tugas sekolah',
        'Listrik dan lampu untuk belajar di rumah pada malam hari',
        'Kepemilikan perangkat lunak (software atau program komputer) pendidikan',
        'Sambungan ke internet di rumah',
        'Kepemilikan alat baca buku digital (seperti Kindle, Kobo, dll)',
    ]

    kolom_keluarga = ['Ada berapa keluarga yang tinggal bersama di rumahmu?']
    kolom_nilai = ['kelas_nilai_AN']

    kolom_pendidikan = [
        'Apa jenjang sekolah tertinggi yang diselesaikan ayahmu?',
        'Apa jenjang sekolah tertinggi yang diselesaikan ibumu?'
    ]

    kolom_buku = [
        'Kepemilikan buku-buku fiksi/sastra seperti novel, kumpulan cerpen, atau kumpulan puisi',
        'Kepemilikan buku-buku tentang sains atau teknologi',
        'Kepemilikan buku-buku tentang seni, musik, atau desain',
        'Kepemilikan buku-buku tentang bahasa, termasuk kamus',
        'Kepemilikan buku-buku yang bisa membantumu mengerjakan tugas-tugas sekolah'
    ]

    kolom_gadget = [
        'Kepemilikan televisi layar datar',
        'Kepemilikan mobil',
        'Kepemilikan Pendingin Ruangan (AC)',
        'Kepemilikan telepon seluler (handphone) dengan akses internet (misal, smartphone)',
        'Kepemilikan komputer tablet (Tab/iPad)',
        'Kpemilikan karya-karya seni (misal: lukisan/patung seni)'
    ]

    mapping_pendidikan = {
        'Tidak Tahu': 0, 'SD': 1, 'SMP': 2, 'SMA/SMK': 3,
        'Diploma (D1, D2, atau D3)': 4, 'Sarjana (S1)': 5,
        'Master (S2)': 6, 'Doktor (S3)': 7
    }

    mapping_buku = {
        'Tidak ada': 0, 'Kira-kira 1-10 buku': 1, 'Kira-kira 11-25 buku': 2,
        'Kira-kira 26-100 buku': 3, 'Kira-kira lebih dari 100 buku': 4
    }

    mapping_gadget = {
        'Tidak ada': 0, 'Satu': 1, 'Dua': 2, 'Tiga atau lebih': 3
    }

    mapping_keluarga = {
        'Hanya keluarga intiku sendiri (orang tua dan saudara kandungku saja)': 0,
        'Keluargaku bersama tiga atau lebih keluarga lain': 3,
        'Keluargaku bersama dua keluarga lain': 2,
        'Keluargaku bersama satu keluarga lain': 1
    }

    mapping_nilai = {'rendah': 0, 'cukup': 1, 'tinggi': 2}

    for col in kolom_pendidikan:
        data_for_encod[col] = data_for_encod[col].map(mapping_pendidikan)
    for col in kolom_buku:
        data_for_encod[col] = data_for_encod[col].map(mapping_buku)
    for col in kolom_gadget:
        data_for_encod[col] = data_for_encod[col].map(mapping_gadget)
    for col in kolom_keluarga:
        data_for_encod[col] = data_for_encod[col].map(mapping_keluarga)
    for col in kolom_nilai:
        data_for_encod[col] = data_for_encod[col].map(mapping_nilai)

    data_encod = pd.get_dummies(data_for_encod, columns=kolom_nominal, dtype=int)
    print("Shape setelah encoding:", data_encod.shape)

    return data_encod

def bangun_graf(data_encod):
    target_col = 'kelas_nilai_AN'

    kolom_wilayah = [
        'daerah_khusus_Tidak', 'daerah_khusus_Ya',
        'jenis_wilayah_KABUPATEN', 'jenis_wilayah_KOTA',
        'status_wilayah_RURAL', 'status_wilayah_URBAN'
    ]

    kolom_sekolah = [
        'proporsi_pendidik_min_s1', 'proporsi_pendidik_sertifikasi',
        'jumlah_peserta_didik', 'jumlah_pendidik', 'rasio_pendidik_peserta_didik',
        'jumlah_r_kelas', 'jumlah_komp_milik', 'jumlah_perpus', 'jumlah_rombel',
        'jumlah_siswa_rombel', 'jumlah_siswa_penerima_PIP', 'rasio_siswa_penerima_PIP',
        'kurikulum_Kurikulum 2013', 'kurikulum_Kurikulum Merdeka',
        'ketersediaan_internet_Ada', 'ketersediaan_internet_Tidak Ada',
        'ketersediaan_listrik_Ada'
    ]

    kolom_ortu = [
        'Ada berapa keluarga yang tinggal bersama di rumahmu?',
        'Apa jenjang sekolah tertinggi yang diselesaikan ayahmu?',
        'Apa jenjang sekolah tertinggi yang diselesaikan ibumu?',
    ] + [c for c in data_encod.columns if c.startswith('Apa pekerjaan ayahmu?_')] \
      + [c for c in data_encod.columns if c.startswith('Apa pekerjaan ibumu?_')]

    kolom_siswa = [
        'Kepemilikan buku-buku fiksi/sastra seperti novel, kumpulan cerpen, atau kumpulan puisi',
        'Kepemilikan buku-buku tentang sains atau teknologi',
        'Kepemilikan buku-buku tentang seni, musik, atau desain',
        'Kepemilikan buku-buku tentang bahasa, termasuk kamus',
        'Kepemilikan buku-buku yang bisa membantumu mengerjakan tugas-tugas sekolah',
        'Kepemilikan televisi layar datar', 'Kepemilikan mobil',
        'Kepemilikan Pendingin Ruangan (AC)',
        'Kepemilikan telepon seluler (handphone) dengan akses internet (misal, smartphone)',
        'Kepemilikan komputer tablet (Tab/iPad)',
        'Kpemilikan karya-karya seni (misal: lukisan/patung seni)',
    ] + [c for c in data_encod.columns if c.startswith('Apakah kamu perempuan atau laki-laki?_')] \
      + [c for c in data_encod.columns if c.startswith('Kamar tidur untuk dirimu sendiri')] \
      + [c for c in data_encod.columns if c.startswith('Kepemilikan Meja belajar di rumah')] \
      + [c for c in data_encod.columns if c.startswith('Tempat belajar yang tenang di rumah')] \
      + [c for c in data_encod.columns if c.startswith('Komputer, laptop, atau tablet')] \
      + [c for c in data_encod.columns if c.startswith('Listrik dan lampu untuk belajar')] \
      + [c for c in data_encod.columns if c.startswith('Kepemilikan perangkat lunak')] \
      + [c for c in data_encod.columns if c.startswith('Sambungan ke internet di rumah')] \
      + [c for c in data_encod.columns if c.startswith('Kepemilikan alat baca buku digital')]

    # --- Buat ID unik untuk simpul Wilayah dan Profil Orang Tua ---
    data_encod['id_wilayah'] = data_encod[kolom_wilayah].astype(str).agg('-'.join, axis=1)
    data_encod['id_ortu'] = data_encod[kolom_ortu].astype(str).agg('-'.join, axis=1)

    data_encod['siswa_idx'] = data_encod['kd_siswa_an'].astype('category').cat.codes
    data_encod['sekolah_idx'] = data_encod['kd_sekolah'].astype('category').cat.codes
    data_encod['wilayah_idx'] = data_encod['id_wilayah'].astype('category').cat.codes
    data_encod['ortu_idx'] = data_encod['id_ortu'].astype('category').cat.codes

    def extract_node_features(df_source, idx_col, feature_cols):
        """Menghapus duplikasi baris per objek, mengubah ke Float Tensor."""
        df_unique = df_source[[idx_col] + feature_cols].drop_duplicates(idx_col).sort_values(idx_col)
        return torch.tensor(df_unique[feature_cols].values, dtype=torch.float)

    X_wilayah = extract_node_features(data_encod, 'wilayah_idx', kolom_wilayah)
    X_sekolah = extract_node_features(data_encod, 'sekolah_idx', kolom_sekolah)
    X_ortu = extract_node_features(data_encod, 'ortu_idx', kolom_ortu)
    X_siswa = extract_node_features(data_encod, 'siswa_idx', kolom_siswa)

    df_target = data_encod[['siswa_idx', target_col]].drop_duplicates('siswa_idx').sort_values('siswa_idx')
    Y_siswa = torch.tensor(df_target[target_col].values, dtype=torch.long)

    graf = HeteroData()
    graf['siswa'].x = X_siswa
    graf['siswa'].y = Y_siswa
    graf['sekolah'].x = X_sekolah
    graf['wilayah'].x = X_wilayah
    graf['profil_ortu'].x = X_ortu

    graf['siswa', 'belajar_di', 'sekolah'].edge_index = torch.tensor([
        data_encod['siswa_idx'].values, data_encod['sekolah_idx'].values
    ], dtype=torch.long)

    df_sek_wil = data_encod[['sekolah_idx', 'wilayah_idx']].drop_duplicates()
    graf['sekolah', 'berlokasi_di', 'wilayah'].edge_index = torch.tensor([
        df_sek_wil['sekolah_idx'].values, df_sek_wil['wilayah_idx'].values
    ], dtype=torch.long)

    graf['siswa', 'memiliki', 'profil_ortu'].edge_index = torch.tensor([
        data_encod['siswa_idx'].values, data_encod['ortu_idx'].values
    ], dtype=torch.long)

    graf = T.ToUndirected()(graf)
    print(graf)

    return graf

def split_data(graf, num_val=0.1, num_test=0.1):
    transform_split = T.RandomNodeSplit(split='train_rest', num_val=num_val, num_test=num_test)
    graf = transform_split(graf)

    print(f"Jumlah Train : {graf['siswa'].train_mask.sum().item()}")
    print(f"Jumlah Val   : {graf['siswa'].val_mask.sum().item()}")
    print(f"Jumlah Test  : {graf['siswa'].test_mask.sum().item()}")

    return graf

if __name__ == "__main__":
    data_barat = cleaning(
        path_data_mentah="data_for_analisis.csv",
        path_output="data_barat.csv"
    )

    data_encod = encoding("data_barat.csv")
    graf = bangun_graf(data_encod)
    graf = split_data(graf)

    torch.save(graf, "graf_barat.pt")
    print("\nGraf disimpan ke: graf_barat.pt")
