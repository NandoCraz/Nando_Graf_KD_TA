"""
Konversi Data Angket Siswa (bg1 - bg25)
Mengubah nama kolom bg1 - bg25 ke pertanyaan deskriptif dan
mengonversi kode jawaban (A1, B1, C1, dst.) menjadi teks jawaban riil.

Input  : rapor-publik-asesmen-nasional-2024-peserta-didik-2024-indonesia-3.xlsx (atau .csv)
Output : data_for_analisis.csv
"""

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np


# ======================================================================
# 1. KAMUS NAMA KOLOM (bg1 s/d bg25)
# ======================================================================
MAP_NAMA_KOLOM = {
    "bg1": "Apakah kamu perempuan atau laki-laki?",
    "bg2": "Ada berapa keluarga yang tinggal bersama di rumahmu?",
    "bg3": "Apa jenjang sekolah tertinggi yang diselesaikan ayahmu?",
    "bg4": "Apa jenjang sekolah tertinggi yang diselesaikan ibumu?",
    "bg5": "Apa pekerjaan ayahmu?",
    "bg6": "Apa pekerjaan ibumu?",
    "bg7": "Kepemilikan Meja belajar di rumah",
    "bg8": "Kamar tidur untuk dirimu sendiri (tidak berbagi dengan saudara atau orang lain)",
    "bg9": "Tempat belajar yang tenang di rumah",
    "bg10": "Listrik dan lampu untuk belajar di rumah pada malam hari",
    "bg11": "Komputer, laptop, atau tablet yang bisa kamu pakai untuk keperluan belajar atau tugas sekolah",
    "bg12": "Kepemilikan perangkat lunak (software atau program komputer) pendidikan",
    "bg13": "Sambungan ke internet di rumah",
    "bg14": "Kepemilikan alat baca buku digital (seperti Kindle, Kobo, dll)",
    "bg15": "Kepemilikan buku-buku fiksi/sastra seperti novel, kumpulan cerpen, atau kumpulan puisi",
    "bg16": "Kepemilikan buku-buku tentang sains atau teknologi",
    "bg17": "Kepemilikan buku-buku tentang seni, musik, atau desain",
    "bg18": "Kepemilikan buku-buku tentang bahasa, termasuk kamus",
    "bg19": "Kepemilikan buku-buku yang bisa membantumu mengerjakan tugas-tugas sekolah",
    "bg20": "Kepemilikan televisi layar datar",
    "bg21": "Kepemilikan mobil",
    "bg22": "Kepemilikan Pendingin Ruangan (AC)",
    "bg23": "Kepemilikan telepon seluler (handphone) dengan akses internet (misal, smartphone)",
    "bg24": "Kepemilikan komputer tablet (Tab/iPad)",
    "bg25": "Kpemilikan karya-karya seni (misal: lukisan/patung seni)",
}


# ======================================================================
# 2. KAMUS OPSI JAWABAN PER KOLOM
# ======================================================================
MAP_PEKERJAAN = {
    "A1": "Tidak bekerja (termasuk jika sudah meninggal), mengurus rumah tangga (termasuk ibu rumah tangga), atau kerja serabutan (tidak jelas)",
    "B1": "Pensiunan dari karyawan pemerintah, perusahaan swasta, militer, atau polisi",
    "C1": "Anggota militer (TNI/tentara) atau polisi (Polri)",
    "D1": "Perwira militer (TNI/tentara) atau polisi (Polri)",
    "E1": "Buruh pabrik/tani/tambang, asisten/pembantu rumah tangga, pengemudi/sopir (termasuk taksi online), petugas keamanan, petugas kebersihan, dan karyawan usaha kecil (pelayan toko, mini market, dll)",
    "F1": "Bekerja mandiri dengan keterampilan khusus (seperti montir bengkel, penjahit, tukang kayu, bertani di ladang sendiri, nelayan dengan kapal sendiri, dll.)",
    "G1": "Tenaga pemasaran (seperti sales, marketing)",
    "H1": "Pekerja administratif (seperti sekretaris, tata usaha, receptionist, front office, costumer service) di kantor pemerintah (termasuk PNS), di perusahaan BUMN, dan di perusahaan swasta.",
    "I1": "Manajer atau pemimpin/kepala/pejabat di kantor pemerintah, perusahaan BUMN, atau perusahaan swasta.",
    "J1": "Profesional (seperti dosen, guru, dokter, bidan, pengacara, jaksa, insinyur/engineer, musisi, nakoda kapal, pilot, dll.)",
    "K1": "Anggota legislatif (DPR atau DPRD) atau pemimpin pemerintah (seperti gubernur, walikota, bupati)",
    "L1": "Pemilik usaha kecil (seperti pemilik toko kecil, bengkel, warung, pemilik rental mobil, dll.)",
    "M1": "Pemilik perusahaan besar (seperti pabrik besar, restoran besar, toko dengan banyak cabang, dll.",
    "N1": "Tidak tahu.",
}

MAP_PENDIDIKAN = {
    "A1": "SD",
    "B1": "SMP",
    "C1": "SMA/SMK",
    "D1": "Diploma (D1, D2, atau D3)",
    "E1": "Sarjana (S1)",
    "F1": "Master (S2)",
    "G1": "Doktor (S3)",
    "H1": "Tidak Tahu",
}

MAP_NILAI_OPSI = {
    # bg1: Jenis kelamin (mendukung A1/B1 maupun format A1/A2)
    "bg1": {
        "A1": "Laki-laki",
        "B1": "Perempuan",
        "A2": "Perempuan",
    },

    # bg2: Keluarga tinggal bersama
    "bg2": {
        "A1": "Hanya keluarga intiku sendiri (orang tua dan saudara kandungku saja)",
        "B1": "Keluargaku bersama satu keluarga lain",
        "C1": "Keluargaku bersama dua keluarga lain",
        "D1": "Keluargaku bersama tiga atau lebih keluarga lain",
    },

    # bg3 & bg4: Jenjang pendidikan orang tua
    "bg3": MAP_PENDIDIKAN,
    "bg4": MAP_PENDIDIKAN,

    # bg5 & bg6: Pekerjaan orang tua
    "bg5": MAP_PEKERJAAN,
    "bg6": MAP_PEKERJAAN,

    # bg7 s/d bg14: Fasilitas belajar di rumah (opsi A=Tidak ada, B=Ada)
    "bg7":  {"A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg8":  {"A2": "Tidak ada", "B2": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg9":  {"A3": "Tidak ada", "B3": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg10": {"A4": "Tidak ada", "B4": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg11": {"A5": "Tidak ada", "B5": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg12": {"A6": "Tidak ada", "B6": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg13": {"A7": "Tidak ada", "B7": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},
    "bg14": {"A8": "Tidak ada", "B8": "Ada", "A1": "Tidak ada", "B1": "Ada", "A": "Tidak ada", "B": "Ada"},

    # bg15 s/d bg19: Kepemilikan buku (A=Tidak ada, B=1-10, C=11-25, D=26-100, E=>100)
    "bg15": {
        "A1": "Tidak ada", "B1": "Kira-kira 1-10 buku", "C1": "Kira-kira 11-25 buku",
        "D1": "Kira-kira 26-100 buku", "E1": "Kira-kira lebih dari 100 buku",
    },
    "bg16": {
        "A2": "Tidak ada", "B2": "Kira-kira 1-10 buku", "C2": "Kira-kira 11-25 buku",
        "D2": "Kira-kira 26-100 buku", "E2": "Kira-kira lebih dari 100 buku",
        "A1": "Tidak ada", "B1": "Kira-kira 1-10 buku", "C1": "Kira-kira 11-25 buku",
        "D1": "Kira-kira 26-100 buku", "E1": "Kira-kira lebih dari 100 buku",
    },
    "bg17": {
        "A3": "Tidak ada", "B3": "Kira-kira 1-10 buku", "C3": "Kira-kira 11-25 buku",
        "D3": "Kira-kira 26-100 buku", "E3": "Kira-kira lebih dari 100 buku",
        "A1": "Tidak ada", "B1": "Kira-kira 1-10 buku", "C1": "Kira-kira 11-25 buku",
        "D1": "Kira-kira 26-100 buku", "E1": "Kira-kira lebih dari 100 buku",
    },
    "bg18": {
        "A4": "Tidak ada", "B4": "Kira-kira 1-10 buku", "C4": "Kira-kira 11-25 buku",
        "D4": "Kira-kira 26-100 buku", "E4": "Kira-kira lebih dari 100 buku",
        "A1": "Tidak ada", "B1": "Kira-kira 1-10 buku", "C1": "Kira-kira 11-25 buku",
        "D1": "Kira-kira 26-100 buku", "E1": "Kira-kira lebih dari 100 buku",
    },
    "bg19": {
        "A5": "Tidak ada", "B5": "Kira-kira 1-10 buku", "C5": "Kira-kira 11-25 buku",
        "D5": "Kira-kira 26-100 buku", "E5": "Kira-kira lebih dari 100 buku",
        "A1": "Tidak ada", "B1": "Kira-kira 1-10 buku", "C1": "Kira-kira 11-25 buku",
        "D1": "Kira-kira 26-100 buku", "E1": "Kira-kira lebih dari 100 buku",
    },

    # bg20 s/d bg25: Kepemilikan barang/fasilitas (A=Tidak ada, B=Satu, C=Dua, D=Tiga atau lebih)
    "bg20": {
        "A1": "Tidak ada", "B1": "Satu", "C1": "Dua", "D1": "Tiga atau lebih",
    },
    "bg21": {
        "A2": "Tidak ada", "B2": "Satu", "C2": "Dua", "D2": "Tiga atau lebih",
        "A1": "Tidak ada", "B1": "Satu", "C1": "Dua", "D1": "Tiga atau lebih",
    },
    "bg22": {
        "A3": "Tidak ada", "B3": "Satu", "C3": "Dua", "D3": "Tiga atau lebih",
        "A1": "Tidak ada", "B1": "Satu", "C1": "Dua", "D1": "Tiga atau lebih",
    },
    "bg23": {
        "A4": "Tidak ada", "B4": "Satu", "C4": "Dua", "D4": "Tiga atau lebih",
        "A1": "Tidak ada", "B1": "Satu", "C1": "Dua", "D1": "Tiga atau lebih",
    },
    "bg24": {
        "A5": "Tidak ada", "B5": "Satu", "C5": "Dua", "D5": "Tiga atau lebih",
        "A1": "Tidak ada", "B1": "Satu", "C1": "Dua", "D1": "Tiga atau lebih",
    },
    "bg25": {
        "A6": "Tidak ada", "B6": "Satu", "C6": "Dua", "D6": "Tiga atau lebih",
        "A1": "Tidak ada", "B1": "Satu", "C1": "Dua", "D1": "Tiga atau lebih",
    },
}


# ======================================================================
# 3. FUNGSI UTAMA PEMROSESAN
# ======================================================================
def proses_konversi(
    file_input="rapor-publik-asesmen-nasional-2024-peserta-didik-2024-indonesia-3.xlsx",
    file_output="data_for_analisis.csv",
    sheet_name="rapor_publik"
):
    start_time = time.time()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Memulai proses konversi...")

    # Cek ketersediaan file input
    if not os.path.exists(file_input):
        # Coba fallback ke ekstensi .csv jika ada
        file_csv_fallback = file_input.replace(".xlsx", ".csv")
        if os.path.exists(file_csv_fallback):
            file_input = file_csv_fallback
        else:
            raise FileNotFoundError(f"File input '{file_input}' tidak ditemukan di direktori saat ini.")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Membaca file: {file_input}")
    print("Catatan: Jika membaca file Excel besar (~340 MB), proses memuat ke memori membutuhkan waktu beberapa menit...")

    # Baca file input
    if file_input.endswith(".xlsx") or file_input.endswith(".xls"):
        try:
            df = pd.read_excel(file_input, sheet_name=sheet_name)
        except Exception:
            # Fallback membaca sheet pertama jika nama sheet berbeda
            print(f"Sheet '{sheet_name}' tidak ditemukan, membaca sheet pertama...")
            df = pd.read_excel(file_input)
    else:
        df = pd.read_csv(file_input, low_memory=False)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Data berhasil dimuat. Shape data awal: {df.shape}")

    # Konversi nilai jawaban bg1 - bg25 sesuai kamus
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Mengonversi nilai kode jawaban bg1 - bg25 ke teks...")
    for col, mapping in MAP_NILAI_OPSI.items():
        if col in df.columns:
            # Petakan kode ke teks deskriptif, tetap pertahankan NaN jika data kosong
            df[col] = df[col].apply(lambda x: mapping.get(str(x).strip(), x) if pd.notna(x) else np.nan)

    # Ubah nama kolom bg1 - bg25 menjadi nama pertanyaan
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Mengubah nama kolom bg1 - bg25 ke pertanyaan deskriptif...")
    kolom_rename = {col: MAP_NAMA_KOLOM[col] for col in MAP_NAMA_KOLOM if col in df.columns}
    df = df.rename(columns=kolom_rename)

    # Simpan hasil ke data_for_analisis.csv
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Menyimpan hasil ke file: {file_output}")
    df.to_csv(file_output, index=False)

    elapsed_time = time.time() - start_time
    menit = int(elapsed_time // 60)
    detik = elapsed_time % 60

    print("=" * 60)
    print(f"Proses Selesai Sukses!")
    print(f"File tersimpan      : {file_output}")
    print(f"Dimensi data akhir  : {df.shape[0]} baris x {df.shape[1]} kolom")
    print(f"Total waktu proses  : {menit} menit {detik:.2f} detik")
    print("=" * 60)

    return df


if __name__ == "__main__":
    proses_konversi()
