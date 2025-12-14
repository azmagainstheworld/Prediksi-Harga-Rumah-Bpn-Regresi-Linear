import pandas as pd
import numpy as np 
import joblib 
from flask import Flask, render_template, request
import os

# KONSTANTA DARI PELATIHAN MODEL
# MAE digunakan untuk rentang harga: Prediksi ± MAE (Rp 473,813,412)
MAE_FINAL = 473_813_412 

# Daftar 12 kolom fitur 
FEATURE_COLUMNS = [
    'Luas Tanah (m²)', 
    'Luas Bangunan (m²)', 
    'Kamar Tidur', 
    'Kamar Mandi',
    'Daerah_Balikpapan Barat', 
    'Daerah_Balikpapan Kota', 
    'Daerah_Balikpapan Selatan', 
    'Daerah_Balikpapan Tengah',
    'Daerah_Balikpapan Timur', 
    'Daerah_Balikpapan Utara', 
    'harga_per_m2_tanah', 
    'rasio_lb_lt'         
]

# JALUR FILE
MODEL_PATH = "model/model_regresi_linear_harga_rumah.pkl"
COLUMNS_PATH = "model/kolom_fitur_model.pkl"

# LOAD MODEL
try:
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(f) 
    print(f"Model berhasil dimuat dari {MODEL_PATH}")
except FileNotFoundError:
    print(f"ERROR: File model tidak ditemukan di {MODEL_PATH}. Harap periksa kembali jalur Anda.")
    exit()
except Exception as e:
    print(f"ERROR saat memuat model dari {MODEL_PATH}: {e}")
    print("Petunjuk: Coba pastikan versi scikit-learn saat training dan deployment sama.")
    exit()

# LOAD FEATURE COLUMNS
try:
    with open(COLUMNS_PATH, "rb") as f:
        loaded_feature_names = joblib.load(f)
        # Verifikasi panjang harus 12
        if len(loaded_feature_names) == 12: 
            FEATURE_COLUMNS = loaded_feature_names
            print("Kolom fitur berhasil dimuat dan terverifikasi (12 kolom).")
        else:
            print(f"⚠️ Peringatan: Kolom fitur dari file {COLUMNS_PATH} tidak sesuai (bukan 12 kolom). Menggunakan list hardcoded.")

except FileNotFoundError:
    print(f"ERROR: File kolom fitur tidak ditemukan di {COLUMNS_PATH}. Menggunakan list hardcoded.")


app = Flask(__name__)

kecamatan_list = [
    'Balikpapan Kota',
    'Balikpapan Selatan',
    'Balikpapan Tengah',
    'Balikpapan Timur',
    'Balikpapan Utara',
    'Balikpapan Barat' 
]

# FUNGSI UNTUK MENYIAPKAN INPUT FITUR
def create_input_dataframe_final_model(kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan):
    
    # 1. Base features & Feature Engineering
    input_data = {
        'Luas Tanah (m²)': luas_tanah,
        'Luas Bangunan (m²)': luas_bangunan,
        'Kamar Tidur': kamar_tidur,
        'Kamar Mandi': kamar_mandi,
        'rasio_lb_lt': luas_bangunan / luas_tanah,
        # 'harga_per_m2_tanah' diisi 0 karena hanya fitur saat training
        'harga_per_m2_tanah': 0.0 
    }

    for col in FEATURE_COLUMNS:
        if col.startswith('Daerah_'):
            input_data[col] = 0
            
    daerah_col = f"Daerah_{kecamatan}"
    if daerah_col in FEATURE_COLUMNS:
         input_data[daerah_col] = 1

    # Buat DataFrame dan pastikan urutan kolom sesuai FEATURE_COLUMNS
    fitur = pd.DataFrame([input_data])
    
    return fitur[FEATURE_COLUMNS]

@app.route('/')
def index():
    return render_template('index.html', kecamatan_list=kecamatan_list)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        kamar_tidur = int(request.form['kamar_tidur'])
        kamar_mandi = int(request.form['kamar_mandi'])
        luas_tanah = float(request.form['luas_tanah'])
        luas_bangunan = float(request.form['luas_bangunan'])
        kecamatan = request.form['kecamatan']
        
        # Validasi batas wajar data training
        if not (1 <= kamar_tidur <= 10 and 1 <= kamar_mandi <= 10 and 30 <= luas_tanah <= 2000 and 20 <= luas_bangunan <= 1500):
             raise ValueError("Input di luar batas wajar data training (cth: KT/KM 1-10, LT 30-2000, LB 20-1500).")

    except ValueError as ve:
        error_msg = str(ve) if str(ve) else "Input harus berupa angka yang valid."
        return render_template('index.html', error=error_msg, kecamatan_list=kecamatan_list,
                               kamar_tidur=kamar_tidur, kamar_mandi=kamar_mandi, 
                               luas_tanah=luas_tanah, luas_bangunan=luas_bangunan, 
                               kecamatan=kecamatan)
    except Exception as e:
        return render_template('index.html', error=f"Terjadi kesalahan: {e}", kecamatan_list=kecamatan_list)

    # 1. Susun fitur & Lakukan Feature Engineering
    try:
        fitur = create_input_dataframe_final_model(
            kamar_tidur, kamar_mandi, luas_tanah, luas_bangunan, kecamatan
        )
    except Exception as e:
         return render_template('index.html', error=f"Error saat menyiapkan fitur: {e}", kecamatan_list=kecamatan_list)

    # 2. Prediksi harga (skala log) & Konversi ke skala asli
    try:
        # Prediksi menghasilkan ln(Harga)
        log_prediksi = model.predict(fitur)[0]
        
        # Kembalikan ke skala harga asli (Rp)
        prediksi = np.exp(log_prediksi)
    except Exception as e:
        return render_template('index.html', error=f"Error saat prediksi. Cek konsistensi fitur model: {e}", kecamatan_list=kecamatan_list)

    # 3. RANGE HARGA menggunakan MAE
    bawah = max(0, prediksi - MAE_FINAL) 
    atas = prediksi + MAE_FINAL
    
    # Formatting
    prediksi_tengah = f"Rp{prediksi:,.0f}"
    range_harga = f"Rp{bawah:,.0f} – Rp{atas:,.0f}"

    return render_template(
        'index.html',
        kamar_tidur=kamar_tidur,
        kamar_mandi=kamar_mandi,
        luas_tanah=luas_tanah,
        luas_bangunan=luas_bangunan,
        kecamatan=kecamatan,
        range_harga=range_harga,
        prediksi_tengah=prediksi_tengah,
        kecamatan_list=kecamatan_list 
    )

if __name__ == '__main__':
    app.run(debug=True)