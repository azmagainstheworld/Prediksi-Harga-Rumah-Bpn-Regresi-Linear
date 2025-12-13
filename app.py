import pickle
import pandas as pd
from flask import Flask, render_template, request

# Load model + std_residual
model_data = pickle.load(open("model/model.pkl", "rb"))
model = model_data["model"]
std_residual = model_data["std_residual"]

app = Flask(__name__)

# List kecamatan
kecamatan_list = [
    'Balikpapan Barat',
    'Balikpapan Kota',
    'Balikpapan Selatan',
    'Balikpapan Tengah',
    'Balikpapan Timur',
    'Balikpapan Utara'
]

# Nama kolom fitur sesuai training
feature_names = [
    'Luas Tanah (m²)',
    'Luas Bangunan (m²)',
    'Kamar Tidur',
    'Kamar Mandi',
    'Daerah_Balikpapan Barat',
    'Daerah_Balikpapan Kota',
    'Daerah_Balikpapan Selatan',
    'Daerah_Balikpapan Tengah',
    'Daerah_Balikpapan Timur',
    'Daerah_Balikpapan Utara'
]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data input user
    kamar_tidur = int(request.form['kamar_tidur'])
    kamar_mandi = int(request.form['kamar_mandi'])
    luas_tanah = float(request.form['luas_tanah'])
    luas_bangunan = float(request.form['luas_bangunan'])
    kecamatan = request.form['kecamatan']

    # One-hot encoding kecamatan
    daerah = [0] * len(kecamatan_list)
    daerah[kecamatan_list.index(kecamatan)] = 1

    # Buat DataFrame fitur dengan nama kolom yang sama seperti training
    fitur = pd.DataFrame([[ 
        luas_tanah,
        luas_bangunan,
        kamar_tidur,
        kamar_mandi
    ] + daerah], columns=feature_names)

    # Prediksi harga
    prediksi = model.predict(fitur)[0]

    # Hitung range
    bawah = prediksi - std_residual
    atas = prediksi + std_residual
    range_harga = f"Rp{bawah:,.0f} – Rp{atas:,.0f}"

    # Kirim semua ke template, termasuk detail input user
    return render_template(
        'index.html',
        kamar_tidur=kamar_tidur,
        kamar_mandi=kamar_mandi,
        luas_tanah=luas_tanah,
        luas_bangunan=luas_bangunan,
        kecamatan=kecamatan,
        range_harga=range_harga
    )

if __name__ == '__main__':
    app.run(debug=True)
