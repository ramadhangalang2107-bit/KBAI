import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Prediksi Produktivitas Sawit", layout="wide")


@st.cache_resource
def muat_model():
    return joblib.load("model_sawit.joblib")


@st.cache_data
def muat_riwayat():
    return pd.read_csv("riwayat.csv", parse_dates=["date"]).sort_values(["Block_ID", "date"])


paket = muat_model()
model, fitur, bm = paket["model"], paket["fitur"], paket["blok_mean"]
riw = muat_riwayat()

st.title("Prediksi Produktivitas Kelapa Sawit Bulanan")
st.caption("Random Forest (target selisih terhadap bulan lalu) dengan fitur lag temporal dan tuning Optuna")

tab_prediksi, tab_model = st.tabs(["Prediksi", "Evaluasi model"])

with tab_prediksi:
    blok = st.selectbox("Pilih blok", sorted(riw["Block_ID"].unique()))
    h = riw[riw["Block_ID"] == blok]
    y = h["Productivity_t_Ha"].values
    terakhir = h["date"].iloc[-1]
    depan = terakhir + pd.DateOffset(months=1)

    st.write(f"Data terakhir: **{terakhir:%B %Y}**. Prediksi untuk: **{depan:%B %Y}**")
    st.line_chart(h.set_index("date")["Productivity_t_Ha"])

    if len(y) < 12:
        st.warning(
            "Blok ini baru menghasilkan kurang dari 12 bulan, jadi fitur lag 12 bulan "
            "belum bisa dihitung. Prediksi tidak tersedia."
        )
    elif st.button("Prediksi"):
        m = depan.month
        lag_1 = y[-1]
        baris = {
            "lag_1": lag_1, "lag_3": y[-3], "lag_12": y[-12],
            "rm_3": y[-3:].mean(), "rm_12": y[-12:].mean(),
            "month_sin": np.sin(2 * np.pi * m / 12),
            "month_cos": np.cos(2 * np.pi * m / 12),
            "Plant_Age_Years": h["Plant_Age_Years"].iloc[-1],
            "diff_1_3": lag_1 - y[-3:].mean(),
            "diff_1_12": lag_1 - y[-12:].mean(),
            "blok_mean": bm[blok], "blok_gap": lag_1 - bm[blok],
        }
        for f in fitur:
            if f.startswith("Seed_Type_"):
                baris[f] = int(f == "Seed_Type_" + h["Seed_Type"].iloc[-1])
        X = pd.DataFrame([baris])[fitur]
        pred = lag_1 + model.predict(X)[0]

        c1, c2 = st.columns(2)
        c1.metric("Prediksi produktivitas (ton/ha)", f"{pred:.3f}",
                  delta=f"{pred - lag_1:+.3f} vs bulan lalu")
        c2.metric("Pembanding naif: bulan lalu (ton/ha)", f"{lag_1:.3f}")

    st.subheader("Faktor yang paling berpengaruh")
    st.bar_chart(pd.Series(model.feature_importances_, index=fitur).sort_values())

with tab_model:
    st.subheader("Perbandingan pada data uji (2023-2024)")
    st.dataframe(pd.DataFrame(
        {"MAE": [0.280, 0.201, 0.165],
         "RMSE": [0.358, 0.275, 0.242],
         "R2": [0.476, 0.691, 0.760]},
        index=["Naif (bulan lalu)", "XGBoost + Optuna", "Random Forest + Optuna"],
    ))
    st.caption(
        "Pembagian data berdasarkan waktu. Tuning memakai validasi berjenjang 2019-2022; "
        "data uji dipakai satu kali. Pada validasi berjenjang, MAE Random Forest (0,254) "
        "dan XGBoost (0,251) hampir setara; perbedaan jelas baru terlihat di data uji."
    )
    st.caption("Sumber data: Mimboro, Mendeley Data, DOI 10.17632/93k3733p6x.2 (CC BY 4.0).")
