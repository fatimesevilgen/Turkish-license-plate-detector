import streamlit as st
import cv2
import numpy as np
from predict import predict
import torch

torch.classes.__path__ = []

# Başlık
st.title("Araç Plaka Tespit Uygulaması")

# Kullanıcıdan görüntü yükleme
uploaded_file = st.file_uploader("Bir fotoğraf yükleyin", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Yüklenen dosyayı OpenCV formatına dönüştür
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Tahmin fonksiyonunu çalıştır
    annotated = predict(image)

    # Sonucu göster (BGR -> RGB dönüşümü)
    result = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    st.image(result, caption="Tespit Edilen Plaka", use_column_width=True)

    # İsteğe bağlı: Sonucu indirme
    # convert image to PNG for download
    _, buffer = cv2.imencode('.png', annotated)
    st.download_button(
        label="Sonucu İndir",
        data=buffer.tobytes(),
        file_name="annotated.png",
        mime="image/png"
    )