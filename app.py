import json
import streamlit as st
import torch
import numpy as np
import librosa
import soundfile as sf
import os
from app_utils import classify, recommend, AST_recommend
from model_definitions import VGG16, Resnet
from transformers import ASTFeatureExtractor, ASTForAudioClassification
from scipy.io import wavfile
from ConvertDataset import get_frames, window_Hamming, get_spectrogram, getFilters
import numpy.fft as fft
from scipy.fftpack import dct

if "trigger_classify" not in st.session_state:
    st.session_state["trigger_classify"] = False

label_map = None
with open("label_map.json", "r") as f:
    label_map = json.load(f)
inv_label_map = {v: k for k, v in label_map.items()}

st.sidebar.title("Выберите модель")
model_choice = st.sidebar.radio("Модель", ["VGG", "Resnet", "AST"])
st.title("Классификатор музыкальных жанров и формирование рекомендаций")

uploaded_file = st.file_uploader("Загрузите .wav файл", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    if st.button("Классифицировать аудио"):
        st.session_state["trigger_classify"] = True

if st.session_state["trigger_classify"] and uploaded_file:
    top3_probs, top3_labels = [], []

    if model_choice in ["VGG", "Resnet"]:
        model = VGG16() if model_choice == "VGG" else Resnet()
        model.eval()

        sr, data = wavfile.read("temp.wav")
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        duration = 30 - 0.025
        assert sr == 22050
        sample_num = int(sr*duration)
        if len(data) < 2 * sample_num:
            raise Exception("Audio too short! 1 minute or more required.")
        
        sections = [data[i:i+sample_num] for i in range(0, len(data), sample_num)]
        last_pad_len = sample_num - len(sections[-1])
        last_empty_ratio = 0.0
        if last_pad_len < sample_num:
            sections[-1] = np.pad(sections[-1], (0, last_pad_len), mode="constant")
            last_empty_ratio = last_pad_len / sample_num
        print(len(sections[0]), len(sections[1]), len(sections[-1]))
        sections_count = len(sections)
        print(sections_count)
        modifier_first, modifier_last = 1.0, 1.0
        if len(sections) > 2:
            modifier_first, modifier_last = 0.75, 0.75
        melSpectrograms, mfccs = [], []
        modifiers = [modifier_first, modifier_last]
        
        for data in sections:
            preemp = 0.66
            preemp_data = np.append(data[0], data[1:] - preemp * data[:-1])

            framed_data = get_frames(preemp_data, sr, wlen=0.025, wstep=0.010)
            flen = framed_data.shape[1]
            filter = window_Hamming(flen)
            windowed_data = framed_data * filter

            specter = np.abs(fft.rfft(windowed_data))
            spectrogram = get_spectrogram(specter)
            filterMatrix = getFilters()
            normFilterMatrix = filterMatrix.T / filterMatrix.sum(axis=1)
            melSpectrogram = np.transpose(normFilterMatrix).dot(np.transpose(spectrogram))
            numToKeep = 16
            mfcc = dct(melSpectrogram.T, type=2, axis=1, norm='ortho')[:, : numToKeep]

            melSpectrograms.append(melSpectrogram)
            mfccs.append(mfcc.T)

        mfccs.pop() # убираем mfcc последней секции
        probs, pred_label, top3_ids = classify(mfccs, model_choice, modifiers=modifiers)
        total_probs = sum(probs)
        norm_probs = total_probs / total_probs.sum()
        top3_probs = [norm_probs[i] for i in top3_ids]
        top3_labels = [inv_label_map[i] for i in top3_ids]

    elif model_choice == "AST":
        model = ASTForAudioClassification.from_pretrained("./saved_ast_model")
        feature_extractor = ASTFeatureExtractor.from_pretrained("./saved_ast_model")
        model.eval()
        
        sr, audio_input = wavfile.read("temp.wav")
        if len(audio_input.shape) > 1:
            audio_input = np.mean(audio_input, axis=1)
        audio_input = librosa.resample(audio_input.astype(float), orig_sr=sr, target_sr=feature_extractor.sampling_rate)
        sr = feature_extractor.sampling_rate

        inputs = feature_extractor(audio_input, sampling_rate=sr, return_tensors="pt")
        
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        top3_probs, top3_ids = torch.topk(probs, k=3, dim=-1)
        
        top3_probs = top3_probs.squeeze().tolist()
        top3_ids = top3_ids.squeeze().tolist()
        top3_labels = [inv_label_map[i] for i in top3_ids]


    st.subheader("Топ-3 предсказания класса:")
    cols = st.columns(3)
    for i, (label, prob) in enumerate(zip(top3_labels, top3_probs)):
        with cols[i]:
            st.metric(label=label, value=f"{prob:.2%}")
    
    st.subheader("Рекомендации")
    for i, label in enumerate(top3_labels):
        rec_button_key = f"rec_button_{i}_{label}"
        if st.button(f"Получить рекомендации по жанру {label}", key=rec_button_key):
            st.markdown(f"**Топ рекомендаций для `{label}`**")

            if model_choice == "AST":
                recs = AST_recommend(audio_input, sr, label, top_n=3, debug=False)
            else:
                recs = recommend(mfccs, label, mode="mfcc", top_n=3, debug=False)

            for j, (path, dist) in enumerate(recs, start=1):
                audio_label = os.path.basename(path)
                st.markdown(f"Рекомендация №{j}: {audio_label}; Расстояние: `{dist:.4f}`")
                try:
                    audio_bytes = open(path, "rb").read()
                    st.audio(audio_bytes, format="audio/wav")
                except Exception as e:
                    st.warning(f"Couldn't load audio: {e}")