import os
import random
import librosa
from sklearn.discriminant_analysis import StandardScaler
from model_definitions import LeNet, VGG16, Resnet
import torch
import numpy as np
from skimage.io import imread
from scipy.spatial.distance import cosine
from fastdtw import fastdtw

def classify(data, modelname, modifiers=[1.0, 1.0], min_confidence=0.5, mode="mfcc", debug=False):
    probabilities = []
    predictions = []
    num_segments = len(data)
    for i, segment in enumerate(data):
        probs, pred = classify_step(segment, modelname, i, mode, debug)
        maxconf = np.max(probs)
        if maxconf < min_confidence:
            continue
        weight = 1.0
        if i == 0:
            weight *= modifiers[0]
        if i == num_segments - 1:
            weight *= modifiers[1]
            if len(modifiers) == 3 and mode == "ms":
                weight *= modifiers[2]
        wprobs = weight * probs
        probabilities.append(wprobs)
        predictions.append(pred)
    # if debug:
    #     print(probabilities)
    if len(probabilities) == 0:
        print("No high-confidence segments found.")
        probs, pred = classify_step(data[1], modelname, 0, mode, debug)
        fallback_top3 = [pred] + [i for i in range(10) if i != pred][:2]
        return [probs], pred, fallback_top3
    # totalprobs = sum(probabilities)
    # normprobs = totalprobs / totalprobs.sum()
    # prediction = int(torch.argmax(torch.tensor(normprobs)))
    total_probs = sum(probabilities)
    norm_probs = total_probs / total_probs.sum()
    top3 = np.argsort(norm_probs)[::-1][:3]
    prediction = int(top3[0])

    if debug:
        print(f"Top-3 predicted classes: {top3.tolist()}")
        print(f"Final predicted class: {prediction}")

    return probabilities, prediction, top3.tolist()

def classify_step(data, modelname, segnum, mode="mfcc", debug=False):
    model = None
    if modelname == "LeNet":
        model = LeNet()
        model.load_state_dict(torch.load("LeNet_best.pt"))
        model.eval()
    elif modelname == "VGG":
        model = VGG16()
        model.load_state_dict(torch.load("VGG_best.pt"))
        model.eval()
    elif modelname == "Resnet":
        model = Resnet()
        model.load_state_dict(torch.load("Resnet_best.pt"))
        model.eval()
    else:
        raise Exception("Incorrect modelname provided")
    
    if mode == "mfcc":
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        tensor_data = torch.tensor(scaled_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor_data)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
    elif mode == "ms":
        if isinstance(data, np.ndarray):
            data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        else:
            data_tensor = data.unsqueeze(0)

        with torch.no_grad():
            output = model(data_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1)
    # if debug:
    #     print(f"Probabilities for segment №{segnum}: {probabilities.numpy().flatten()}")
    #     print(f"Predicted class for segment №{segnum}: {prediction.item()}")

    return probabilities.numpy().flatten(), prediction.item()

def recommend(data, predicted_label, mode="mfcc", top_n=5, debug=False):
    modepath = ""
    if mode == "mfcc":
        modepath = "MFCCs"
    elif mode == "ms":
        modepath = "Mel Spectrograms"
    dataset_path = f"./Dataset/{modepath}/{predicted_label}"
    genre_path = f"./Dataset/genres/{predicted_label}"
    candidates = []

    data = np.concatenate(data, axis=1)
    if mode == "mfcc":
        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-9)

    for root, _, files in os.walk(dataset_path):
        files = [f for f in files if f.endswith('.png')]
        random.shuffle(files)
        num_samples = max(1, len(files) // 5)
        sampled_files = files[:num_samples]
        for fname in sampled_files:
            fpath = os.path.join(root, fname)
            sample = imread(fpath)
            if mode == "mfcc":
                sample = sample.T
                sample = (sample - sample.mean(axis=1, keepdims=True)) / (sample.std(axis=1, keepdims=True) + 1e-9)
            distance, dtw_path = fastdtw(data.T, sample, dist=cosine)
            norm_distance = distance / len(dtw_path)
            audio_fname = fname.replace('_', '.').rsplit('.', 1)[0] + ".wav"
            audio_path = os.path.join(genre_path, audio_fname)
            audio_path = os.path.normpath(audio_path)
            audio_path = os.path.join(genre_path, audio_fname)
            candidates.append((audio_path, norm_distance))

    candidates.sort(key=lambda x: x[1])
    return candidates[:top_n]

def AST_recommend(audio_input, sampling_rate, predicted_label, top_n=5, debug=False):
    dataset_path = f"./Dataset/genres/{predicted_label}"

    def extract_mfcc(y, sr, n_mfcc=16):
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return mfcc.T

    input_mfcc = extract_mfcc(audio_input, sampling_rate)

    candidates = []

    for root, _, files in os.walk(dataset_path):
        wav_files = [f for f in files if f.endswith('.wav')]
        random.shuffle(wav_files)
        num_samples = max(1, len(wav_files) // 5)
        sampled_files = wav_files[:num_samples]

        for fname in sampled_files:
            fpath = os.path.join(root, fname)
            y, sr = librosa.load(fpath, sr=sampling_rate)  # resample
            candidate_mfcc = extract_mfcc(y, sr)
            distance, dtw_path = fastdtw(input_mfcc, candidate_mfcc, dist=cosine)
            norm_distance = distance / len(dtw_path)

            candidates.append((fpath, norm_distance))

    candidates.sort(key=lambda x: x[1])
    return candidates[:top_n]