import os
import numpy as np
import librosa
from skimage.transform import resize
from joblib import load

def preprocess_audio(file_path, size=32, n_mfcc=6, n_mels=20):
    # Load audio (match your training SR)
    audio, sr = librosa.load(file_path, sr=44000)
    
    # --- Extract features (EXACTLY like during training) ---
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # --- Resize and concatenate (critical step!) ---
    mfccs = resize(np.expand_dims(mfccs, axis=-1), (size, size))
    mel_spec_db = resize(np.expand_dims(mel_spec_db, axis=-1), (size, size))
    combined = np.concatenate((mfccs, mel_spec_db), axis=-1)
    
    # --- Flatten to match training data shape ---
    return combined.flatten()  # Shape = (size*size*2,)

# Load model (use your best-performing params from filename)
model_path = "/home/mio/Documents/code/Cat_Dog_Detection/rf_50_mf6_ml20_ims_32_acc_ 0.959.pkl"
rf_model = load(model_path)

# Prediction loop
main_dir = "/home/mio/Documents/code/Cat_Dog_Detection/catdetdataset"
class_mapping = {0: "cat", 1: "other"}

for folder in os.listdir(main_dir):
    if os.path.isdir(os.path.join(main_dir, folder)):
        for file in os.listdir(os.path.join(main_dir, folder)):
            if file.endswith('.wav'):
                file_path = os.path.join(main_dir, folder, file)
                
                # Preprocess (match training: size=32, n_mfcc=6, n_mels=20)
                features = preprocess_audio(file_path, size=32, n_mfcc=6, n_mels=20)
                features = features.reshape(1, -1)  # RF expects (1, n_features)
                
                # Predict
                pred_class = rf_model.predict(features)[0]
                proba = rf_model.predict_proba(features)[0]
                confidence = proba.max()
                
                print(f"File: {file}")
                print(f"Predicted: {class_mapping[pred_class]}, Actual: {folder}")
                print(f"Confidence: {confidence:.2%}\n")