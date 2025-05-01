import os
import numpy as np
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Paths
class_1_folder = '/home/mio/Documents/code/Cat_Dog_Detection/classificationdataset/cat'  # e.g., 'dataset/class1'
class_2_folder = '/home/mio/Documents/code/Cat_Dog_Detection/classificationdataset/cat'  # e.g., 'dataset/class2'

# Parameters
sr = 22050  # Sampling rate
n_mfcc = 20  # Number of MFCCs
n_mels = 128  # Number of Mel bands

# Feature extraction function
def extract_features(file_path, sr=22050, max_len=200):
    y, _ = librosa.load(file_path, sr=sr)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Pad or truncate to fixed size
    if mfcc.shape[1] < max_len:
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        mel_db = np.pad(mel_db, pad_width=((0,0),(0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
        mel_db = mel_db[:, :max_len]
    
    features = np.concatenate((mfcc.flatten(), mel_db.flatten()))
    return features

# Loading data
X = []
y = []

# Load class 1
for file_name in os.listdir(class_1_folder):
    if file_name.endswith('.wav'):
        file_path = os.path.join(class_1_folder, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(0)  # Label 0 for class 1

# Load class 2
for file_name in os.listdir(class_2_folder):
    if file_name.endswith('.wav'):
        file_path = os.path.join(class_2_folder, file_name)
        features = extract_features(file_path)
        X.append(features)
        y.append(1)  # Label 1 for class 2

X = np.array(X)
y = np.array(y)

print(f"Features shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Models to test
models = {
    "Support Vector Machine": SVC(kernel='rbf', probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

# Training and evaluating
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy for {name}: {acc:.4f}")
    print(classification_report(y_test, y_pred))
