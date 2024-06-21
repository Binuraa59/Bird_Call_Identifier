# Imrting libraries 
import os
import librosa
import numpy as np

# loading dataset
dataset= "C:/Users/Binura Perera/Downloads/dataset_root"
birds = ["cinereous_tinamou", "great_tinamou", "brown_tinamu"]

# Preprocessing and Feature Extraction Functions
def load_audio_files(bird_folder):
    files = [f for f in os.listdir(bird_folder) if f.endswith('.mp3')]
    audio_files = []
    for file in files:
        file_path = os.path.join(bird_folder, file)
        y, sr = librosa.load(file_path, sr=16000)
        audio_files.append((y, sr))
    return audio_files

def add_noise(data, noise_factor=0.005):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    return augmented_data

def shift_time(data, shift_max=0.2):
    shift = np.random.randint(int(len(data) * shift_max))
    augmented_data = np.roll(data, shift)
    return augmented_data

def extract_features(y, sr, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs.T, axis=0)

# Process all bird folders
features = []
labels = []

for label, bird in enumerate(birds):
    bird_folder = os.path.join(dataset, bird)
    audio_files = load_audio_files(bird_folder)
    
    for y, sr in audio_files:
        # Apply augmentation
        y_noisy = add_noise(y)
        y_shifted = shift_time(y)
        
        # Extract features
        mfcc_features = extract_features(y, sr)
        mfcc_features_noisy = extract_features(y_noisy, sr)
        mfcc_features_shifted = extract_features(y_shifted, sr)
        
        # Append to lists
        features.append(mfcc_features)
        labels.append(label)
        
        features.append(mfcc_features_noisy)
        labels.append(label)
        
        features.append(mfcc_features_shifted)
        labels.append(label)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Save the extracted features and labels
np.save('features.npy', X)
np.save('labels.npy', y)
