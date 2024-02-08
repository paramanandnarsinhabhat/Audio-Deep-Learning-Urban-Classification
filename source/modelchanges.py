import pandas as pd
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os
import numpy as np

# Load training and test data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Check the first 5 rows
print(train.head())
print(test.head())


def find_max_pad_len(file_paths):
    max_len = 0
    for file_path in file_paths:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] > max_len:
            max_len = mfccs.shape[1]
    return max_len

#Directory for train and test data
train_directory = 'data/Train/'
test_directory = 'data/Test/'

#Making sure the current directory is test or train
is_test_data = False
current_directory = test_directory if is_test_data else train_directory
audio_files = []


for filename in os.listdir(current_directory):
    if filename.endswith(".wav") or filename.endswith(".mp3"):
        file_path = os.path.join(current_directory, filename)
        audio_files.append(file_path)

