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


max_pad_len = find_max_pad_len(audio_files)
print("Maximum padding length:", max_pad_len)

def extract_features(file_paths, max_pad_len=max_pad_len):
    features = []
    for file_path in file_paths:
        audio, sample_rate = librosa.load(file_path, sr=44100, res_type='kaiser_fast')
        n_fft = min(2048, 2**int(np.ceil(np.log2(len(audio)))))
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:
            mfccs = mfccs[:, :max_pad_len]
        else:
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        features.append(mfccs.flatten())
    return np.array(features)

audio_files_directory = current_directory
file_extension = '.wav'

train = pd.read_csv('data/train.csv')

def generate_file_path(row, is_test_data, train_directory, test_directory, extension):
    directory = test_directory if is_test_data else train_directory
    file_path = os.path.join(directory, str(row['ID']) + extension)
    return file_path

train['file_paths'] = train.apply(generate_file_path, args=(False, train_directory, test_directory, '.wav'), axis=1)
test['file_paths'] = test.apply(generate_file_path, args=(True, train_directory, test_directory, '.wav'), axis=1)

print(train.head())
print(test.head())

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

X_train = extract_features(train['file_paths'], 751)
X_test = extract_features(test['file_paths'], 751)

le = LabelEncoder()
y_train = to_categorical(le.fit_transform(train['Class']))

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
from tensorflow.keras.layers import BatchNormalization
model = Sequential([
    Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(2),
    BatchNormalization(),
    Conv1D(128, kernel_size=3, activation='relu'),
    MaxPooling1D(2),
    BatchNormalization(),
    GlobalAveragePooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=32, verbose=1)

# Convert predictions back to class labels
# Make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = le.inverse_transform(np.argmax(predictions, axis=1))

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({'ID': test['ID'], 'Class': predicted_labels})

# Save the DataFrame to a CSV file
predictions_df.to_csv('data/finalpredictionsagain.csv', index=False)

print("Predictions saved to predictions.csv")

