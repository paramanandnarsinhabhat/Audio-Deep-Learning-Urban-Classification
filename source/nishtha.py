import pandas as pd
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import os
import numpy as np

# Load training and test data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#Check the first 5 rows
print(train.head())

print(test.head())

#Function to find the maximum padding length needed

def find_max_pad_len(file_paths):
    max_len = 0
    for file_path in file_paths:
        audio, sample_rate = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        if mfccs.shape[1] > max_len:
            max_len = mfccs.shape[1]
    return max_len

import os
# Specify the directories containing your audio files
train_directory = 'data/Train/'
test_directory = 'data/Test/'

# Flag to control which directory to use
is_test_data = False  # Set to True if working with test data, False for training data

# Determine which directory to use based on is_test_data flag
current_directory = test_directory if is_test_data else train_directory

# List to store file paths
audio_files = []

# Loop through the files in the current directory
for filename in os.listdir(current_directory):
    if filename.endswith(".wav") or filename.endswith(".mp3"):  # Add any other audio formats you need
        # Create the full file path and add it to the list
        file_path = os.path.join(current_directory, filename)
        audio_files.append(file_path)
# print(audio_files)
        

max_pad_len = find_max_pad_len(audio_files)

print("Maximum padding length:", max_pad_len)

def extract_features(file_paths, max_pad_len=max_pad_len):
    features = []  # List to hold the extracted features for all files
    
    for file_path in file_paths:
        # Load the audio file
        audio, sample_rate = librosa.load(file_path, sr=44100, res_type='kaiser_fast')
        
        # Dynamically adjust n_fft if audio is too short
        n_fft = min(2048, 2**int(np.ceil(np.log2(len(audio)))))
        
        # Extract MFCCs with the adjusted n_fft
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40, n_fft=n_fft)
        
        # Pad or truncate the MFCCs to max_pad_len
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width < 0:  # Truncate
            mfccs = mfccs[:, :max_pad_len]
        else:  # Pad with zeros
            mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        
        # Flatten and add to the list of features
        features.append(mfccs.flatten())
    
    return np.array(features)

# Specify the directory containing your audio files
audio_files_directory = current_directory  #Update this path
file_extension = '.wav' # Update this as needed

#Reading the training dataset
train = pd.read_csv('data/train.csv')


# Define a function to generate the file path
def generate_file_path(row, is_test_data, train_directory, test_directory, extension):
    # Determine the directory based on is_test_data
    directory = test_directory if is_test_data else train_directory
    
    # Construct the file path. Assumes the Id column contains the filename without the extension
    file_path = os.path.join(directory, str(row['ID']) + extension)
    return file_path

print(file_path)



