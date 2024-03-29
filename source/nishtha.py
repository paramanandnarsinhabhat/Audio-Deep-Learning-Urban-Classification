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

# Apply the function to each row in the DataFrame to create the new column for training data
train['file_paths'] = train.apply(generate_file_path, args=(False, train_directory, test_directory, '.wav'), axis=1)

# Apply the function to each row in the DataFrame to create the new column for test data
test['file_paths'] = test.apply(generate_file_path, args=(True, train_directory, test_directory, '.wav'), axis=1)

# Verify the new columns
print(train.head())
print(test.head())


print('Control comes here ')
import tensorflow as tf
from tensorflow.python.tools import module_util as _module_util
from numpy import to_categorical

print("Import successful")

# Extract features from training and test data
X_train = extract_features(train['file_paths'],751)
X_test = extract_features(test['file_paths'],751)

# Encode labels
le = LabelEncoder()
y_train = to_categorical(le.fit_transform(train['Class']))

def preprocess_audio(file_paths, max_pad_len):
    features = []
    for file_path in file_paths:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=None)
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        
        # Pad or truncate MFCCs to max_pad_len
        if mfccs.shape[1] < max_pad_len:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_pad_len - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        
        # Reshape MFCCs to add channel dimension
        mfccs = mfccs.reshape((*mfccs.shape, 1))
        
        features.append(mfccs)
    
    return np.array(features)

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define model architecture
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(le.classes_), activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=1)

# Convert predictions back to class labels
# Make predictions on the test data
predictions = model.predict(X_test)
predicted_labels = le.inverse_transform(np.argmax(predictions, axis=1))

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({'ID': test['ID'], 'Class': predicted_labels})

# Save the DataFrame to a CSV file
predictions_df.to_csv('data/finalpredictions.csv', index=False)

print("Predictions saved to predictions.csv")

