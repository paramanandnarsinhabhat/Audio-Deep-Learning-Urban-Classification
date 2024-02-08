
# Urban Sound Classification

## Overview
This project aims to classify urban sounds using machine learning techniques. The classification is performed using a Convolutional Neural Network (CNN) built with the TensorFlow and Keras libraries.

## Dataset
The dataset consists of labeled sound excerpts from urban environments. The training and testing data are contained within 'train.csv' and 'test.csv' respectively.

## Features
The model uses Mel-frequency cepstral coefficients (MFCCs) extracted from the audio data as features for classification.

## Requirements
The project requires the following Python libraries:
- pandas
- librosa
- tensorflow
- scikit-learn
- numpy

To install the required libraries, run `pip install -r requirements.txt`.

## Files and Directories
- `data/train.csv`: The training dataset.
- `data/test.csv`: The testing dataset.
- `data/Train/`: Directory containing training audio files.
- `data/Test/`: Directory containing testing audio files.
- `data/finalpredictions.csv`: Output file with predictions from the model.

## Usage
1. Prepare the data by placing your audio files in the `data/Train/` and `data/Test/` directories.
2. Run the script to train the model and make predictions on the test set.
3. The predictions will be saved to `data/finalpredictions.csv`.

## Model Architecture
The model is a Sequential CNN with the following layers:
- Conv1D with 64 filters
- MaxPooling1D
- BatchNormalization
- Conv1D with 128 filters
- MaxPooling1D
- BatchNormalization
- GlobalAveragePooling1D
- Dense
- Dropout
- Dense (output layer)

## Training
The model is trained using the Adam optimizer with categorical crossentropy as the loss function.

## Output
After training, the model's predictions are saved to 'data/finalpredictions.csv', mapping each audio file to its predicted class.



## License
MIT License
```

