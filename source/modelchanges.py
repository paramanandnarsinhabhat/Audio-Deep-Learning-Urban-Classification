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

