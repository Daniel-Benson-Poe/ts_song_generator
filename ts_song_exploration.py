# Import the dependencies
import numpy as np
import pandas as pd
import sys 
from keras.models import Sequential
from keras.layers import LSTM, Activation, Flatten, Dropout, Dense, Embedding, TimeDistributed, CuDNNLSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os

# Load dataset
dataset = pd.read_csv("taylor_swift_lyrics.csv", encoding="latin1")
# print(dataset.head())
# print(dataset.describe())

# Concat each line to get each song together in its own string
def processFirstLine(lyrics, songID, songName, row):
    lyrics.append(row['lyric'] + '\n')
    songID.append(row['year']*100 + row['track_n'])
    songName.append(row['track_title'])
    return lyrics, songID, songName

# Define empty lists for lyrics, songID, songName
lyrics = []
songID = []
songName = []

# songNumber indicates the song number in the dataset
songNumber = 1

# i indicates song number
i = 0
isFirstLine = True

# Iterate through every lyrics line and join them together for each song independentaly
for index, row in dataset.iterrows():
    if(songNumber == row['track_n']):
        if(isFirstLine):
            lyrics,songID,songName = processFirstLine(lyrics, songID, songName, row)
            isFirstLine = False
        else:
            # if we still in same song, keep joining lyrics lines
            lyrics[i] += row['lyric'] + '\n'
    # when it's done joining song's lyrics lines, go to next song
    else:
        lyrics,songID,songName = processFirstLine(lyrics,songID,songName,row)
        songNumber = row['track_n']
        i+= 1

# Define new dataframe to save above results
lyrics_data = pd.DataFrame({'songID':songID, 'songName':songName, 'lyrics':lyrics})
# print(lyrics_data.head())
# print(lyrics_data.tail())

# PreProcessing
lyricsText = ''
for listitem in lyrics:
    lyricsText += listitem

# convert lyrics to lowercase
raw_text = lyricsText
raw_text = raw_text.lower()

# Mapping characters
chars = sorted(list(set(raw_text)))
int_chars = dict((i,c) for i, c in enumerate(chars))
chars_int = dict((i,c) for c, i in enumerate(chars))

# Get number chars and vocab in our text
n_chars = len(raw_text)
n_vocab = len(chars)

print(f"Total Characters: {n_chars}") # number of all characters in taylor_swift_lyrics
print(f"Total Vocab: {n_vocab}") # number of unique characters

# Make samples and labels
# process dataset
seq_len = 100
data_X = []
data_y = []

for i in range(0, n_chars - seq_len, 1):
    # input sequence(used as samples)
    seq_in = raw_text[i:i+seq_len]
    # output sequence (used as target)
    seq_out = raw_text[i + seq_len]
    # Store samples in data_X
    data_X.append([chars_int[char] for char in seq_in])
    # Store targets in data_y
    data_y.append(chars_int[seq_out])
n_patterns = len(data_X)
print(f"Total Patterns: {n_patterns}")

# Prepare samples and labels
# Reshape X to be suitable to go into LSTM RNN
X = np.reshape(data_X, (n_patterns, seq_len, 1))
# Normalizing input data
X = X/ float(n_vocab)
# One hot encode output targets
y = np_utils.to_categorical(data_y)

# Create Model
LSTM_layer_num = 4 # number of LSTM layers
layer_size = [256,256,256,256] # number of nodes in each layer

# Define sequential model
model = Sequential()

model.add(CuDNNLSTM(layer_size[0], input_shape=(X.shape[1], X.shape[2]), return_sequences=True))

# Add hidden layers
for i in range(1, LSTM_layer_num):
    model.add(CuDNNLSTM(layer_size[i], return_sequences=True))

# Flatter data from last hidden layer to input it to output layer
model.add(Flatten())

# Add output layer
model.add(Dense(y.shape[1]))
model.add(Activation('softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam')

# Print summary of model
print(model.summary())

# Configure checkpoint
checkpoint_name = 'Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
checkpoint = ModelCheckpoint(checkpoint_name, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# Fit model
model_params = {'epochs':30,
                'batch_size':128,
                'callbacks':callbacks_list,
                'verbose':1,
                'validation_split':0.2,
                'validation_data':None,
                'shuffle':True,
                'initial_epoch':0,
                'steps_per_epoch':None,
                'validation_steps':None}

model.fit(X,
          y,
          epochs = model_params['epochs'],
           batch_size = model_params['batch_size'],
           callbacks= model_params['callbacks'],
           verbose = model_params['verbose'],
           validation_split = model_params['validation_split'],
           validation_data = model_params['validation_data'],
           shuffle = model_params['shuffle'],
           initial_epoch = model_params['initial_epoch'],
           steps_per_epoch = model_params['steps_per_epoch'],
           validation_steps = model_params['validation_steps'])

# load weights
weights_file = 'Weights-LSTM-improvement-{epoch:03d}-{loss:.5f}-bigger.hdf5'
model.load_weights(weights_file)
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Generating lyrics
start = np.random.randint(0, len(data_X)-1)
pattern = data_X[start]
print('Seed : ')
print("\"",''.join([int_chars[value] for value in pattern]), "\"\n")

# Number characters wanted to generate
generated_characters = 300

# Generate characters
for i in range(generated_characters):
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x / float(n_vocab)
    prediction = model.predict(x,verbose = 0)
    index = np.argmax(prediction)
    result = int_chars[index]
    #seq_in = [int_chars[value] for value in pattern]
    sys.stdout.write(result)
    pattern.append(index)
    pattern = pattern[1:len(pattern)]
print('\nDone')
