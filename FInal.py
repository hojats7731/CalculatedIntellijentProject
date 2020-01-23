import os
import pickle
import tensorflow as tf
from mido import MidiFile
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.utils import to_categorical
import statistics 
from statistics import mode

files = []
songs = {}
genres = {}

def gettingFiles(files_dir):
    for r, d, f in os.walk(files_dir):
        for file in f:
            if '.mid' in file:
                files.append(os.path.join(r, file))


def getNoteRange():
    for file in files:
        vector = []
        mid = MidiFile(file)
        for i, track in enumerate(mid.tracks):
            for msg in track:
                if hasattr(msg, 'note'):
                    if msg.velocity != 0:
                        vector.append(msg.note)
        # print(vector)
        name = file[file.find("\\") + 1:]
        print(name)
        songs[name] = vector
    print(len(songs))

def getGenre():
    file = open("../MIDI_Genres/trainLabels.txt", "r");
    for line in file:
        index = line.find(",")
        name = line[0:index]
        genre = line[index + 1:]
        genres[name] = genre
        print(name + " : " + genre)

def avg(song):
    return sum(song) / len(song)

def common(song):
    return mode(song)

def commonDistance(song):
    dist = [song[i + 1] - song[i] for i in range(len(song) - 1)]
    return common(dist)

def avgDistance(song):
    dist = [song[i + 1] - song[i] for i in range(len(song) - 1)]
    return avg(dist)

gettingFiles('../MIDI_Genres/MIDI_Genres/train_set')
getNoteRange()
getGenre()

for name in songs:
    song = songs[name]
    genre = genres[name]
    
    features = []
    features.append(max(song))
    features.append(avg(song))
    features.append(common(song))
    features.append(commonDistance(song))
    features.append(avgDistance(song))

    x = numpy.array([features])
    y = numpy.array(genre)
    y_binary = to_categorical(y, num_classes=10)

    model = Sequential()
    model.add(Dense(100, input_dim=5, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, y_binary, epochs=10)

    with open("final.1.0.pickle", "wb") as f:
        pickle.dump(model, f)

