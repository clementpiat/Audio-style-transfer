import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

from tensorflow.python.lib.io import file_io
import soundfile
from scipy import signal
"""
@description: Method to split a song into multiple songs using overlapping windows
"""
def splitsongs(X, y, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)

"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

def to_spectrogram(songs,fs):
    # Transformation function
    melspec = lambda x: ((abs(signal.stft(x, fs=fs, nperseg=512)[2])**2)*2500)[:128,:,np.newaxis]
    
    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    
    
    return np.array(list(tsongs))


"""
@description: Read audio files from folder
"""
def read_data(src_dir, genres, song_samples,  
    n_fft = 1024, hop_length = 512, debug = True):
    # Empty array of dicts with the processed features from all files
    arr_specs = []
    arr_genres = []
    print(src_dir)
    print(genres.items())
    # Read files from the folders
    filenames = file_io.get_matching_files(src_dir+"genres/**/*.*")
    labels = [genres[x.split('/')[-2]] for x in filenames]
    #sprint(filenames)
    for i in range(len(filenames)):
        ##folder = src_dir +'genres/'+ x
       
        ##for root, subdirs, files in os.walk(folder):
        ##for file in filenames:
            # Read the audio file
            ##file_name = folder + "/" + file
        
        tmp = file_io.FileIO(filenames[i], "rb")
        signal, sr = soundfile.read(tmp)
        signal = signal[:song_samples]
        
        # Debug process
        if debug:
            print("Reading file: {}".format(tmp))
        
        # Convert to dataset of spectograms/melspectograms
        signals, y = splitsongs(signal, labels[i])
        
        # Convert to "spec" representation
        #specs = to_melspectrogram(signals, n_fft, hop_length)
        specs = to_spectrogram(signals,sr)
        # Save files
        #print(np.array(specs).shape)
        arr_genres.extend(y)
        arr_specs.extend(specs)
        #print(np.array(arr_specs).shape)

               
    return np.array(arr_specs), to_categorical(np.array(arr_genres))
