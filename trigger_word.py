
""" Trigger word project:
This program defines and detects a word in an audio stream
This is a similar to trigger word 'Alexa' or 'Hey google'

 """
import numpy as np
from pydub import AudioSegment 
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
import matplotlib.pyplot as plt
import time


#x = graph_spectrogram("./audio_examples/example_train.wav")

_, data = wavfile.read("./audio_examples/example_train.wav")

print("Time steps in audio recording before spectrogram", data[:,0].shape)
#print("Time steps in input after spectrogram", x.shape)

IPython.display.Audio("./raw_data/activates/1.wav")

Tx = 5511 # length of the time vector in the spectrogram
n_freq= 101
Ty = 1375 # GRU output

activates, negatives, backgrounds = load_raw_audio()

print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec c
print("activate[0] len: " + str(len(activates[1])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
print("activate[1] len: " + str(len(activates[4])))     # Different "activate" clips can have different lengths 



def get_random_time_segment(segment_ms):
    segment_start = np.random.randint(low=0,high=10000-segment_ms)
    segment_end = segment_start + segment_ms - 1
    return (segment_start,segment_end)

# check if the segment is overlapping with existing segments

def is_overlapping(new_segment,previous_segments): 
    
    new_start,new_end= new_segment

    for prev_seg_start, prev_seg_end in previous_segments:
        if new_start <= prev_seg_end and new_end >= prev_seg_start:
            return True
    
    return False

# This function inserts an audioclip within the background noise

def insertAudioClip(background,clip,existing_segemnts):
    len_ms = len(clip)
    
    new_segment = get_random_time_segment(len_ms)

    while is_overlapping(new_segment,existing_segemnts):
        new_segment = get_random_time_segment(len_ms)
    
    existing_segemnts.append(new_segment)
    new_background = background.overlay(clip,new_segment[0])

    return new_background, new_segment

# This function inserts 1's in the output vector to identify the
# end of the trigger word. Here Y is the output of the network
# and segments_end_ms is the time from which a series of ones starts
def insertOnes(y,segment_end_ms):
    
    index = int(Ty*segment_end_ms/1e4)

    y[0,index+1:index+51] = 1

    return y




def creatreTrainingExample(background,activates,negatives):
    """ this function creates a training instance 
     it uses a backgournd clip from backgrounds, an activate clip from 
     activates and a negative clip from negatives. All of these clips are
     overlayed into the background clip. 
     """
    
    np.random.seed(int(time.time()))
    y = np.zeros([1,Ty])
    existing_segments = []

    rand_activates = np.random.randint(len(activates),size=4)
    rand_negatives = np.random.randint(len(negatives),size=2)
    print(rand_activates)
     # choose a random activate clip from the list of activates
    for i in rand_activates:
        background, activate_segment = insertAudioClip(background,  activates[i],existing_segments)
        y = insertOnes(y,activate_segment[1])
    
    for i in rand_negatives:
        background, _ = insertAudioClip(background,negatives[i], existing_segments)
    

    background = match_target_amplitude(background, -20.0)

    file_handle = background.export("train" + ".wav", format="wav")
    print("File (train.wav) was saved in your directory.")

    x = graph_spectrogram("train.wav")

    return x,y


#x, y = creatreTrainingExample(backgrounds[0], activates, negatives)
#print(x.shape)
#fig2 = plt.figure(2)
#ax2 = fig2.gca()

#ax2.plot(y[0])

# Full training procedure 
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape,LSTM
from keras.optimizers import Adam

# Load preprocessed training set examples
X = np.load("./XY_train/X.npy")
Y = np.load("./XY_train/Y.npy")

# Load preprocessed dev set examples
X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")



#This section defines the trigger word model
def model(input_shape):
    X_input = Input(shape=input_shape)

    # Convolutional layer
    X = Conv1D(196,kernel_size=15,strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    # 1st GRU layer 

    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    # 2nd GRU layer

    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    # Dense layer

    X = TimeDistributed(Dense(1,activation='sigmoid'))(X)

    model = Model(inputs=X_input, outputs=X)

    return model

model = model(input_shape=(Tx,n_freq))
model.summary()

#model = load_model('./models/tr_model.h5')

opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X,Y,batch_size=5,epochs=5)
loss, acc = model.evaluate(X_dev,Y_dev)
print("Dev set accuracy:", acc)


def detectTriggerWord(file_name):
    plt.subplot(2,1,1)
    x = graph_spectrogram(file_name)
    x = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    prediction = model.predict(x)
    return prediction


chime_file = "audio_examples/chime.wav"
def chime_on_activate(filename, predictions, threshold):
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 75 consecutive output steps have passed
        if predictions[0,i,0] > threshold and consecutive_timesteps > 75:
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds)*1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')
    IPython.display.Audio("./raw_data/dev/1.wav")