import numpy
import cv2
import pandas as pd
import os
import json
import resampy
import matplotlib.pyplot as plt
import soundfile as sf
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
#from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import TimeDistributed,Bidirectional,Conv2D,MaxPooling2D, Flatten,Dropout,Masking
from pickle import load, dump
from numpy import argmax

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.models import Model
from keras.models import load_model
import fileinput
from numpy import reshape
use_multiprocessing=True;

outFileName="GAF2sec"

def my_range(start, step, end):
    while start <= end:
        yield start
        start += step;
def model1():

    cnn= Sequential()
    cnn.add(Conv2D(64,(2,2), activation='relu',input_shape=(150, 150,4)))
    cnn.add(Conv2D(64, (2, 2), activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(0.5))
    cnn.add(Conv2D(128, (3, 3), activation='relu'))
    cnn.add(Conv2D(128, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(MaxPooling2D())
    cnn.add(Dropout(0.5))
    cnn.add(Conv2D(256, (3, 3), activation='relu'))
    cnn.add(Conv2D(256, (3, 3), activation='relu'))
    cnn.add(MaxPooling2D())
    cnn.add(Flatten())
    cnn.add(Dropout(0.5))

    input=Input(shape=(None,150,150,4))
    mask = Masking(mask_value=0)(input)
    enc_frame_seq=TimeDistributed(cnn)(input)
    enc_fet=Bidirectional(LSTM(256))(enc_frame_seq)
    enc_fet=Dropout(0.5)(enc_fet)
    out=Dense(2000)(enc_fet)

    out=Dense(1,activation="sigmoid")(out)
    model=Model(inputs=[input],outputs=out)

    learning_rate=0.0001;
    decay_rate=learning_rate/n_epochs

    adam1=Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)

    model.compile(loss='binary_crossentropy',optimizer=adam1, metrics=['accuracy'])
    # summarize model
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

def define_model():
    model = Sequential()
    # define CNN model
    model.add(TimeDistributed(Conv2D(64,(2,2), activation='relu'),input_shape=(None,150, 150,4)))
    model.add(TimeDistributed(Conv2D(64, (2, 2), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(128, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu')))
    model.add(TimeDistributed(Conv2D(256, (3, 3), activation='relu')))
    model.add(TimeDistributed(MaxPooling2D()))
    model.add(TimeDistributed(Flatten()))
    # define LSTM model
    model.add(LSTM(256))
    model.add(Dropout(0.5))
    model.add(Dense(2000))
    model.add(Dense(1,activation="sigmoid"))
    #model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
    # summarize model
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model
# extract features from each photo in the directory

def data_generator(n_step,filenames) :
    #print("loop for ever over images")
    photos, labels = Input_generator(filenames)
    while 1:
        for i in range(0,len(photos),n_step):
            yield numpy.array([photos[i:i+n_step]]),numpy.array([labels[i:i+n_step]])

def data_generator1(n_step,filenames) :
    #print("loop for ever over images")
    #print(len(filenames))
    #print(filenames[294:300])
    
    while 1:
        for i in range(0,len(filenames),n_step):
            
            photos, labels = Input_generator(filenames[i:i+n_step])
            #check if photos is not properly filled, fill it by zero, and fill label by 0
            yield numpy.array(photos),numpy.array(labels)


def getFile(path):
    filenames=list()
    count=0
    directory = path#'./Tabla';
    for root, dirs, files in os.walk(directory, topdown=False):
        for name in files:
            fname=os.path.join(root, name)
            filenames.append(fname)
            count+=1
    return filenames, count


def Input_generator(filenames):
    XSeq = [];
    y=[]
    inSeq=[];
    yy=list();
    for file in filenames:
        [x, fs2] = sf.read(file);
        #print(file);
        dim=x.shape;
        #makes monophonic
        if len(x.shape) == 1:
            X=numpy.array(x);
        if len(x.shape) == 2:
            X = numpy.array(numpy.mean(x, axis=1));
        X=resampy.resample(X,fs2,16000)
        fs1=16000
        #sampling frequency
        fs = int(fs1 / 100);
        len1 = int(len(X));
        dSec = 2
        limit=30
        tSec=int(len1/(fs1*dSec));
        windowSize = int(len1 / fs);
        s = int(fs);
        #image initialise
        IMG = numpy.zeros((s, s));
        #find garmin matrix
        Step2=int(len1/tSec)
        ss=0;
        ss2=Step2;
        c=0
        for l in my_range(1, Step2, len1):
            c=c+1;
            if ss2>len1:
                break;
            if c>limit :
                break;
            for i in my_range(ss, fs, ss2):
                M = X[i:s + i];
                N = len(list(M));
                Y = numpy.zeros((N, 1));
                Max = numpy.max(M);
                Min = numpy.min(M)
                for i in range(N):
                    Y[i] = ((M[i] - Max) + (M[i] - Min)) / (Max - Min);
                Phi = numpy.zeros((N, 1));
                for i in range(N):
                    Phi[i] = numpy.arccos(Y[i]);
                G = numpy.zeros((N, N));
                for i in range(N):
                    for j in range(N):
                        G[i][j] = numpy.cos(Phi[i] + Phi[j]);
                IMG = IMG + G;
                    #IMG.resize((150,150), refcheck=False)
                AA=cv2.resize(IMG, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
                cmap = plt.get_cmap('jet')
                rgba_img = cmap(AA)
               
                ss=ss2;
                ss2=ss2+Step2;
                if ss2>=l:
                    break;
            #write image in inSeq
            inSeq.append(rgba_img);
            #inSeq.resize((1, 25), refcheck=False)
            #IMG1=cv2.imread("Figure_1.png",-1)
            IMG = numpy.zeros((s, s));
        #padding the left space
        tSec=len(inSeq);
        for i in range(tSec, limit):
            inSeq.insert(i,  numpy.zeros((150,150,4)));
        #inSeq = pad_sequences([inSeq], maxlen=25, padding='post', value=2)[0]
        #print('len(inSeq) ', len(inSeq))
        XSeq.append(inSeq)
        #print('XSeq  ',len(XSeq))
        s=file
        str1=s[ :15]
        if str1=="./tabla16/other":
            y.append(0)
        else :
            y.append(1)
            #print('label  ', len(y))
        inSeq=[];
    return numpy.array(XSeq),numpy.array(y)

filenames,count=getFile('./tabla16')
#valfilenames,count=getFile('./valTabla')

training_files,validation_files=train_test_split(filenames, test_size=0.1, random_state=42)
#validation_files,abc=train_test_split(valfilenames, test_size=0.0, random_state=42)

#features=extract_features(input)
#train_features, test_features = train_test_split(list(features.items()), test_size=0.1, random_state=42)
# define experiment
model_name = 'audiosignal'
verbose = 1
n_epochs = 25
n_sent_per_update = 1
n_batches_per_epoch = int(len(training_files) / n_sent_per_update)
n_repeats = 1
# run experiment
train_results, test_results = list(), list()
for i in range(n_repeats):
    # define the model
    model = model1()
    bst_model_path =outFileName+ "model" + '.hdf5'
    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint(bst_model_path, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint,early_stopping]
    history=model.fit_generator(data_generator1(n_sent_per_update,training_files),steps_per_epoch=n_batches_per_epoch, epochs=n_epochs, verbose=verbose, callbacks=callbacks_list, validation_data=data_generator1(n_sent_per_update,validation_files),validation_steps=len(validation_files)/n_sent_per_update)
    
    model.save("model1s-1.h5")
    bst_val_score = min(history.history['val_loss'])
    print("Saved model to disk")
    hist = pd.DataFrame(history.history)
    dump(hist, open(outFileName+'-history.pkl', 'wb'))

    plt.style.use("ggplot")
    fig=plt.figure(figsize=(12,12))
    #plt.plot(hist["loss"])
    plt.plot(hist["acc"])
    #plt.plot(hist["val_loss"])
    plt.plot(hist["val_acc"])
    plt.show()
    #plt.ylabel('Frequency [Hz]')
    #plt.xlabel('Time [sec]')
    plt.savefig('./photos/'+outFileName+"plt")
    fig.savefig('./photos/'+outFileName+"fig")
    plt.savefig(outFileName + "-graph.png",bbox_inches='tight')
    fig.savefig(outFileName + "-graph.png",bbox_inches='tight')
