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

outFileName="GAF0.5sec"

def my_range(start, step, end):
    while start <= end:
        yield start
        start += step;

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
    inSeq=[];
    file=filenames;
    [x, fs2] = sf.read(file);
    print('Read file',file);
    if len(x.shape) == 1:
        X=numpy.array(x);
    if len(x.shape) == 2:
        X = numpy.array(numpy.mean(x, axis=1));
    X = resampy.resample(X, fs2, 16000)
    fs1=16000
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
        if ss2>len1 :
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
            AA=cv2.resize(IMG, dsize=(150, 150), interpolation=cv2.INTER_CUBIC)
            cmap = plt.get_cmap('jet')
            rgba_img = cmap(AA)
            ss=ss2;
            ss2=ss2+Step2;
            if ss2>=l:
                break;
        inSeq.append(rgba_img);
        IMG = numpy.zeros((s, s));
    for i in range(tSec, limit):
        inSeq.insert(i,  numpy.zeros((150,150,4)));
    XSeq.append(inSeq)
    inSeq=[];
    return numpy.array(XSeq)

filenames, count=getFile('./others')

model=load_model('GAF2secmodel.hdf5')

for file in filenames:
    prediction = model.predict(Input_generator(file))  # test_seq)
    print('prediction  ',prediction)
