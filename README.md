# teentaal-identify
Teen Tala Identification System

Supervised learning method is adopted for the identified of teentaal from audio signal. This program uses deep neural network for training teen taal from tabla signals. The audio signal is first converted into Gramian matrix and then image is obtained from this matrix.  The input signal is transformed to images thus convert the problem domain to image classification. From
the input signal , a sequence of image will be created by dividing input data into fixed size of intervals. The images will be passed through CNN network to extract the features of images which are thus further passed through LSTM layer to classify the sequence as teentaal or others.

# Data
For data mail us at josangurpreet@pbi.ac.in

# Training
For training use command

python3 GAF2s.py

# Testing
For testing file type

python3 GAFTest.py

The data should be directory tabla16

