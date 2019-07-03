# Chattermill_NLP_Challenge_Alex_G

This following has been built by alexgilbert3142@gmail.com
in aim of the chattermill NLP challenge: sentiment analysis on IMDB database of 
IMDB movie reviews.

--------------------------------------------------------------------------

Notes

GitHub did not upload that data files, in order to use the files:
negative_reviews.txt, 
positive_reviews.txt, 
unsupervised_reviews.txt 
need to be placed in the 'Data' folder


There is a potential issue with small batch sizes (the final batch whichis the remainder batch)
in the test set validation function. it doesn't affect the given batch size but it may in the future.

the mode called tfidf actually only performs frequency count,
the IDF was turned off.

It is advised to use the inbuilt pickle files since the stemming and stop word removal is very slow,
it could be parallelised given more time

In order to run the featurisation and import of data punkt and stopwords will need to be downloaded
(instructions for this will be printed to the terminal if required)

There is currently no ability to send in arguments from the command line, 
by deafult it used gensim word embedding featurisation

This software has been tested with python version 3.7 running on a 64 bit windows 10 operating system
with an intel i7 processor.

-----------------------------------------------------------------------

requirements

the following python libraries are required
torch 1.0.1

os

numpy 

pickle

sklearn

nltk

datetime 

gensim

math 

pandas as pd

in addition the following nltk data maybe required:

punkt

stopwords 



-----------------------------------------------------------------------
License

Copyright 2019 Alex Gilbert, alexgilbert3142@gmail.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, 
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies 
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

-----------------------------------------------------------------------

included files/folders

Folder - Data - holds raw data and python data objects
Folder - Data_Featurised - holds python data object which are readyfor input into the network
Folder - Feature_Models - hold models used to create features (needed to featurise more data)
Folder - ML_Models - holds the trained neural network models
Folder - Slurm - holds slurm scripts to run models

src - contains the following source code:
Data_Import_And_Feature.py - functions for import and featurisation of data
Text_Classifier_Neural_Network.py - contains functions to run neural network
NLP_Challenge_Main.py - runs the software


-----------------------------------------------------------------------

Instructions

Under the assumption the required nltk packages have been installed, or pickle
files of prebuit data are used.

navigate to "chattermill_nlp_challenge_alex_g\src" using a command line tool
with python available and run "NLP_Challenge_Main.py" with python

By default gensim will be used as features, to change this line 76 of
NLP_Challenge_Main.py will need to be changed to: featureType = 'tfidf'
to run word count featurisation


