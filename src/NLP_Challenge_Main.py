# -*- coding: utf-8 -*-
"""
Alex Gilbert's code for the Chattermill NLP Challenge

https://github.com/chattermill/nlp-challenge

NLP_Challenge_Main.py

06/2019

Alexgilbert3142@gmail.com

"""

#################################################
##########      ideas/observations     ##########
#################################################

# both stemmers with this wordList = ["can", "can't", "cannot", "eat", "eating"]
# produce "can can't cannot eat eat"
# ideally I'd want cannot and can't to be mapped to the same stem...

# there is a lot of slag e.g.
# aahhhhhh
# aaagagghhhh
# agh

# these should probably be mapped to the same stem
# maybe there is an internet slag stemmer out there somewhere...

#punctuation has been removed don't = don t, this may have some interseting side effects...


# might be useful 
#https://medium.com/@GeneAshis/nlp-sentiment-analysis-on-imdb-movie-dataset-fb0c4d346d23

# accirding to the above all approaches get 90%+ accuracy


#gensim link
#https://radimrehurek.com/gensim/models/doc2vec.html



import numpy as np
import pandas as pd
import pickle
import os




import Text_Classifier_Neural_Network
import Data_Import_And_Feature

# surpress error when using lamdba function on the 
# dataframes, I had a look and the data appears to be find
# gven more time I'd investigate further
pd.options.mode.chained_assignment = None  # default='warn'


from sklearn.model_selection import KFold







if __name__ == "__main__":
    
    # NOTE - the gensim feature DF has unsupervised data with label -1
    # the tfidf has only negative or positive
    
    # options for featurisation: 'gensim or tfidf'
    featureType = 'gensim'
    #featureType = 'tfidf'
    
    thisVerbose = True
    
    if thisVerbose:
        print("using model %s" % featureType)
    
    standarise = True
    
    # useful when testing not to have to rerun 
    # the same slow preprocessing since it only needs to be done
    # once then can be stored
    
    # first check if featureised data is available
    
    dirname = os.path.dirname(__file__)
     
    fileNameFeatureDataTrain = '../Data_Featurised/train_data_%s.pkl' % featureType
    fileNameFullFeatureDataTrain = os.path.join(dirname, fileNameFeatureDataTrain)
    trainFeaturesExist = os.path.isfile(fileNameFullFeatureDataTrain)
         
    fileNameFeatureDataTest = '../Data_Featurised/test_data_%s.pkl' % featureType
    fileNameFullFeatureDataTest = os.path.join(dirname, fileNameFeatureDataTest)
    testFeaturesExist = os.path.isfile(fileNameFullFeatureDataTest)
    
         
    fileNameFeatureDataLabelTrain = '../Data_Featurised/train_label_data_%s.pkl' % featureType
    fileNameFullFeatureDataLabelTrain = os.path.join(dirname, fileNameFeatureDataLabelTrain)
    trainLabelFeaturesExist = os.path.isfile(fileNameFullFeatureDataLabelTrain)
    
    fileNameFeatureDataLabelTest = '../Data_Featurised/test_label_data_%s.pkl' % featureType
    fileNameFullFeatureDataLabelTest = os.path.join(dirname, fileNameFeatureDataLabelTest)
    testLabelFeaturesExist = os.path.isfile(fileNameFullFeatureDataLabelTest)
    
    if (trainFeaturesExist and testFeaturesExist and trainLabelFeaturesExist and testLabelFeaturesExist):
        # all the required feature data exists so import it
        
        if thisVerbose:
            print("feature data found, unpickling")
        file = open(fileNameFullFeatureDataTrain, 'rb')
        featureDataTrain = pickle.load(file)
        file.close()
        
        file = open(fileNameFullFeatureDataTest, 'rb')
        featureDataTest = pickle.load(file)
        file.close()
        
        # all the required feature data exists so import it
        file = open(fileNameFullFeatureDataLabelTrain, 'rb')
        labelsTrain = pickle.load(file)
        file.close()
        
        file = open(fileNameFullFeatureDataLabelTest, 'rb')
        labelsTest = pickle.load(file)
        file.close()
        
        
    else: # we have no feature data we need to create it
        if thisVerbose:
            print("feature not data found")
            print("checking for data objects")
    
        ##########################################################
        ################ import training data ####################
        ##########################################################
        
        # check if we already have training data in variables
        # useful when building code 
            
        # see if training data has been previously pickled
        
        fileNameTrain = '../Data/training_data.pkl'
        fullFilenameTrain = os.path.join(dirname, fileNameTrain)
        
        fileNameTest = '../Data/testing_data.pkl' 
        fullFilenameTest = os.path.join(dirname, fileNameTest)
        
        fileNameUnsupervised = '../Data/unsupervised_data.pkl' 
        fullFilenameUnsupervised = os.path.join(dirname, fileNameUnsupervised)
        
        if (os.path.isfile(fullFilenameTrain)) and (os.path.isfile(fullFilenameTest)):
            # if so then open pickle 
            
            if thisVerbose:
                print("data objects found unpickling")

            file = open(fullFilenameTrain, 'rb')
            trainDF = pickle.load(file)
            file.close()
            
            file = open(fullFilenameTest, 'rb')
            testDF = pickle.load(file)
            file.close()
            
            if os.path.isfile(fullFilenameUnsupervised):
                file = open(fullFilenameUnsupervised, 'rb')
                unsupervisedDF = pickle.load(file)
                file.close()
            else:
                unsupervisedDF = None
                
            
        else:
            if thisVerbose:
                print("data objects not found")
                print("building data objects")
            # if not, build the data and pickle for later use
            trainDF, testDF, unsupervisedDF = Data_Import_And_Feature.ImportData(pickleObject = True, verbose = thisVerbose)
            
                
        ##########################################################
        
    
        ##########################################################
        ################   featurise  data   ####################
            ##########################################################            
        if thisVerbose:
            print("building feature objects")
        
        featureDataTrain, featureDataTest, labelsTrain, labelsTest = Data_Import_And_Feature.FeaturiseData(featureType,
                                                                                       trainDF, 
                                                                                       testDF,
                                                                                       unsupervisedDF,
                                                                                       pickleObject = True,
                                                                                       verbose = thisVerbose)

        ##########################################################
        
    
    
    ##########################################################
    ##############  build  cross valdation   #################
    ##########################################################            
   
    if standarise:
        
        if thisVerbose:
            print("standardising features")
        
        thisMean = np.mean(featureDataTrain)
        thisStd = np.std(featureDataTrain)
        
        featureDataTrain = (featureDataTrain - thisMean) / thisStd
        
        featureDataTest = (featureDataTest - thisMean) / thisStd
    
    if thisVerbose:
        print("building cross validation sets")
    
    kf = KFold(n_splits=5)
    
    dataTrainList = []
    dataValList = []
    labelTrainList = []
    labelValList = []
    
    for trainIndex, valIndex in kf.split(featureDataTrain):
        dataTrain, dataVal = featureDataTrain[trainIndex], featureDataTrain[valIndex]
        labelTrain, labelVal = labelsTrain[trainIndex], labelsTrain[valIndex]
        dataTrainList.append(dataTrain)
        dataValList.append(dataVal)
        labelTrainList.append(labelTrain)
        labelValList.append(labelVal)
       

    ##########################################################
    
    
    ##########################################################
    ##################    train models     ###################     
    ##########################################################            
   

    for cvIndex, ele in enumerate(dataTrainList):
        modelFileName = "Model_Fold_%s" % str(cvIndex)
        accuracy = Text_Classifier_Neural_Network.Main(featureType, dataTrainList[cvIndex], labelTrainList[cvIndex], dataValList[cvIndex], labelValList[cvIndex], featureDataTest, labelsTest, modelFileName = modelFileName)
    
    ##########################################################
    

    
