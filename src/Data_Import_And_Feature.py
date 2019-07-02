# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 23:00:28 2019

@author: Alex_User
"""
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import math as maths
import os
import pickle 
import numpy as np
import pandas as pd


stemmer = SnowballStemmer(language='english')




def StemWords(wordList, thisStemmer = stemmer):
    
    #note - assumes word are in list form
    
    tempSentenceList = []
    
    for thisWord in wordList:
        tempSentenceList.append(stemmer.stem(thisWord))
        
    stemmedSentence = ' '.join(tempSentenceList)
    
    return stemmedSentence

def RemoveStopWordsString(wordList):
    
    #note - assumes word are in list form
    
    filteredWords = [word for word in wordList if word not in stopwords.words('english')]
        
    stemmedSentence = ' '.join(filteredWords)
    
    return stemmedSentence


def RemoveStopWordsList(wordList):
    
    #note - assumes word are in list form
    
    filteredWords = [word for word in wordList if word not in stopwords.words('english')]
        
    
    return filteredWords

def ImportData(featureType,
               positiveSentimentFileName = "..\Data\positive_reviews.txt",
               negativeSentimentFileName = "..\Data\\negative_reviews.txt",
               unsupervisedSentimentFileName = "..\Data\\unsupervised_reviews.txt",
               thisStemmer = stemmer, 
               testProportion = 0.2,
               pickleObject = False,
               verbose = False):
    
    ##################################################
    ##########             ETL              ##########
    ##################################################
    
    # first import the data 
    # note - we assume the data is in the 
    
    # import positive sentiment   
    positiveDF = pd.read_csv(positiveSentimentFileName, sep='\n', header = None, names = ['text'])

    # add label '1' showing positive
    positiveDF['labels'] = pd.Series(np.ones(positiveDF.shape[0]))
    
    # import positive sentiment   
    negativeDF = pd.read_csv(negativeSentimentFileName, sep='\n', header = None, names = ['text'])

    # add label '1' showing positive
    negativeDF['labels'] = pd.Series(np.zeros(negativeDF.shape[0]))
    
    allDataDF = pd.concat([positiveDF,negativeDF], axis=0)
    
    # suffle - very important, dont want to forget this!
    allDataDF = allDataDF.sample(frac=1)
    
    # reset index
    allDataDF.reset_index(drop = True, inplace = True)
    
    # split into test and train
    lastTrainingindex = maths.ceil(allDataDF.shape[0] * 0.8)
    
    trainingDF = allDataDF.iloc[:lastTrainingindex, :]
    testDF = allDataDF.iloc[lastTrainingindex:, :]
    
    # reset index
    testDF.reset_index(drop = True, inplace = True)
    
    if verbose:
        print("stemming labelled words...")
        print("this could be parallelised given more time")
        print("this takes a while, maybe grab a coffee...")

    
    # convert to list of words for stemmer 
    trainingDF['word_list'] = trainingDF['text'].apply(lambda x: nltk.word_tokenize(x))
    testDF['word_list'] = testDF['text'].apply(lambda x: nltk.word_tokenize(x))
    
    #stem - needed for tfidf
    trainingDF['stemmed_text'] = trainingDF['word_list'].apply(lambda  x: StemWords(x) )
    testDF['stemmed_text'] = testDF['word_list'].apply(lambda  x: StemWords(x) )
    
    # stems as a list (maybe this is better for gensim...)
    trainingDF['stemmed_list'] = trainingDF['stemmed_text'].apply(lambda x: nltk.word_tokenize(x))
    testDF['stemmed_list'] = testDF['stemmed_text'].apply(lambda x: nltk.word_tokenize(x))
    
    # stems as a list (maybe this is better for gensim...)
    trainingDF['stop_words_removed_list'] = trainingDF['word_list'].apply(lambda x: RemoveStopWordsList(x))
    testDF['stop_words_removed_list'] = testDF['word_list'].apply(lambda x: RemoveStopWordsList(x))
    
    if featureType == 'gensim':
        
        # import unsupervised sentiment
    
        unsupervisedDF = pd.read_csv(unsupervisedSentimentFileName, sep='\n', header = None, names = ['text'])
    
        # add label '1' showing positive
        unsupervisedDF['labels'] = pd.Series(np.zeros(unsupervisedDF.shape[0])- 1)
            
        unsupervisedDF['word_list'] = unsupervisedDF['text'].apply(lambda x: nltk.word_tokenize(x))
        
        if verbose:
            print("stemming unlabelled words...")
            print("this could be parallelised given more time")
            print("this takes a while, maybe read a bit of https://www.bbc.com/news...")

        unsupervisedDF['stemmed_text'] = unsupervisedDF['word_list'].apply(lambda  x: StemWords(x) )
    
        unsupervisedDF['stemmed_list'] = unsupervisedDF['stemmed_text'].apply(lambda x: nltk.word_tokenize(x))
        
        unsupervisedDF['stop_words_removed_list'] = unsupervisedDF['word_list'].apply(lambda x: RemoveStopWordsList(x))
    else:
        unsupervisedDF = None

    
    if pickleObject:
        #pickle training data
        dirname = os.path.dirname(__file__)
        fileName = '../Data/training_data.pkl' 
        filenameFull = os.path.join(dirname, fileName)
        file = open(filenameFull, 'wb')
        pickle.dump(trainingDF, file)
        file.close()
        
        #pickle testing data
        dirname = os.path.dirname(__file__)
        fileName = '../Data/testing_data.pkl'
        filenameFull = os.path.join(dirname, fileName)
        file = open(filenameFull, 'wb')
        pickle.dump(testDF, file)
        file.close()
        
        if featureType == 'gensim':
            #pickle testing data
            dirname = os.path.dirname(__file__)
            fileName = '../Data/unsupervised_data.pkl' 
            filenameFull = os.path.join(dirname, fileName)
            file = open(filenameFull, 'wb')
            pickle.dump(unsupervisedDF, file)
            file.close()
            
    
    return trainingDF, testDF, unsupervisedDF
    
def FeaturiseData(featureType, 
                  trainDF,
                  testDF,
                  unsupervisedDF = None,
                  pickleObject = False, 
                  reduceDims = 500,
                  verbose = False):
    
    
    # convert to test and train,
    # this is needed here so as not to put training data into
    # the featurisation functions, since in reality they would be built
    # using only training dat
    

    
    if featureType == 'tfidf':
        # now we need to convert it into features so the data can be 
        # put into a machine learning model
        vectorizer = TfidfVectorizer(stop_words = 'english', analyzer = 'word', min_df = 0.02, max_df = 0.98, use_idf = False, norm = None)
        
        #
        
        featureDataTrainTemp = vectorizer.fit_transform(trainDF['stemmed_text'])
        
        featureDataTrain = featureDataTrainTemp.todense()
        
        #featureDataTrain = (featureDataTrain - dataMean)/dataSD
        
        labelsTrain = np.array(trainDF['labels'])
        
        words = vectorizer.get_feature_names()
        
        # remember ONLY TRANSFORM, don't fit!!!
        featureDataTestTemp = vectorizer.transform(testDF['stemmed_text'])
        
        featureDataTest = featureDataTestTemp.todense()
        
        #featureDataTest = (featureDataTest - dataMean)/dataSD

        labelsTest = np.array(testDF['labels'])
        
        if verbose:
            startTime = datetime.now() 
            print("reducing dims...") 
            
        # 100 dims chosen arbitrarily...
        #featureData, _, _ = scipy.sparse.linalg.svds(featureDataTrainTemp, k = 100)
        
#        svdObj = TruncatedSVD(n_components=reduceDims, n_iter=7, random_state=42)
#        
#        featureDataTrain = svdObj.fit_transform(featureDataTrainTemp)
#        
#        
#        #ONLY transform!!! do not fit
#        featureDataTest = svdObj.transform(featureDataTestTemp)
#        

        if verbose:
            tookThisLong = datetime.now() - startTime
            print("SVD took %s " % str(tookThisLong))
            print("number of  words = ", len(words))
            

            
            
            
    elif featureType =="gensim":
        
        #gensimDF = pd.concat([trainDF, unsupervisedDF])
        
        # not sure if order is important so shuffle anyway, can't hurt...
        #gensimDF = gensimDF.sample(frac = 1)

        # convert the stemmed words into a format gensim can deal with
        documentsGensim = [TaggedDocument(doc, [i]) for i, doc in enumerate(unsupervisedDF['stop_words_removed_list'])]
        
        # build doc2vec model - this could do with some experimentation...
        modelGensim = Doc2Vec(documentsGensim, vector_size=reduceDims, window=4, min_count=3, workers=6)
        
        # now use the model to infer vectors
        docVecList = []
        labels = []
        for index, row in trainDF.iterrows():

            docVecList.append(modelGensim.infer_vector(row['stop_words_removed_list']))
            labels.append(row['labels'])
            
        featureDataTrain = np.array(docVecList)
        labelsTrain = np.array(labels)
        
        docVecList = []
        labels = []
        for index, row in testDF.iterrows():

            docVecList.append(modelGensim.infer_vector(row['stop_words_removed_list']))
            labels.append(row['labels'])

        featureDataTest = np.array(docVecList)
        labelsTest = np.array(labels)
        
        #print("labelsTest.shape = ", labelsTest.shape)
        
          
    if pickleObject:
        # pickle data
        dirname = os.path.dirname(__file__)
           
        # pickle train Data
        fileNameFeatureDataTrain = '../Data_Featurised/train_data_%s.pkl' % featureType
        fileNameFullFeatureDataTrain = os.path.join(dirname, fileNameFeatureDataTrain)
        file = open(fileNameFullFeatureDataTrain, 'wb')
        pickle.dump(featureDataTrain, file)
        file.close()
        
        # pickle test Data
        fileNameFeatureDataTest = '../Data_Featurised/test_data_%s.pkl' % featureType
        fileNameFullFeatureDataTest = os.path.join(dirname, fileNameFeatureDataTest)
        file = open(fileNameFullFeatureDataTest, 'wb')
        pickle.dump(featureDataTest, file)
        file.close()
        
        # pickle train labels
        fileNameFeatureDataLabelTrain = '../Data_Featurised/train_label_data_%s.pkl' % featureType
        fileNameFullFeatureDataTrainLabel = os.path.join(dirname, fileNameFeatureDataLabelTrain)
        file = open(fileNameFullFeatureDataTrainLabel, 'wb')
        pickle.dump(labelsTrain, file)
        file.close()
        
        # pickle test Data
        fileNameFeatureDataLabelTest = '../Data_Featurised/test_label_data_%s.pkl' % featureType
        fileNameFullFeatureDataTestLabel = os.path.join(dirname, fileNameFeatureDataLabelTest)
        file = open(fileNameFullFeatureDataTestLabel, 'wb')
        pickle.dump(labelsTest, file)
        file.close()
    
        if featureType == 'tfidf':
            
            #pickle tfidf vectorizer and truncated SVD
            fileNameTfidfObj = '../Feature_Models/tfidf_vect.pkl'
            fileNameFullTfidfObj = os.path.join(dirname, fileNameTfidfObj)
            file = open(fileNameFullTfidfObj, 'wb')
            pickle.dump(vectorizer, file)
            file.close()
            
#            #pickle tfidf vectorizer and truncated SVD
#            fileNameSvdObj = '../Feature_Models/svd_obj.pkl'
#            fileNameFullSvdObj = os.path.join(dirname, fileNameSvdObj)
#            file = open(fileNameFullSvdObj, 'wb')
#            pickle.dump(svdObj, file)
#            file.close()
            
        elif featureType == 'gensim':
            
            fileNameGensimObj = '../Feature_Models/gensim_obj.pkl'
            fileNameFullGensimObj = os.path.join(dirname, fileNameGensimObj)
            modelGensim.save(fileNameFullGensimObj)
            
            
        
    return featureDataTrain, featureDataTest, labelsTrain, labelsTest
