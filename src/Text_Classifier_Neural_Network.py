# -*- coding: utf-8 -*-
"""
Text_Classifier_Neural_Net.py

This (hopefully) is a binary sentiment analysis classifier

Borrowed some ideas from here
https://gsarantitis.wordpress.com/2018/06/10/pytorch-for-natural-language-processing-a-sentiment-analysis-example/


Can put gensim directly into pytorch
https://stackoverflow.com/questions/49710537/pytorch-gensim-how-to-load-pre-trained-word-embeddings

"""
import torch


import torch.nn as nn
import torch.nn.functional as F

import os

import torch.optim as optim

import numpy as np
import pickle

from sklearn.model_selection import KFold



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



dataDirectory = "../Data_Featurised/"
modelWeightsDirectory = "../ML_Models/"


class Net(nn.Module):
    # some useful info up in here:
    #https://medium.com/deeplearningbrasilia/deep-learning-introduction-to-pytorch-5bd39421c84
    def __init__(self, inputSize):
        
        # you need to initialise the parent class first
        # because dem's the rules
        super(Net, self).__init__()
        
        
        #853
        self.fc1 = nn.Linear(inputSize, 1000)
        
        self.fc2 = nn.Linear(1000, 600)
        
        self.fc3 = nn.Linear(600, 400)
        
        self.fc4 = nn.Linear(400, 100)
        
        self.fc5 = nn.Linear(100, 1)
        
        

        #self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # here is where we define HOW the previously declared
#        # layers are going to be connected
#        h = F.leaky_relu(self.fc1(x))
#        h = F.leaky_relu(self.fc2(h))
#        h = F.leaky_relu(self.fc3(h))
        
        #sig = nn.Sigmoid()
        
        h = F.leaky_relu(self.fc1(x))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        
        h = self.fc5(h)
        
        #when using BCEWithLogitsLoss you DO NOT NEED a sigmoid at the end
        # h = torch.sigmoid(h)

        return h
    




            
    
def SetNetParams(thisNet):
    ###########################################
    ##### Define loss fn and optimiser ########
    ###########################################
    
    
         
    # NOTE - when using this DO NOT SIGMOID THE OUTPUT!!!!
    # its built-in to this function and you will get totally
    # meaningless outputs if you add one yourself, not so much deep learning 
    # as deep nonesense, and it will be entirely your fault.
    criterionTraining = nn.BCEWithLogitsLoss()
     
#    # loss tracking criterion
#    criterionTracking = nn.BCEWithLogitsLoss(reduction = 'none')
#     
    #optimizer = optim.Adam(thisNet.parameters(), lr=0.0001)
    
    optimizer = optim.SGD(thisNet.parameters(), lr=0.0003, momentum = 0.9)
     
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', patience = 5, verbose = True)
        
    return criterionTraining, optimizer, scheduler
        
            
def Train(net, featureType, trainSet, trainLabels, criterionTraining, optimizer, batchSize, device, verbose = True):
    
    ############################################
    ######### Training the Network #############
    ############################################
    
    if verbose:
        print("Train running...")
    #model_directory_path = 'C:\\Git in C\\Thesis Parent\\thesis\\Training\\CIFAR-10 Classifier Using CNN in PyTorch\\model\\'
    
        
    # this shuffles the data, this is done manually to more easily
    # keep track of the loss of data items
    
    # get a list of randomly permuted indexes of the training set
    trainsetPermutationIdxs = np.random.permutation(np.arange(trainSet.shape[0]))
    
    
    for batchStartIdx, batchIdx  in enumerate(range(0, trainSet.shape[0], batchSize)):
        
                    
        #https://pytorch.org/docs/stable/autograd.html#locally-disable-grad
        # because later in the loop i turn off grad to calculate 
        # the individual loss i need to make sure it is turned back on here
        torch.enable_grad()

        batchIdxs = trainsetPermutationIdxs[batchStartIdx : batchStartIdx + batchSize]
        
        thisBatch = []
        thisBatchCosine = []
        
        for thisIndex in batchIdxs:
            thisBatch.append(torch.from_numpy(trainSet[thisIndex]))
            thisBatchCosine.append(trainSet[thisIndex])
        
        #print("thisBatch[0].shape = ", thisBatch[0].shape)
        
        
        
        
        thisBatchStack = torch.stack(thisBatch)
        
        #print("thisBatchStack.shape = ", thisBatchStack.shape)
        
        thisLabels = torch.Tensor([trainLabels[thisIndex] for thisIndex in batchIdxs.tolist()])
        
        inputs, labels = thisBatchStack.to(device), thisLabels.to(device)
        
        # note, here we cannot rely on the batch size because maybe
        # the data set is not divisible by the batch size, then when the
        # last batch is run there will be issues, making it depend on the 
        # size of the current input is a much better solution
        labels = labels.view(thisBatchStack.shape[0],1)
        
        if featureType == "tfidf":
        #tfidf comes out in a weird shape.... no se por que
            inputs = inputs.view(thisBatchStack.shape[0],thisBatchStack.shape[2])
        
        #########################################################
        #################### do batch update ####################
        #########################################################

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.float())
        loss = criterionTraining(outputs, labels.to(device))
        loss.backward()
#        print("---")
#        print("loss = ", loss.item())
        optimizer.step()
        #########################################################
        


    return net


def TestValidation(net, featureType, valSet, valLabels, criterionTraining, optimizer, batchSize, device, verbose = True):
    
    if verbose:
        print("TestValidation running...")
        
    lossSum = 0
    
    countBatch = 0
        
    with torch.no_grad():    

        for batchIdx, batchStartIdx in enumerate(range(0, valSet.shape[0], batchSize)):
            
            countBatch += 1

            
            batchIdxs = [i for i in range(batchStartIdx, batchStartIdx + batchSize)]
            
            # the final batch will probably be too large so this 
            # ensures it doesn't go over the size of the 
            if max(batchIdxs) >= valSet.shape[0]:
                batchIdxs = [i for i in range(batchStartIdx,valSet.shape[0])]
            
            thisBatch = []
            
            for thisIndex in batchIdxs:
                thisBatch.append(torch.from_numpy(valSet[thisIndex]))
            
            #print("thisBatch[0].shape = ", thisBatch[0].shape)
            
            thisBatchStack = torch.stack(thisBatch)
            
            #print("thisBatchStack.shape = ", thisBatchStack.shape)
            
            thisLabels = torch.Tensor([valLabels[thisIndex] for thisIndex in batchIdxs])
            
            inputs, labels = thisBatchStack.to(device), thisLabels.to(device)
            
            # note, here we cannot rely on the batch size because maybe
            # the data set is not divisible by the batch size, then when the
            # last batch is run there will be issues, making it depend on the 
            # size of the current input is a much better solution
            labels = labels.view(thisBatchStack.shape[0],1)
            
            if featureType == "tfidf":
                #tfidf comes out in a weird shape.... no se por que
                inputs = inputs.view(thisBatchStack.shape[0],thisBatchStack.shape[2])
            
            #########################################################
            #################### do batch update ####################
            #########################################################
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs.float())
            loss = criterionTraining(outputs, labels.to(device))
            #########################################################

            lossSum += loss.item()
            
    meanLoss = lossSum/countBatch
            
    return meanLoss
    
    
def TestWithTestSet(net, featureType, testSet, testLabels, criterionTraining, optimizer, batchSize, device, verbose = True):
    
    if verbose:
        print("TestWithTestSet running, good luck...")
        
    torchSigmoid = nn.Sigmoid()

    totalCorrect = 0
    totalImages = 0
        
    with torch.no_grad():    

        for batchIdx,batchStartIdx  in enumerate(range(0, testSet.shape[0], batchSize)):
            
                      
            
            batchIdxs = [i for i in range(batchStartIdx, batchStartIdx + batchSize)]
            
            # the final batch will probably be too large so this 
            # ensures it doesn't go over the size of the 
            if max(batchIdxs) >= testSet.shape[0]:
                batchIdxs = [i for i in range(batchStartIdx,testSet.shape[0])]
            
            thisBatch = []
            
            for thisIndex in batchIdxs:
                thisBatch.append(torch.from_numpy(testSet[thisIndex]))
            
            #print("thisBatch[0].shape = ", thisBatch[0].shape)

            thisBatchStack = torch.stack(thisBatch)
    
            
            #print("thisBatchStack.shape = ", thisBatchStack.shape)
            
            thisLabels = torch.Tensor([testLabels[thisIndex] for thisIndex in batchIdxs])
            
            inputs, labels = thisBatchStack.to(device), thisLabels.to(device)
            
#                # note, here we cannot rely on the batch size because maybe
#                # the data set is not divisible by the batch size, then when the
#                # last batch is run there will be issues, making it depend on the 
#                # size of the current input is a much better solution
#            try:
            labels = labels.view(thisBatchStack.shape[0],1)
#            except:
#                hippo = 1
            
            if featureType == "tfidf":
                #tfidf comes out in a weird shape.... no se por que
                inputs = inputs.view(thisBatchStack.shape[0],thisBatchStack.shape[2])

            #print("time_1")
            outputs = net(inputs.float())
    
            outputs = outputs.view(outputs.size(0))
            
            outputsSigmoid = torchSigmoid(outputs)
            
            # threshold the output at 0.5 (since we are using sigmoid)
            predicted = outputsSigmoid > 0.5
            
            predicted = predicted.long()
            
            predicted = predicted.squeeze()
            
            labels = labels.squeeze()
            
            compareOutput = predicted == labels.long()
            
            #count the number of 1's in the comparison vecotr
            try:
                nCorrect = sum(compareOutput)
                
                            #conver that to a number 
                nCorrect = nCorrect.item()
                
                totalImages += outputs.size(0)
                
                totalCorrect += nCorrect
            except:
                print("----------------------------------------")
                print("there is some weird thing here")
                print("it doesn't like it if there is only one")
                print("item in the batch")
                print("----------------------------------------")
            

            #print("time_2")


    modelAccuracy = totalCorrect / totalImages * 100
            
    return modelAccuracy


def Main(featureType,
         trainSet, 
         trainLabels, 
         valSet, 
         valLabels, 
         testSet, 
         testLabels, 
         batchSize = 64, 
         dataDirectory = "../Data_Featurised/", 
         modelWeightsDirectory = "../ML_Models/", 
         modelFileName = "tempModelName",
         device = device, 
         verbose = True):
    
    
    dirname = os.path.dirname(__file__)
     
    modelFileName = modelWeightsDirectory + modelFileName + "_%s" % featureType
    
    fullFilenameTrain = os.path.join(dirname, modelFileName)
    
    bestValLoss = np.inf
    
    net = Net(trainSet.shape[1])
    
    criterionTraining, optimizer, scheduler = SetNetParams(net)
    
    accuracy = TestWithTestSet(net, featureType, valSet, valLabels, criterionTraining, optimizer, batchSize, device, verbose = verbose)
    
    print("starting accuracy = ", accuracy)
    
    
    for thisEpoch in range(50):
        
        if verbose:
            print("this epoch = ",str(thisEpoch))
        
        net = Train(net, featureType, trainSet, trainLabels, criterionTraining, optimizer, batchSize, device, verbose = verbose)
        
        valLoss = TestValidation(net, featureType, valSet, valLabels, criterionTraining, optimizer, batchSize, device, verbose = verbose)
        
        #accuracy = TestWithTestSet(net, featureType, valSet, valLabels, criterionTraining, optimizer, batchSize, device, verbose = verbose)
        
        
        if verbose:
            print("validation loss = ",valLoss)
            #print("validation acc = ", accuracy)
        
        
        if valLoss > bestValLoss:
            
            accuracy = TestWithTestSet(net, featureType, testSet, testLabels, criterionTraining, optimizer, batchSize, device, verbose = verbose)
            
            if verbose:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("Test Set Accuracy = ", accuracy)
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print("saving model...")
            
            torch.save(net.state_dict(), fullFilenameTrain)
            
            return accuracy
        else:
            #pass
            bestValLoss = valLoss
    
    accuracy = TestWithTestSet(net, featureType, testSet, testLabels, criterionTraining, optimizer, batchSize, device, verbose = verbose)
    
    if verbose:
        print("reached epoch limit")
        print("Test Set Accuracy = ", accuracy)
    
    return accuracy  

if __name__ == "__main__":
    
    # pretty much only used for 
    
    thisVerbose = True
    standarise = True
    # get feature data
    
    featureType = 'tfidf'
    
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
        
    
    
    
        
    ##########################################################
    ##############  build  cross valdation   #################
    ##########################################################            
   
    # code this
    
    if standarise:
        
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
       
        
    accuracy = Main(featureType, dataTrainList[0], labelTrainList[0], dataValList[0], labelValList[0], featureDataTest, labelsTest)
    
#    
#    inputSize = 100
#    
#    accuray = main(dataTrainList[0],labelTrainList[0],dataValList[0],labelValList[0])
#    
#    netObj.BuildNet(inputSize)
#    
#    netObj.SetNetParams()
#    
#    maxEpochs = 3
#    
#        #keep track of previous validation accuracy
#    # initialise to inf so we ensure the first 
#    # test will be less
#    bestVaildationLoss = maths.inf
#    
#    for thisEpoch in range(maxEpochs):
#        
#        
#        netObj.Train()
#
#        validationLoss = netObj.TestValidation()
#        #if the model has acieved better than the best accuracy then 
#        # save model
#        if (validationLoss > bestVaildationLoss):
#            # if we havent improved on the model then stop at this epoch and 
#            accuracy = netObj.TestWithTestSet(featureDataTest, labelsTest)
#            print("accuracy = ",accuracy)
#            break
#        else:
#            bestVaildationLoss = validationLoss

    
    
    
    
    
    
    
    
    
    
    
    
    
    