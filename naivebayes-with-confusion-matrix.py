from __future__ import division
import pandas as pd
import collections
import math
import sys

class Model:
        def __init__(self, arffFile):
                self.trainingFile = arffFile
                self.features = {}      #all feature names and their possible values (including the class label)
                self.featureNameList = []       #this is to maintain the order of features as in the arff
                self.featureCounts = collections.defaultdict(lambda: 1)#contains tuples of the form (label, feature_name, feature_value)
                self.featureVectors = []        #contains all the values and the label as the last entry
                self.labelCounts = collections.defaultdict(lambda: 0)   #these will be smoothed later

        def TrainClassifier(self):
                for fv in self.featureVectors:
                        self.labelCounts[fv[len(fv)-1]] += 1 #udpate count of the label
                        for counter in range(0, len(fv)-1):
                                self.featureCounts[(fv[len(fv)-1], self.featureNameList[counter], fv[counter])] += 1

                for label in self.labelCounts: 
                        for feature in self.featureNameList[:len(self.featureNameList)-1]:
                                self.labelCounts[label] += len(self.features[feature])

        def Classify(self, featureVector):      #featureVector is a simple list like the ones that we use to train
                probabilityPerLabel = {}
                for label in self.labelCounts:
                        logProb = 0
                        for featureValue in featureVector:
                                logProb += math.log(self.featureCounts[(label, self.featureNameList[featureVector.index(featureValue)], featureValue)]/self.labelCounts[label])
                        probabilityPerLabel[label] = (self.labelCounts[label]/sum(self.labelCounts.values())) * math.exp(logProb)
                #print probabilityPerLabel
                return max(probabilityPerLabel, key = lambda classLabel: probabilityPerLabel[classLabel])

        def GetValues(self):
                file = open(self.trainingFile, 'r')
                for line in file:
                        if line[0] != '@':  #start of actual data
                                self.featureVectors.append(line.strip().lower().split(','))
                        else:   #feature definitions
                                if line.strip().lower().find('@data') == -1 and (not line.lower().startswith('@relation')):
                                        self.featureNameList.append(line.strip().split()[1])
                                        self.features[self.featureNameList[len(self.featureNameList) - 1]] = [featureName.strip() for featureName in line[line.find('{')+1: line.find('}')].strip().split(',')]
                
                #print("Now printing self features: ",self.features)
                #print '\n'
                #print("now printing feature name list: ",self.featureNameList)
                #print '\n'
                #print("Now printing featurevectors: ",self.featureVectors)
                #print '\n'
                #print("now printing self.labelcounts : ", self.labelCounts)
                file.close()

        def TestClassifier(self, arffFile):
                file = open(arffFile, 'r')
                y_prediction = []
                y_actual = []
                for line in file:
                        if line[0] != '@':
                                vector = line.strip().lower().split(',')
                                print "classifier: " + self.Classify(vector) + " given " + vector[len(vector) - 1]
                                y_prediction.append(self.Classify(vector))
                                y_actual.append(vector[len(vector) - 1])
                                
                y_actu = pd.Series(y_actual, name='Actual')
                y_pred = pd.Series(y_prediction, name='Predicted')
                df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
                print df_confusion
                
                '''
                print('prediction set {0}').format(y_prediction)
                print '\n'
                print('actual set {0}').format(y_actual)
                '''

if __name__ == "__main__":
        model = Model(sys.argv[1]) # argv[1] contains training file
        model.GetValues()
        model.TrainClassifier()
        model.TestClassifier(sys.argv[2]) #argv[2] contains test file
