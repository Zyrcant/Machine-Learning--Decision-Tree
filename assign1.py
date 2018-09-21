# -*- coding: utf-8 -*-
import pandas
from math import log
import sys
import time

class node: 
    def __init__(self, leaf=False, right=None, left=None, name=None, label=None):
        #if the node is a leaf node
        self.leaf = leaf;
        
        #left and right nodes
        self.right = right;
        self.left = left;
        
        #name of the attribute
        self.name = name;
        
        #label of the attribute (0 or 1)
        self.label = label;

'''
calculate best attribute by information gain
Args: dataset
Output: attribute name that has the best information gain
'''
def makeDecisionTree(examples, method):
    root = node()
    
    #count number of positive and negative in the entire set
    try:
        numPositive = ((examples.groupby('Class').size())[1])
    except:
        numPositive = 0
    try:
        numNegative = ((examples.groupby('Class').size())[0])
    except:
        numNegative = 0
        
    #the root is a leaf if all examples are negative or positive, or if number of attribtues is 0 
    root.leaf=True
    #all examples are negative
    if numPositive == 0:
        root.label = 0
        return root
    #all examples are positive
    elif numNegative == 0:
        root.label = 1
        return root
    #if number of attributes is 0
    elif len(examples.columns) == 1:
        if numPositive > numNegative:
            root.label=1
            return root
        else:
            root.label=0
            return root
    #no conditions apply, so node is not a leaf
    root.leaf = False
    
    #find best attribute
    A = None
    if method == "ig":
        A = findAttributeByInformationGain(examples)
    elif method == "vi":
        A = findAttributeByVariance(examples)
    root.name = A
    #For each possible value subsets Vi of A, only 0 or 1 because binary
    vi1 = examples[(examples[A]==1)]
    vi0 = examples[(examples[A]==0)]
    #if either subsets are empty
    if vi0.shape[0]==0:
        #count most common occurrence of Class
        try:
            vi0positive = ((examples.groupby('Class').size())[1])
        except:
            vi0positive = 0
        try:
            vi0negative = ((examples.groupby('Class').size())[0])
        except:
            vi0negative = 0
        if vi0positive > vi0negative:
            root.left= node(True, None, None, None, 1)
        else:
            root.left= node(True, None, None, None, 0)
    else:
        examplesvi0 = vi0.copy(deep=True)
        examplesvi0 = examplesvi0.drop([A], axis=1)
        root.left = makeDecisionTree(examplesvi0, method)

    if vi1.shape[0]==0:
        #count most common occurrence of Class
        try:
            vi1positive = ((examples.groupby('Class').size())[1])
        except:
            vi1positive = 0
        try:
            vi1negative = ((examples.groupby('Class').size())[0])
        except:
            vi1negative = 0
        if vi1positive > vi1negative:
            root.right= node(True, None, None, None, 1)
        else:
            root.right = node(True, None, None, None, 0)
    else:
        examplesvi1 = vi1.copy(deep=True)
        examplesvi1 = examplesvi1.drop([A], axis=1)
        root.right = makeDecisionTree(examplesvi1, method)
    return root
    
'''
calculate best attribute by information gain
Args: dataset
Output: attribute name that has the best information gain
'''
def findAttributeByInformationGain(examples):
    #calculate number of positive and negative for full set S
    
    try:
        numPositiveS = ((examples.groupby('Class').size())[1])
    except:
        numPositiveS = 0
    try:
        numNegativeS = ((examples.groupby('Class').size())[0])
    except:
        numNegativeS = 0
    entropyS = calculateEntropy(numPositiveS, numNegativeS)
    
    #store best attribute
    bestAttribute = None
    bestInformationGain = -0.1
    for column in examples.columns:
        #iterate over all attributes but class
        if(column != "Class"):
            try:
                numPositive = (examples.groupby(column).size())[1]
            except:
                numPositive = 0
            try:
                numNegative = (examples.groupby(column).size())[0]
            except:
                numNegative = 0
            total = numPositive + numNegative
            
            #subsets of the attribute that have 1 and 0 to calculate subset entropy
            df1 = examples[(examples[column]==1)]
            df0 = examples[(examples[column]==0)]
            try:
                positive1 = ((df1.groupby('Class').size())[1])
            except:
                #no positive examples in the 1 subclass
                positive1 = 0
            try:
                negative1 = ((df1.groupby('Class').size())[0])
            except:
                #no negative examples in the 1 subclass
                negative1 = 0
            try:
                #no positive examples in the 0 subclass
                positive0 = ((df0.groupby('Class').size())[1])
            except:
                positive0 = 0
            try:
                #no negative examples in the 0 subclass
                negative0 = ((df0.groupby('Class').size())[0])
            except:
                negative0 = 0
                
            #calculate entropy
            informationGain = entropyS - (((numPositive/total)*calculateEntropy(positive1, negative1)) + ((numNegative/total)*calculateEntropy(positive0, negative0)))
            #set the attribute to the best information gain
            if informationGain > bestInformationGain:
                bestInformationGain = informationGain
                bestAttribute = column
    return bestAttribute
    
'''
function to calculate the entropy
Args: number of positive and negative examples
Output: entropy as a float
'''
def calculateEntropy(numPositive, numNegative):
    total = numPositive + numNegative
    entropy = 0.0
    if numPositive == 0 or numNegative == 0:
        return 0.0
    else:
        entropy = -(numPositive/total)*log(numPositive/total, 2)-((numNegative/total)*log(numNegative/total, 2))
    return entropy

'''
function to calculate best attribute by variance impurity
Args: number of positive and negative examples
Output: best attribute name
'''
def findAttributeByVariance(examples):
    #calculate number of positive and negative for full set S
    try:
        numPositiveS = ((examples.groupby('Class').size())[1])
    except:
        numPositiveS = 0
    try:
        numNegativeS = ((examples.groupby('Class').size())[0])
    except:
        numNegativeS = 0
    viS = calculateVarianceImpurity(numPositiveS, numNegativeS)
    
    #store best attribute
    bestAttribute = None
    bestVI = -0.1
    for column in examples.columns:
        #iterate over all attributes but class
        if(column != "Class"):
            try:
                numPositive = (examples.groupby(column).size())[1]
            except:
                numPositive = 0
            try:
                numNegative = (examples.groupby(column).size())[0]
            except:
                numNegative = 0
            total = numPositive + numNegative
            
            #subsets of the attribute that have 1 and 0 to calculate subset entropy
            df1 = examples[(examples[column]==1)]
            df0 = examples[(examples[column]==0)]
            try:
                positive1 = ((df1.groupby('Class').size())[1])
            except:
                #no positive examples in the 1 subclass
                positive1 = 0
            try:
                negative1 = ((df1.groupby('Class').size())[0])
            except:
                #no negative examples in the 1 subclass
                negative1 = 0
            try:
                #no positive examples in the 0 subclass
                positive0 = ((df0.groupby('Class').size())[1])
            except:
                positive0 = 0
            try:
                #no negative examples in the 0 subclass
                negative0 = ((df0.groupby('Class').size())[0])
            except:
                negative0 = 0
                
            #calculate gain
            gain = viS - ((numPositive/total)*calculateVarianceImpurity(positive1,negative1) + ((numNegative/total)*calculateVarianceImpurity(positive0, negative0)))
            #set the attribute to the best information gain
            if gain > bestVI:
                bestVI = gain
                bestAttribute = column
    return bestAttribute
    
    
'''
function to calculate the variance impurity
Args: number of positive and negative examples
Output: variance impurity as a float
'''
def calculateVarianceImpurity(numPositive, numNegative):
    total = numPositive + numNegative
    viS = 0.0
    if numPositive == 0 or numNegative == 0:
        return 0.0
    viS = (numPositive/total)*(numNegative/total)
    return viS

'''
function to print the tree in preorder
Args: root to start the print, level of the tree
Output: Prints the decision tree
'''
def printTree(root, level):
    if root:
        spaces = ""
        for i in range(level):
            spaces += "| "
        if root.leaf == True:
            toprint = str(root.label)
            print(toprint, end=" ")
            level = level-1
        else:
            print("")
            toprint = spaces + str(root.name) + " = 0 :"
            print(toprint, end=" ")
        printTree(root.left, level+1)
        if root.leaf == False:
            print("")
            toprint = spaces + str(root.name) + " = 1 :"
            print(toprint, end=" ")
        printTree(root.right, level+1)
    

if __name__ == "__main__": 
    '''
    L = sys.argv[1];
    K = sys.argv[2]; 
    training = sys.argv[3]
    validate = sys.argv[4]
    test = sys.argv[5]
    to_print = sys.argv[6]
    '''
    training_data = pandas.read_csv('training_set2.csv')
    training_attributes = list(training_data)
    training_values = training_data.values
    root = makeDecisionTree(training_data, "vi")
    printTree(root, 0)