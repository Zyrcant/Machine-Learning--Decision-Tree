# -*- coding: utf-8 -*-
'''
Written by Tiffany Do (tdd160030)
CS 6375.003
September 2018

Description: 
A program that creates a decision tree given binary attributes with binary classifications out of a training set.
Uses supervised learning and the ID3 algorithm to generate tree and a greedy post-pruning algorithm to maximize accuracy given a validation set.
Tests accuracy of pre-pruned tree and post-pruned tree given a test set given two heuristics: information gain and variance impurity.

Arguments are given through command line:
.\program <L> <K> <training-set> <validation-set> <test-set> <to-print>
where L and K are integers used in the post-pruning algorithm
'''

import pandas
from math import log
import sys
import copy
import random

'''
Class for node
@param leaf - Whether or not the node is a leaf node
@param right - Right child of node
@param left - Left child of node
@param name - Attribute name of node
@param label - If the node is 1 or 0 (for leaf nodes)
'''
class node: 
    def __init__(self, leaf=False, right=None, left=None, name=None, label=None, data=None):
        #if the node is a leaf node
        self.leaf = leaf
        
        #left and right nodes
        self.right = right
        self.left = left
        
        #name of the attribute
        self.name = name
        
        #label of the attribute (0 or 1)
        self.label = label
        
        #data subset at this node
        self.data = data

'''
Generates a decision tree
@param examples - Dataset to be used
@param method - Method by information gain or variance impurity 
Output: attribute name that has the best information gain
'''
def makeDecisionTree(examples, method):
    root = node()
    root.data = examples
    
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
            root.left= node(True, None, None, None, 1, examples)
        else:
            root.left= node(True, None, None, None, 0, examples)
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
            root.right= node(True, None, None, None, 1, examples)
        else:
            root.right = node(True, None, None, None, 0, examples)
    else:
        examplesvi1 = vi1.copy(deep=True)
        examplesvi1 = examplesvi1.drop([A], axis=1)
        root.right = makeDecisionTree(examplesvi1, method)
    return root
    
'''
Calculate best attribute by information gain
@param examples - Dataset to calculate information gain
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
@param numPositive - Number of positive examples
@param numNegative - Number of negative examples
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
@param examples - Dataset to calculate best attribute
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
@param numPositive - number of positive examples
@param numNegative - number of negative examples
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
@param root: node to begin printing at
@param level:level of the tree (height it is currently at)
@return: Prints the decision tree from the node in preorder
'''
def printTree(root, level=0):
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

'''
greedy function to prune a tree for best accuracy
@param l: given value for pruning
@param k: given value for pruning
@param tree: node of decision tree
@validation: validation set to prune with
@return: a pruned tree 
'''
def postpruning(l, k, tree, validation):
    bestTree = tree
    bestAcc = calculateAccuracy(bestTree, validation)
    for i in range(1, l):
        treeD = copy.deepcopy(tree)
        m = random.randint(1, k)
        for j in range(1, m):
            treeList = makeTreeList(treeD)
            n = len(treeList)
            p = random.randint(1, n)
            pruneNode = treeList[p-1]
            left = findMajority(pruneNode.left)
            right = findMajority(pruneNode.right)
            pruneNode.left = node(True, None, None, None, left, None)
            pruneNode.right = node(True, None, None, None, right, None)
        treeDAcc = calculateAccuracy(treeD, validation)
        if treeDAcc > bestAcc:
            bestTree = treeD
            bestAcc = treeDAcc
    return bestTree

'''
greedy function to prune a tree for best accuracy
@param l: given value for pruning
@param k: given value for pruning
@param tree: node of decision tree
@validation: validation set to prune with
@return: a pruned tree 
'''
def findMajority(node):
    if node.leaf == True:
        return node.label
    else:
        try:
            positive = (node.left.data.groupby('Class').size())[1]
        except:
            positive = 0
        try: 
            negative = (node.right.data.groupby('Class').size())[0]
        except:
            negative = 0
    if positive >= negative:
        return 0
    else:
        return 1

'''
Makes a list out of a tree in preorder
@param tree: Node to start the list at
@return: A list that has the tree starting from the node in preorder
'''
def makeTreeList(tree):
    #don't count leaves
    if(tree.leaf):
        return[]
    return [tree] + makeTreeList(tree.left) + makeTreeList(tree.right)

'''
function to calculate accuracy of tree given a dataset
@param tree: node that the decision tree begins at
@param dataset: Dataset to be classified
@return: Percentage of accurate classifications of dataset
'''
def calculateAccuracy(tree, dataset):
    total = dataset.shape[0]
    #number of correctly classified instances
    correct = 0
    rootNode = copy.deepcopy(tree)
    for index, row in dataset.iterrows():
        while tree.leaf == False:
            attribute = tree.name
            if row[attribute] == 0:
                tree = tree.left
            else:
                tree = tree.right
        #tree is now leaf
        if tree.label == 1 and row['Class']==1:
            correct = correct + 1
        elif tree.label == 0 and row['Class']==0:
            correct = correct + 1
        #go back to root
        tree = rootNode
    return (correct/total)
    
if __name__ == "__main__": 
    #command line arguments
    l = int(sys.argv[1]);
    k = int(sys.argv[2]); 
    training = sys.argv[3]
    validate = sys.argv[4]
    test = sys.argv[5]
    to_print = sys.argv[6]
    
    
    training_data = pandas.read_csv(training)
    validation_data = pandas.read_csv(validate)
    test_data = pandas.read_csv(test)
    
    #copy training_data for variance impurity tree
    training_data2 = copy.deepcopy(training_data)
    
    #make trees
    infogainTree = makeDecisionTree(training_data, "ig")
    viTree = makeDecisionTree(training_data2, "vi")
    
    print("Accuracy using information gain: " + str(calculateAccuracy(infogainTree, test_data)))
    print("Accuracy using variance impurity: " + str(calculateAccuracy(viTree, test_data)))
    newIG = postpruning(l, k, infogainTree, validation_data)
    print("Postpruning accuracy of information gain: " + str(calculateAccuracy(newIG, test_data)))
    newVI = postpruning(l, k, viTree, validation_data)
    print("Postpruning accuracy of variance impurity: " + str(calculateAccuracy(newVI, test_data)))
    
    if(to_print == "yes"):
        print("Tree using information gain:")
        printTree(infogainTree)
        print("\n\nTree using variance impurity:")
        printTree(viTree)
        print("\n\nPostpruned information gain tree:")
        printTree(newIG)
        print("\n\nPostpruned variance impurity tree:")
        printTree(newVI)