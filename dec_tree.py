
import math
import operator

import pickle


def calShannonEntropy(dataSet):
    numData = len(dataSet)
    labelCount = {}
    for item in dataSet:
        currentLabel = item[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel] = 0
        labelCount[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCount.keys():
        prob = labelCount[key] / numData
        shannonEnt -= (prob * math.log(prob, 2))

    return shannonEnt

def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no'],
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def splitDataSet(dataSet, axis, value):
    subDataSet = []
    for item in dataSet:
        if item[axis] == value:
            subLine = item[:axis]
            subLine.extend(item[axis+1:])
            subDataSet.append(subLine)
    return subDataSet

def chooseBestSplit(dataSet):
    subDataSet = []
    baseShannonEntropy = calShannonEntropy(dataSet)
    baseInfoGain = 0.0
    bestFeature = -1
    numFeature = len(dataSet[0]) - 1
    for i in range(numFeature):
        columnFeature = [feature[i] for feature in dataSet]
        uniqueFeature = set(columnFeature)
        newEntropy = 0.0
        for feature in uniqueFeature:
            subDataSet = splitDataSet(dataSet, i, feature)
            prob = len(subDataSet) / len(dataSet)
            newEntropy += prob * calShannonEntropy(subDataSet)
        # newEntropy 值越小，数据集划分的越清晰，相比较信息增益越大
        infoGain = baseShannonEntropy - newEntropy
        if infoGain > baseInfoGain:
            baseInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCount(classList):
    labelCount = {}
    for item in classList:
        if item not in labelCount.keys():
            labelCount[item] = 0
        labelCount[item] += 1
    sortedLabelCount = sorted(labelCount.items(), operator.itemgetter(1), reverse=True)
    return sortedLabelCount[0][0]

def createTree(dataSet, labels):
    classList = [item[-1] for item in dataSet]
    # 仅有一种类别
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征后，返回最多次数特征
    if len(dataSet[0]) == 1:
        return majorityCount(classList)
    bestFeature = chooseBestSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel: {}}
    featureList = [feature[bestFeature] for feature in dataSet]
    uniqueFeature = set(featureList)
    for feature in uniqueFeature:
        subLabels = labels[:]
        del subLabels[bestFeature]
        myTree[bestFeatureLabel][feature] = createTree(splitDataSet(dataSet, bestFeature, feature), subLabels)
    return myTree

def classify(myTree, labels, testVec):
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    featureIndex = labels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featureIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], labels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

def storeTree(tree, fileName):
    with open(fileName, 'wb') as fi:
        pickle.dump(tree, fi)

def loadTree(fileName):
    with open(fileName, 'rb') as fi:
        return pickle.load(fi)

if __name__ == '__main__':
    dataSet, labels = createDataSet()
    print(dataSet)
    print(labels)
    # shannonEnt = calShannonEntropy(dataSet)
    # print(shannonEnt)
    # subDataSet = splitDataSet(dataSet, 0, 1)
    # print(subDataSet)
    # bestFeature = chooseBestSplit(dataSet)
    # print(bestFeature)
    myTree = createTree(dataSet, labels)
    print(myTree)
    storeTree(myTree, 'classifierStorage.txt')
    classifyTree = loadTree('classifierStorage.txt')
    classifyLabel = classify(classifyTree, labels, [1, 0])
    print('分类结果 {}'.format(classifyLabel))













