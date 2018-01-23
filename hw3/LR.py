import re
import sys
import math
import time
import numpy as np

# helper functions here


def tokenizeDoc(curDoc):
    return re.findall('\\w+', curDoc)


def word2id(word, V):
    id = hash(word) % V
    if id < 0:
        id += V
    return id


def sigmoid(score):
    overflow = 20.0
    if score > overflow:
        score = overflow
    elif score < -overflow:
        score = -overflow
    exp = math.exp(score)
    return exp / (1 + exp)


def correct(label, p, labels):
    if label in labels:
        if p >= 0.5:
            return True
    else:
        if p < 0.5:
            return True
    return False


class LR(object):
    """docstring for LR"""

    def __init__(self, V):
        self.label_A = {}
        self.label_B = {}
        self.LCL = {}
        self.averageLCL = {}
        self.labels = ['Person', 'other', 'Place', 'Species', 'Work']
        self.totalLCL = []
        for label in self.labels:
            self.label_A[label] = [0] * V
            self.label_B[label] = [0] * V
            self.LCL[label] = []
            self.averageLCL[label] = []

    def train(self, trainFile, V, eta, mu, maxIter, trainSize):
        iter = 0
        k = 0
        lambd = eta
        for curDoc in trainFile:
            k += 1
            if k % trainSize == 1:
                iter += 1
                lambd = eta / (iter * iter)
            docID, labels, doc = curDoc.split('\t')
            labels = labels.split(',')
            features = tokenizeDoc(doc)
            ids = []
            for word in features:
                ID = word2id(word, V)
                ids.append(ID)
            for label in self.labels:
                A = self.label_A[label]
                B = self.label_B[label]
                if label in labels:
                    y = 1
                else:
                    y = 0
                score = sum(B[ID] for ID in ids)
                p = sigmoid(score)
                if abs(p - y) > 0.1:
                    for ID in ids:
                        B[ID] += lambd * (y - p)
                        if abs(B[ID]) > 0.05:
                            B[ID] *= (1 - 2 * lambd * mu)**(k - A[ID])
                            A[ID] = k
            #     self.computeLCL(label, p, y)
            # if k % trainSize == 0:
            #     self.averageTotalLCL()
        for label in self.labels:
            A = self.label_A[label]
            B = self.label_B[label]
            for j in range(len(B)):
                B[j] *= (1 - 2 * lambd * mu)**(k - A[j])
        # print(self.averageLCL, self.totalLCL)

    def averageTotalLCL(self):
        totalLCL = 0
        for label in self.LCL.keys():
            self.averageLCL[label].append(np.mean(self.LCL[label]))
            totalLCL += np.sum(self.LCL[label])
            self.LCL[label] = []
        self.totalLCL.append(totalLCL)

    def computeLCL(self, label, p, y):
        if y == 1:
            self.LCL[label].append(np.log(p))
        else:
            self.LCL[label].append(np.log(1 - p))

    def test(self, testFile, V):
        testFile = open(testFile, 'r')
        nCorrect = 0.0
        nPredict = 0.0
        for curDoc in testFile:
            res = ''
            docID, labels, doc = curDoc.split('\t')
            labels = labels.split(',')
            nPredict += 5
            features = tokenizeDoc(doc)
            ids = []
            for word in features:
                ID = word2id(word, V)
                ids.append(ID)
            for label in self.labels:
                B = self.label_B[label]
                score = sum(B[ID] for ID in ids)
                p = sigmoid(score)
                if correct(label, p, labels):
                    nCorrect += 1
                res += '%s\t%f,' % (label, p)
            print(res[:-1])
        print("Test accuracy (D=%d): %f" % (V, nCorrect / nPredict))


if __name__ == "__main__":
    V = int(sys.argv[1])
    eta = float(sys.argv[2])
    mu = float(sys.argv[3])
    maxIter = int(sys.argv[4])
    trainSize = int(sys.argv[5])
    testFile = sys.argv[6]

    lr = LR(V)
    start = time.time()
    lr.train(sys.stdin, V, eta, mu, maxIter, trainSize)
    end = time.time()
    print("Time for training: %d" % (end - start))
    lr.test(testFile, V)
