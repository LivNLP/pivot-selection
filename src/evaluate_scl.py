"""
Forked Code from Danushka Bollegala
Implementation of SCL following steps after pivot selection
Used for evaluation of pivot selection methods
"""

import numpy as np
import scipy.io as sio 
import scipy.sparse as sp

from sparsesvd import sparsesvd

import sys, math, subprocess, time
def learnProjection(sourceDomain, targetDomain):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of SVD dimensions.
    n = 500 # no. of pivots.
    # Load pivots.
    pivotsFileName = "../work/%s-%s/DI_list" % (sourceDomain, targetDomain)
    pivots = []
    pivotsFile = open(pivotsFileName)
    for line in pivotsFile:
        pivots.append(line.split()[1])
    pivotsFile.close()

    # Load domain specific features
    DSwords = []
    DSFileName = "../work/%s-%s/DS_list" % (sourceDomain, targetDomain)
    DSFile = open(DSFileName)
    for line in DSFile:
        DSwords.append(line.split()[1])
    DSFile.close()

    feats = DSwords[:]
    feats.extend(pivots)

    # Load train vectors.
    print "Loading Training vectors...",
    startTime = time.time()
    vects = []
    vects.extend(loadFeatureVecors("../work/%s/train.positive" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../work/%s/train.negative" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../work/%s/train.unlabeled" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../work/%s/train.unlabeled" % targetDomain, feats))
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     

    print "Total no. of documents =", len(vects)
    print "Total no. of features =", len(feats)

    # Learn pivot predictors.
    print "Learning Pivot Predictors.."
    startTime = time.time()
    M = sp.lil_matrix((len(feats), len(pivots)), dtype=np.float)
    for (j, w) in enumerate(pivots[:n]):
        print "%d of %d %s" % (j, len(pivots), w)
        for (feat, val) in getWeightVector(w, vects):
            i = feats.index(feat)
            M[i,j] = val
    endTime = time.time()
    print "Took %ss" % str(round(endTime-startTime, 2))   

    # Perform SVD on M
    print "Perform SVD on the weight matrix...",
    startTime = time.time()
    ut, s, vt = sparsesvd(M.tocsc(), h)
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     
    sio.savemat("../work/%s-%s/proj.mat" % (sourceDomain, targetDomain), {'proj':ut.T})
    pass


def getWeightVector(word, vects):
    """
    Train a binary classifier to predict the given word and 
    return the corresponding weight vector. 
    """
    trainFileName = "../work/trainFile"
    modelFileName = "../work/modelFile"
    trainFile = open(trainFileName, 'w')
    for v in vects:
        fv = v.copy()
        if word in fv:
            label = 1
            fv.remove(word)
        else:
            label = -1
        trainFile.write("%d %s\n" % (label, " ".join(fv)))
    trainFile.close()
    trainLBFGS(trainFileName, modelFileName)
    return loadClassificationModel(modelFileName)


def loadFeatureVecors(fname, feats):
    """
    Returns a list of lists that contain features for a document. 
    """
    F = open(fname)
    L = []
    for line in F:
        L.append(set(line.strip().split()).intersection(set(feats)))
    F.close()
    return L


def evaluate_SA(source, target, project):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    gamma = 1.0
    print "Source Domain", source
    print "Target Domain", target
    if project:
        print "Projection ON", "Gamma = %f" % gamma
    else:
        print "Projection OFF"
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat("../work/%s-%s/proj.mat" % (source, target))['proj'])
    (nDS, h) = M.shape

    # Load domain independent features.
    pivots = []
    pivotsFileName = "../work/%s-%s/DI_list" % (source, target)
    pivotsFile = open(pivotsFileName)
    for line in pivotsFile:
        pivots.append(line.split()[1])
    pivotsFile.close()

    # Load domain specific features.
    DSwords = []
    DSFileName = "../work/%s-%s/DS_list" % (source, target)
    DSFile = open(DSFileName)
    for line in DSFile:
        DSwords.append(line.split()[1])
    DSFile.close()

    feats = DSwords[:]
    feats.extend(pivots)
    
    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.SCL" % (source, target)
    testFileName = "../work/%s-%s/testVects.SCL" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'train.positive'), (-1, 'train.negative')]:
        F = open("../work/%s/%s" % (source, fname))
        for line in F:
            count += 1
            #print "Train ", count
            words = set(line.strip().split())
            # write the original features.
            featFile.write("%d " % label)
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            for w in words:
                #featFile.write("%s:1 " % w)
                if w in feats:
                    x[0, feats.index(w)] = 1
            # write projected features.
            if project:
                y = x.tocsr().dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # write test feature vectors.
    featFile = open(testFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'test.positive'), (-1, 'test.negative')]:
        F = open("../work/%s/%s" % (target, fname))
        for line in F:
            count += 1
            #print "Test ", count
            words = set(line.strip().split())
            # write the original features.
            featFile.write("%d " % label)
            x = sp.lil_matrix((1, nDS), dtype=np.float64)
            for w in words:
                #featFile.write("%s:1 " % w)
                if w in feats:
                    x[0, feats.index(w)] = 1
            # write projected features.
            if project:
                y = x.dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.SCL" % (source, target)
    trainLBFGS(trainFileName, modelFileName)
    # Test using classias.
    acc = testLBFGS(testFileName, modelFileName)
    print "Accuracy =", acc
    print "###########################################\n\n"
    return acc


def batchEval():
    """
    Evaluate on all 12 domain pairs. 
    """
    resFile = open("../work/batchSCL.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, Proj\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            learnProjection(source, target)
            resFile.write("%s, %s, %f\n" % (source, target, evaluate_SA(source, target, True)))
            resFile.flush()
    resFile.close()
    pass

if __name__ == "__main__":
    #source = "books"
    #target = "dvd"
    #learnProjection(source, target)
    #evaluate_SA(source, target, True)
    #evaluate_SA(source, target, True)
    #batchEval()