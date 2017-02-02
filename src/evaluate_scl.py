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

import select_pivots as pi
import re
import scipy.stats

def clopper_pearson(k,n,alpha=0.05):
    """
    http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval
    alpha confidence intervals for a binomial distribution of k expected successes on n trials
    Clopper Pearson intervals are a conservative estimate.
    """
    lo = scipy.stats.beta.ppf(alpha/2, k, n-k+1)
    hi = scipy.stats.beta.ppf(1 - alpha/2, k+1, n-k)
    return lo, hi

def trainLBFGS(train_file, model_file):
    """
    Train lbfgs on train file. and evaluate on test file.
    Read the output file and return the classification accuracy.
    """
    retcode = subprocess.call(
        "classias-train -tb -a lbfgs.logistic -pc1=0 -pc2=1 -m %s %s > /dev/null"  %\
        (model_file, train_file), shell=True)
    return retcode


def testLBFGS(test_file, model_file):
    """
    Evaluate on the test file.
    Read the output file and return the classification accuracy.
    """
    output = "../work/output"
    retcode = subprocess.call("cat %s | classias-tag -m %s -t > %s" %\
                              (test_file, model_file, output), shell=True)
    F = open(output)
    accuracy = 0
    correct = 0
    total = 0
    for line in F:
        if line.startswith("Accuracy"):
            p = line.strip().split()
            accuracy = float(p[1])
            [correct, total]=[int(s) for s in re.findall(r'\b\d+\b',p[2])]
    F.close()
    return accuracy,correct,total


def loadClassificationModel(modelFileName):
    """
    Read the model file and return a list of (feature, weight) tuples.
    """
    modelFile = open(modelFileName, "r") 
    weights = []
    for line in modelFile:
        if line.startswith('@'):
            # this is @classias or @bias. skip those.
            continue
        p = line.strip().split()
        featName = p[1].strip()
        featVal = float(p[0])
        if featName == "__BIAS__":
            # This is the bias term
            bias = featVal
        else:
            # This is an original feature.
            if featVal > 0:
                weights.append((featName, featVal))
    modelFile.close()
    return weights

def selectTh(h, t):
    """
    Select all elements of the dictionary h with frequency greater than t. 
    """
    p = {}
    for (key, val) in h.iteritems():
        if val > t:
            p[key] = val
    del(h)
    return p

def learnProjection(sourceDomain, targetDomain, pivotsMethod, n):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of SVD dimensions.
    #n = 500 # no. of pivots.

    # Parameters to reduce the number of features in the tail
    domainTh = {'books':5, 'dvd':5, 'kitchen':5, 'electronics':5}

    # Load pivots.
    pivotsFile = "../work/%s-%s/obj/%s" % (sourceDomain, targetDomain, pivotsMethod)
    features = pi.load_stored_obj(pivotsFile)
    pivots = dict(features[:n]).keys()
    print "selecting top-%d features in %s as pivots" % (n, pivotsMethod)

    # Load features and get domain specific features
    fname = "../work/%s-%s/obj/freq" % (sourceDomain, targetDomain)
    if "un_" in pivotsMethod:
        fname = "../work/%s-%s/obj/un_freq" % (sourceDomain, targetDomain)
    features = pi.load_stored_obj(fname)
    feats = selectTh(dict(features),domainTh[sourceDomain])
    print "experimental features = ", len(feats)
    # print feats.keys()

    # DSwords = [item for item in feats if item not in pivots]

    feats = feats.keys()
    # Load train vectors.
    print "Loading Training vectors...",
    startTime = time.time()
    vects = []
    vects.extend(loadFeatureVecors("../data/%s/train.positive" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../data/%s/train.negative" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../data/%s/train.unlabeled" % sourceDomain, feats))
    vects.extend(loadFeatureVecors("../data/%s/train.unlabeled" % targetDomain, feats))
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))     

    print "Total no. of documents =", len(vects)
    print "Total no. of features =", len(feats)

    # Learn pivot predictors.
    print "Learning Pivot Predictors.."
    startTime = time.time()
    M = sp.lil_matrix((len(feats), len(pivots)), dtype=np.float)
    for (j, w) in enumerate(pivots):
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
        L.append(set(line.strip().split())&(set(feats)))
    F.close()
    print L
    return L


def evaluate_SA(source, target, project, gamma, method, n):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    # Parameters to reduce the number of features in the tail
    domainTh = {'books':5, 'dvd':5, 'kitchen':5, 'electronics':5}

    # gamma = 1.0
    print "Source Domain", source
    print "Target Domain", target
    if project:
        print "Projection ON", "Gamma = %f" % gamma
    else:
        print "Projection OFF"
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat("../work/%s-%s/proj.mat" % (source, target))['proj'])
    (nDS, h) = M.shape

    # Load pivots.
    pivotsFile = "../work/%s-%s/obj/%s" % (source, target, method)
    features = pi.load_stored_obj(pivotsFile)
    pivots = dict(features[:n]).keys()
    print "selecting top-%d features in %s as pivots" % (n, method)

    # Load features 
    fname = "../work/%s-%s/obj/freq" % (source, target)
    if "un_" in method:
        fname = "../work/%s-%s/obj/un_freq" % (source, target)
    features = pi.load_stored_obj(fname)
    feats = selectTh(dict(features),domainTh[source])
    print "experimental features = ", len(feats)
    #print feats

    # DSwords = [item for item in feats if item not in pivots]

    feats = feats.keys()
    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.SCL" % (source, target)
    testFileName = "../work/%s-%s/testVects.SCL" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'train.positive'), (-1, 'train.negative')]:
        F = open("../data/%s/%s" % (source, fname))
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
        F = open("../data/%s/%s" % (target, fname))
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
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals

# NoAdapt Baseline
def evaluate_NA(source, target):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    print "Source Domain", source
    print "Target Domain", target 

    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.NA" % (source, target)
    testFileName = "../work/%s-%s/testVects.NA" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'train.positive'), (-1, 'train.negative')]:
        F = open("../data/%s/%s" % (source, fname))
        for line in F:
            count += 1
            # print "Train ", count
            words = set(line.strip().split())
            # write the original features.
            for w in words:
                featFile.write("%d %s\n" % (label, w))
                # print "%d %s\n" % (label, w)
        F.close()
    featFile.close()
    # write test feature vectors.
    featFile = open(testFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'test.positive'), (-1, 'test.negative')]:
        F = open("../data/%s/%s" % (target, fname))
        for line in F:
            count += 1
            #print "Test ", count
            words = set(line.strip().split())
            # write the original features.
            for w in words:
                featFile.write("%d %s\n" % (label, w))
                # print "%d %s\n" % (label, w)
        F.close()
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.NA" % (source, target)
    trainLBFGS(trainFileName, modelFileName)
    # Test using classias.
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals

# InDomain Baseline
def evaluate_InDomain(source, target):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    print "Source Domain", source
    print "Target Domain", target 

    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.ID" % (source, target)
    testFileName = "../work/%s-%s/testVects.ID" % (source, target)
    featFile = open(trainFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'train.positive'), (-1, 'train.negative')]:
        F = open("../data/%s/%s" % (target, fname))
        for line in F:
            count += 1
            # print "Train ", count
            words = set(line.strip().split())
            # write the original features.
            for w in words:
                featFile.write("%d %s\n" % (label, w))
                # print "%d %s\n" % (label, w)
        F.close()
    featFile.close()
    # write test feature vectors.
    featFile = open(testFileName, 'w')
    count = 0
    for (label, fname) in [(1, 'test.positive'), (-1, 'test.negative')]:
        F = open("../data/%s/%s" % (target, fname))
        for line in F:
            count += 1
            #print "Test ", count
            words = set(line.strip().split())
            # write the original features.
            for w in words:
                featFile.write("%d %s\n" % (label, w))
                # print "%d %s\n" % (label, w)
        F.close()
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.ID" % (source, target)
    trainLBFGS(trainFileName, modelFileName)
    # Test using classias.
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals


def batchEval(method, gamma, n):
    """
    Evaluate on all 12 domain pairs. 
    """
    resFile = open("../work/batchSCL.%s.csv"% method, "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, Method, Acc, IntLow, IntHigh\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            learnProjection(source, target, method, n)
            evaluation = evaluate_SA(source, target, True, gamma, method, n)
            resFile.write("%s, %s, %s, %f, %f, %f\n" % (source, target, method, evaluation[0], evaluation[1][0],evaluation[1][1]))
            resFile.flush()
    resFile.close()
    pass

def batchNA():
    resFile = open("../work/batchNA.NA.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, Method, Acc, IntLow, IntHigh\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            evaluation = evaluate_NA(source, target)
            resFile.write("%s, %s, %s, %f, %f, %f\n" % (source, target, 'NA', evaluation[0], evaluation[1][0],evaluation[1][1]))
            resFile.flush()
    resFile.close()
    pass

def batchID():
    resFile = open("../work/batchID.ID.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, Method, Acc, IntLow, IntHigh\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            evaluation = evaluate_InDomain(source, target)
            resFile.write("%s, %s, %s, %f, %f, %f\n" % (source, target, 'ID', evaluation[0], evaluation[1][0],evaluation[1][1]))
            resFile.flush()
    resFile.close()
    pass

def choose_gamma(source, target, method, gammas, n):
    resFile = open("../work/gamma/%s-%s/SCLgamma.%s.csv"% (source, target, method), "w")
    resFile.write("Source, Target, Method, Proj, Gamma\n")
    learnProjection(source, target, method, n)
    for gamma in gammas:    
        resFile.write("%s, %s, %s, %f, %f\n" % (source, target, method, evaluate_SA(source, target, True, gamma, method, n), gamma))
        resFile.flush()
    resFile.close()
    pass

def choose_param(method,params,gamma,n):
    resFile = open("../work/sim/SCLparams.%s.csv"% method, "w")
    resFile.write("Source, Target, Model, Acc, IntLow, IntHigh, Param\n")
    domains = ["books", "electronics", "dvd", "kitchen"]
    for param in params:
        test_method = "test_%s_%f"% (method,param)
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                learnProjection(source, target, test_method, n)
                evaluation = evaluate_SA(source, target, True, gamma, test_method, n)
                resFile.write("%s, %s, %s, %f, %f, %f, %f\n" % (source, target, method , evaluation[0], evaluation[1][0],evaluation[1][1],param))
                resFile.flush()
    resFile.close()
    pass

if __name__ == "__main__":
    # source = "dvd"
    # target = "books"
    # evaluate_NA(source,target)
    # batchNA()
    # batchID()
    # methods = ["un_mi"]
    # learnProjection(source, target, method, 500)
    # evaluate_SA(source, target, True, method, 500)
    # methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi"]
    # methods += ["ppmi",'un_ppmi']
    # methods = ["mi","un_mi","pmi","un_pmi"]
    # methods += ["landmark_pretrained_word2vec","landmark_pretrained_word2vec_ppmi","landmark_pretrained_glove","landmark_pretrained_glove_ppmi"]
    # methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove"]
    # n = 500
    # for method in methods:
    #     batchEval(method, 1, n)
    # gammas = [1,5,10,20,50,100]
    # for method in methods:
        # choose_gamma(source, target, method,gammas,n)
    # params = [0,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    # params += [10e-3,10e-4,10e-5,10e-6]
    # params.sort()
    # params = [1,50,100,1000,10000]
    # params = [0,1,50,100,1000,10000]
    # for method in methods:
    #     choose_param(method,params,1,n)
    domainTh = {'books':5, 'dvd':5, 'kitchen':5, 'electronics':5}
    source = "kitchen"
    target = "books"
    method = "un_mi"
    M = sp.csr_matrix(sio.loadmat("../work/%s-%s/proj.mat" % (source, target))['proj'])
    (nDS, h) = M.shape
    print nDS,h
    fname = "../work/%s-%s/obj/freq" % (source, target)
    if "un_" in method:
        fname = "../work/%s-%s/obj/un_freq" % (source, target)
    features = pi.load_stored_obj(fname)
    
    feats = selectTh(dict(features),domainTh[source])
    feats = feats.keys()
    print "experimental features = ", len(feats)
    x = sp.lil_matrix((1, nDS), dtype=np.float64)
    print x[0].shape
    if 'wish' in feats:
        x[0,feats.index('wish')]=1
    print x
    y = x.tocsr().dot(M)
    print y[0].shape
    



