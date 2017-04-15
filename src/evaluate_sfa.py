"""
Forked Code from Danushka Bollegala
Implementation of SFA following steps after pivot selection
Used for evaluation of pivot selection methods
"""
import sys
import math
import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
from sparsesvd import sparsesvd
import subprocess

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
            accuracy = float(p[1])
            [correct, total]=[int(s) for s in re.findall(r'\b\d+\b',p[2])]
    F.close()
    return accuracy,correct,total


def getCounts(S, M, fname):
    """
    Get the feature co-occurrences in the file fname and append 
    those to the dictionary M. We only consider features in S.
    """
    count = 0
    F = open(fname)
    for line in F:
        count += 1
        #if count > 1000:
        #   break
        allP = line.strip().split()
        p = []
        for w in allP:
            if w in S:
                p.append(w) 
        n = len(p)
        for i in range(0,n):
            for j in range(i + 1, n):
                pair = (p[i], p[j])
                rpair = (p[j], p[i])
                if pair in M:
                    M[pair] += 1
                elif rpair in M:
                    M[rpair] += 1
                else:
                    M[pair] = 1
    F.close()
    pass

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

def getVal(x, y, M):
    """
    Returns the value of the element (x,y) in M.
    """
    if (x,y) in M:
        return M[(x,y)] 
    elif (y,x) in M:
        return M[(y,x)]
    else:
        return 0
    pass

def getVocab(S, fname):
    """
    Get the frequency of each feature in the file named fname. 
    """
    F = open(fname)
    for line in F:
        p = line.strip().split()
        for w in p:
            S[w] = S.get(w, 0) + 1
    F.close()
    pass

def createMatrix(source, target, method, n):
    """
    Read the unlabeled data (test and train) for both source and the target domains. 
    Compute the full co-occurrence matrix. Drop co-occurrence pairs with a specified
    minimum threshold. For a feature w, compute its score(w),

    and sort the features in the descending order of their scores. 
    Write the co-occurrence matrix to a file with name source-target.cooc (fid, fid, cooc) and the 
    scores to a file with name source-target.pmi (feat, fid, score).
    """

    # Parameters
    domainTh = {'books':5, 'dvd':5, 'kitchen':5, 'electronics':5}
    coocTh = 5
    print "Source = %s, Target = %s" % (source, target)
    
    # Load features
    fname = "../work/%s-%s/obj/freq" % (source, target)
    if "un_" in method:
        fname = "../work/%s-%s/obj/un_freq" % (source, target)
    features = pi.load_stored_obj(fname)
    V = selectTh(dict(features),domainTh[source])
    
    # Compute the co-occurrences of features in reviews
    M = {}
    print "Vocabulary size =", len(V)
    getCounts(V, M, "../data/%s/train.positive" % source)
    print "%s positive %d" % (source, len(M)) 
    getCounts(V, M, "../data/%s/train.negative" % source)
    print "%s negative %d" % (source, len(M))
    getCounts(V, M, "../data/%s/train.unlabeled" % source)
    print "%s unlabeled %d" % (source, len(M))
    # getCounts(V, M, "../data/%s/train.positive" % target)
    # print "%s positive %d" % (target, len(M))   
    # getCounts(V, M, "../data/%s/train.negative" % target)
    # print "%s negative %d" % (target, len(M))   
    getCounts(V, M, "../data/%s/train.unlabeled" % target)
    print "%s unlabeled %d" % (target, len(M))  
    # Remove co-occurrence less than the coocTh
    M = selectTh(M, coocTh)

    print "selecting top-%d features in %s as pivots" % (n, method)
    pivotsFile = "../work/%s-%s/obj/%s" % (source, target, method)
    features = pi.load_stored_obj(pivotsFile)
    pi.load_stored_obj(pivotsFile)
    DI = dict(features[:n]).keys()
    # DI = []
    # for w, v in pivots:
    #     pivotsFile.write("%d %s P %s\n" % (i+1, w, str(v))) 
    #     DI.append(w)
    # pivotsFile.close()

    DSList = [item for item in V.keys() if item not in DI]
    print "Total no. of domain specific features =", len(DSList)

    # Domain specific feature list.
    DSFile = open("../work/%s-%s/DS_list" % (source, target), 'w')
    count = 0
    for w in DSList:
        count += 1
        DSFile.write("%d %s\n" % (count, w))
    DSFile.close() 
    nDS = len(DSList)
    nDI = len(DI)
    # Compute matrix DSxSI and save it. 
    R = np.zeros((nDS, nDI), dtype=np.float)
    for i in range(0, nDS):
        for j in range(0, nDI):
            val = getVal(DSList[i], DI[j], M)
            if val > coocTh:
                R[i,j] = val
    print "Writing DSxDI.mat...",
    sio.savemat("../work/%s-%s/DSxDI.mat" % (source, target), {'DSxDI':R})
    print "Done"
    pass

def learnProjection(sourceDomain, targetDomain):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of latent dimensions.
    print "Loading the bipartite matrix...",
    coocData = sio.loadmat("../work/%s-%s/DSxDI.mat" % (sourceDomain, targetDomain))
    M = sp.lil_matrix(coocData['DSxDI'])
    (nDS, nDI) = M.shape
    print "Done."
    print "Computing the Laplacian...",
    D1 = sp.lil_matrix((nDS, nDS), dtype=np.float64)
    D2 = sp.lil_matrix((nDI, nDI), dtype=np.float64)
    for i in range(0, nDS):
        D1[i,i] = 1.0 / np.sqrt(np.sum(M[i,:].data[0]))
    for i in range(0, nDI):
        D2[i,i] = 1.0 / np.sqrt(np.sum(M[:,i].T.data[0]))
    B = (D1.tocsr().dot(M.tocsr())).dot(D2.tocsr())
    print "Done."
    print "Computing SVD...",
    ut, s, vt = sparsesvd(B.tocsc(), h)
    sio.savemat("../work/%s-%s/proj.mat" % (sourceDomain, targetDomain), {'proj':ut.T})
    print "Done."    
    pass


def evaluate_SA(source, target, project,gamma, n):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
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
    # Load the domain specific features.
    DSfeat = {}
    DSFile = open("../work/%s-%s/DS_list" % (source, target))
    for line in DSFile:
        p = line.strip().split()
        DSfeat[p[1].strip()] = int(p[0])
    DSFile.close()
    
    
    # write train feature vectors.
    trainFileName = "../work/%s-%s/trainVects.SFA" % (source, target)
    testFileName = "../work/%s-%s/testVects.SFA" % (source, target)
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
                if w in DSfeat:
                    x[0, DSfeat[w] - 1] = 1
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
                if w in DSfeat:
                    x[0, DSfeat[w] - 1] = 1
            # write projected features.
            if project:
                y = x.dot(M)
                for i in range(0, h):
                    featFile.write("proj_%d:%f " % (i, gamma * y[0,i])) 
            featFile.write("\n")
        F.close()
    featFile.close()
    # Train using classias.
    modelFileName = "../work/%s-%s/model.SFA" % (source, target)
    trainLBFGS(trainFileName, modelFileName)
    # Test using classias.
    [acc,correct,total] = testLBFGS(testFileName, modelFileName)
    intervals = clopper_pearson(correct,total)
    print "Accuracy =", acc
    print "Intervals=", intervals
    print "###########################################\n\n"
    return acc,intervals

def batchEval(method, gamma):
    """
    Evaluate on all 12 domain pairs. 
    """
    resFile = open("../work/f-batchSFA.%s.csv"% method, "w")
    # domains = ["books", "electronics", "dvd", "kitchen"]
    domains = ["books","dvd"]
    numbers = [100,200,300,400,500,600,700,800,900,1000]
    resFile.write("Source, Target, Method, Acc, IntLow, IntHigh,#pivots\n")
    for source in domains:
        for target in domains:
            if source == target:
                continue
            for n in numbers:
                createMatrix(source, target, method, n)
                learnProjection(source, target)
                evaluation = evaluate_SA(source, target, True, gamma, n)
                resFile.write("%s, %s, %s, %f, %f, %f, %f\n" % (source, target, method, evaluation[0], evaluation[1][0],evaluation[1][1],n))
                resFile.flush()
    resFile.close()
    pass

def choose_gamma(source, target, method, gammas, n):
    resFile = open("../work/gamma/%s-%s/SFAgamma.%s.csv"% (source, target, method), "w")
    resFile.write("Source, Target, Method, NoProj, Proj, Gamma\n")
    createMatrix(source, target, method, n)
    learnProjection(source, target)
    for gamma in gammas:    
        resFile.write("%s, %s, %s, %f, %f, %f\n" % (source, target, method, 
        evaluate_SA(source, target, False, gamma, n), evaluate_SA(source, target, True, gamma, n), gamma))
        resFile.flush()
    resFile.close()
    pass

def choose_param(method,params,gamma):
    domains = ["books", "electronics", "dvd", "kitchen"]
    # domains = ["books", "dvd"]
    # numbers = [100,200,300,400,500,600,700,800,900,1000]
    numbers=[500]
    resFile = open("../work/sim/SFAparams.%s.csv"% method, "w")
    resFile.write("Source, Target, Model, Acc, IntLow, IntHigh, Param\n")
    for param in params:
        test_method = "test_%s_%f"% (method,param)
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                for n in numbers:
                    createMatrix(source, target, test_method, n)
                    learnProjection(source, target)
                    evaluation = evaluate_SA(source, target, True, gamma, n)
                    resFile.write("%s, %s, %s, %f, %f, %f, %f, %f\n" % (source, target, method, evaluation[0], evaluation[1][0],evaluation[1][1],param,n))
                    resFile.flush()
    resFile.close()
    pass


if __name__ == "__main__":
    # source = "dvd"
    # target = "books"
    # method = "freq"
    # createMatrix(source, target, method, 500)
    # learnProjection(source, target)
    # evaluate_SA(source, target, False,1,500)
    # evaluate_SA(source, target, True, 500)
    methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi"]
    methods += ["ppmi",'un_ppmi']
    # methods = ["freq"]
    # methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove","landmark_wiki_ppmi"]
    # methods = ['landmark_wiki_ppmi']
    # n = 500
    for method in methods:
        batchEval(method,1)
    # gammas = [1,5,10,20,50,100,1000]
    # for method in methods:
    #     choose_gamma(source, target, method,gammas,n)
    # params = [0,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    # params += [10e-3,10e-4,10e-5,10e-6]
    # params = [10e-4]
    # params.sort()
    # params = [0,1,50,100,1000,10000]
    # params = [0,10e-3,0.2,0.4,0.6,0.8,1]
    # for method in methods:
    #     choose_param(method,params,1)