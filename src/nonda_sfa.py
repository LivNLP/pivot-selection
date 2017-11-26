"""
Forked Code from Danushka Bollegala
Implementation of SFA following steps after pivot selection
Used for evaluation of pivot selection methods
------------------------------------------
changelog: add more datasets and for non-da datasets.
"""
import sys,time
import math
import numpy as np
import scipy.io as sio 
import scipy.sparse as sp
# from sparsesvd import sparsesvd
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

def createMatrix(dataset, method, n):
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

    
    # Load features
    fname = "../work/%s/obj/freq" % (dataset)
    if "un_" in method:
        fname = "../work/%s/obj/un_freq" % (dataset)
    features = pi.load_stored_obj(fname)
    V = dict(features)
    
    # Compute the co-occurrences of features in reviews
    M = {}
    print "Vocabulary size =", len(V)
    getCounts(V, M, "../data/%s/train-sentences" % dataset)
    print "%s train %d" % (dataset, len(M)) 

    # Remove co-occurrence less than the coocTh
    M = selectTh(M, coocTh)

    print "selecting top-%d features in %s as pivots" % (n, method)
    pivotsFile = "../work/%s/obj/%s" % (dataset, method)
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
    DSFile = open("../work/%s/DS_list" % (dataset), 'w')
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
    sio.savemat("../work/%s/DSxDI.mat" % (dataset), {'DSxDI':R})
    print "Done"
    pass

def learnProjection(dataset):
    """
    Learn the projection matrix and store it to a file. 
    """
    h = 50 # no. of latent dimensions.
    print "Loading the bipartite matrix...",
    coocData = sio.loadmat("../work/%s/DSxDI.mat" % (dataset))
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
    # Perform SVD on B
    print "Perform SVD on the weight matrix..."
    startTime = time.time()
    # ut, s, vt = sparsesvd(B.tocsc(), h)
    B = sp.csc_matrix(B, dtype=float)
    ut, s, vt = sp.linalg.svds(B, h)
    print ut.shape
    endTime = time.time()
    print "%ss" % str(round(endTime-startTime, 2))    
    sio.savemat("../work/%s/proj_sfa.mat" % (dataset), {'proj':ut.T})
    print "Done."    
    pass


def evaluate_SA(dataset, project,gamma, n):
    """
    Report the cross-domain sentiment classification accuracy. 
    """
    # gamma = 1.0
    if project:
        print "Projection ON", "Gamma = %f" % gamma
    else:
        print "Projection OFF"
    # Load the projection matrix.
    M = sp.csr_matrix(sio.loadmat("../work/%s/proj_sfa.mat" % (dataset))['proj'])
    (nDS, h) = M.shape
    # Load the domain specific features.
    DSfeat = {}
    DSFile = open("../work/%s/DS_list" % (dataset))
    for line in DSFile:
        p = line.strip().split()
        DSfeat[p[1].strip()] = int(p[0])
    DSFile.close()
    
    
    # write train feature vectors.
    trainFileName = "../work/%s/trainVects.SFA" % (dataset)
    testFileName = "../work/%s/testVects.SFA" % (dataset)
    featFile = open(trainFileName, 'w')
    count = 0
    F = open("../data/%s/train" % dataset)
    for line in F:
        count += 1
        #print "Train ", count
        words = [word.replace(":1","") for word in set(line.strip().split()[1:])]
        # write the original features.
        featFile.write("%d " % int(line.strip().split()[0]))
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
    F = open("../data/%s/test" % dataset)
    for line in F:
        count += 1
        #print "Test ", count
        words = [word.replace(":1","") for word in set(line.strip().split()[1:])]
        # write the original features.
        featFile.write("%d " % int(line.strip().split()[0]))
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
    modelFileName = "../work/%s/model.SFA" % (dataset)
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
    resFile = open("../work/nonDA-batchSFA.%s.csv"% method, "w")
    domains = ["TR", "CR", "SUBJ","MR"]
    numbers = [500]
    resFile.write("dataset, Method, Acc, IntLow, IntHigh,#pivots\n")
    for dataset in domains:
        for n in numbers:
            createMatrix(dataset, method, n)
            learnProjection(dataset)
            evaluation = evaluate_SA(dataset, True, gamma, n)
            resFile.write("%s, %s, %f, %f, %f, %f\n" % (dataset, method, evaluation[0], evaluation[1][0],evaluation[1][1],n))
            resFile.flush()
    resFile.close()
    pass

def choose_gamma(dataset, method, gammas, n):
    resFile = open("../work/gamma/%s/SFAgamma.%s.csv"% (dataset, method), "w")
    resFile.write("dataset, Method, NoProj, Proj, Gamma\n")
    createMatrix(dataset, method, n)
    learnProjection(dataset)
    for gamma in gammas:    
        resFile.write("%s, %s,  %f, %f, %f\n" % (dataset, method, 
        evaluate_SA(dataset, False, gamma, n), evaluate_SA(dataset, True, gamma, n), gamma))
        resFile.flush()
    resFile.close()
    pass

def choose_param(method,params,gamma):
    domains = ["TR", "CR", "SUBJ","MR"]
    numbers=[500]
    resFile = open("../work/sim/SFAparams.%s.csv"% method, "w")
    resFile.write("dataset, Model, Acc, IntLow, IntHigh, Param\n")
    for param in params:
        test_method = "test_%s_%f"% (method,param)
        for dataset in domains:
            for n in numbers:
                createMatrix(dataset, test_method, n)
                learnProjection(dataset)
                evaluation = evaluate_SA(dataset, True, gamma, n)
                resFile.write("%s, %s, %f, %f, %f, %f, %f\n" % (dataset, method, evaluation[0], evaluation[1][0],evaluation[1][1],param,n))
                resFile.flush()
    resFile.close()
    pass


if __name__ == "__main__":
    # methods = ["un_freq","un_mi","un_pmi","un_ppmi"]
    methods = ["un_ppmi"]
    # methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove","landmark_wiki_ppmi"]
    # methods = ['landmark_wiki_ppmi']
    # n = 500
    for method in methods:
        batchEval(method,1)
    # gammas = [1,5,10,20,50,100,1000]
    # for method in methods:
    #     choose_gamma(dataset, method,gammas,n)
    # params = [0,0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    # params += [10e-3,10e-4,10e-5,10e-6]
    # params = [10e-4]
    # params.sort()
    # params = [0,1,50,100,1000,10000]
    # params = [0,10e-3,0.2,0.4,0.6,0.8,1]
    # for method in methods:
    #     choose_param(method,params,1)