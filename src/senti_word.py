
import numpy 
from nltk.corpus import sentiwordnet as swn
import select_pivots as pi

# sentiwordnet score
# sum up all the scores and take the average
def senti_score(feature):
    score = 0
    for t in feature.split('__'):
        temp = swn.senti_synsets(t)
        score+=numpy.mean([x.pos_score()-x.neg_score() for x in temp]) if temp else 0
    return score
    
# return number of neutral,pos,neg within a list
def senti_list(feats):
    mid = sum(1 for x in feats if senti_score(x) == 0 )
    pos = sum(1 for x in feats if senti_score(x) > 0 )
    neg = sum(1 for x in feats if senti_score(x) < 0 )
    print 'neutral = %d, positive = %d, negative = %d'%(mid,pos,neg)
    return mid,pos,neg 

def choose_param(method,params,n):
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile = open("../work/sim/Sentiparams.%s.csv"% method, "w")
    resFile.write("Source, Target, Model, #Positive, #Negative, #Neutral, Param\n")
    for param in params:
        test_method = "test_%s_%f"% (method,param)
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                pivotsFile = "../work/%s-%s/obj/%s" % (source, target, method)
                features = pi.load_stored_obj(pivotsFile)
                mid,pos,neg = senti_list(dict(features[:n]).keys())
                resFile.write("%s, %s, %s, %f, %f, %f, %f\n"%(source,target,method,pos,neg,mid,param))
                resFile.flush()
    resFile.close()
    pass

# main
if __name__ == "__main__":
    methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove"]
    n = 1000
    # params = [1,50,100,1000,10000]
    params = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    for method in methods:
        choose_param(method,params,n)   
    #######test#########
    # feats = ['happy','what','very__disappointed','bad']
    # senti_list(feats)