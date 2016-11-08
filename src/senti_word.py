
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
    # print 'neutral = %d, positive = %d, negative = %d'%(mid,pos,neg)
    return mid,pos,neg 

# different lambdas for single word embedding model
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
                pivotsFile = "../work/%s-%s/obj/%s" % (source, target, test_method)
                features = pi.load_stored_obj(pivotsFile)
                mid,pos,neg = senti_list(dict(features[:n]).keys())
                # print "dir: %s, %d,%d,%d"%(pivotsFile,mid,pos,neg)
                resFile.write("%s, %s, %s, %f, %f, %f, %f\n"%(source,target,method,pos,neg,mid,param))
                resFile.flush()
    resFile.close()
    pass

# diffferent pv methods
def method_eval(methods,n):
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile = open("../work/sim/Sentisim.csv", "w")
    resFile.write("Source, Target, Method, #Positive, #Negative, #Neutral\n")
    for method in methods:
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                pivotsFile = "../work/%s-%s/obj/%s" % (source, target, method)
                features = pi.load_stored_obj(pivotsFile)
                mid,pos,neg = senti_list(dict(features[:n]).keys())
                print "method: %s, %d,%d,%d"%(method,mid,pos,neg)
                resFile.write("%s, %s, %s, %f, %f, %f\n"%(source,target,method,pos,neg,mid))
                resFile.flush()
    resFile.close()
    pass

def get_best_k_with_score(feats):
    temp = ""
    for x in feats:
        temp+=  "%s%s "%(x.replace('__','+'),symbol(senti_score(x)))
    return temp

def get_best_k(feats):
    for x in feats:
        temp += "%s "%x.replace('__','+')
    return temp


def symbol(score):
    return '(+)' if score > 0 else '(-)' if score < 0 else '(n)'

def create_top_k_table(methods,params,n,source,target):
    domain_pair = "%s-%s"%(source[0].capitalize(),target[0].capitalize())
    if params:
        resFile = open("../work/sim/top-%d %s.csv"%(n,domain_pair),"w")
        resFile.write("Lambda, S-CBOW, S-GloVe\n")
        for param in params:
            resFile.write('%f'%param)
            for method in methods:
                test_method = "test_%s_%f"% (method,param)
                pivotsFile = "../work/%s-%s/obj/%s" % (source, target, test_method)
                features = pi.load_stored_obj(pivotsFile)
                temp = get_best_k(dict(features[:n]).keys())
                # temp = get_best_k_with_score(dict(features[:n]).keys())
                # print temp
                resFile.write(', %s'%temp)
            resFile.write('\n')
            resFile.flush()
        resFile.close()
    else:
        resFile = open("../work/sim/top-%d %s others.csv"%(n,domain_pair),"w")
        for method in methods:
            pivotsFile = "../work/%s-%s/obj/%s" % (source, target, method)
            features = pi.load_stored_obj(pivotsFile)
            temp = get_best_k(dict(features[:n]).keys())
            # temp = get_best_k_with_score(dict(features[:n]).keys())
            # print temp
            resFile.write("%s, %s\n"%(convert(method),temp))
            resFile.flush()
        resFile.close()
    pass

def convert(method):
    if "un_" in method:
        return "%s$_U$" % method.replace("un_","").upper()
    else:
        return "%s$_L$" % method.upper()

# main
if __name__ == "__main__":
    methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove"]
    n = 500
    # # params = [0,1,50,100,1000,10000]
    params = [0,10e-3,10e-4,10e-5,10e-6]
    params += [0.1,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    params.sort()
    for method in methods:
        choose_param(method,params,n)
    # source = 'kitchen'
    # target = 'electronics'
    # source = 'books'
    # target = 'dvd'
    # params = []
    methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi","ppmi","un_ppmi"]
    # methods += ["landmark_pretrained_word2vec","landmark_pretrained_word2vec_ppmi","landmark_pretrained_glove","landmark_pretrained_glove_ppmi"]
    n = 500
    method_eval(methods,n)
    # n = 5
    # create_top_k_table(methods,params,n,source,target)
    #######test#########
    # feats = ['happy','what','very__disappointed','bad']
    # senti_list(feats)