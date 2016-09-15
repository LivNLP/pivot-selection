
import numpy 
from nltk.corpus import sentiwordnet as swn


# sentiwordnet score
# sum up all the scores and take the average
def senti_score(feature):
    temp = swn.senti_synsets(feature)
    return numpy.mean([x.pos_score()-x.neg_score() for x in temp]) if temp else 0

# return number of neutral,pos,neg within a list
def senti_list(feats):
    mid = 0
    pos = 0
    neg = 0
    for x in feats:
        if senti_score(x) == 0:
            mid += 1
        else:
            if senti_score(x) > 0:
                pos += 1
            else:
                if senti_score(x) < 0:
                    neg += 1
    print 'neutral = %d, positive = %d, negative = %d'%(mid,pos,neg)
    return mid,pos,neg 


# main
if __name__ == "__main__":
    feats = ['happy','what','word','bad']
    senti_list(feats)