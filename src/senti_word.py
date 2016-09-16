
import numpy 
from nltk.corpus import sentiwordnet as swn


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


# main
if __name__ == "__main__":
    feats = ['happy','what','very__disappointed','bad']
    senti_list(feats)