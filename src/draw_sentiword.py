import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from decimal import *
import itertools
import math
# font = {  'size'   : 18}
# matplotlib.rc('font', **font)
matplotlib.rc('ytick', labelsize=22) 
plt.legend(loc=2,prop={'size':22})
opacity = 0.6

def collect_sentiword(methods,lookfor_pair):
    new_list = []
    for method in methods:
        if 'landmark' in method:
            input_file = open("../work/sim/Sentiparams.%s.csv"%method,'r')
        else:
            input_file = open("../work/sim/Sentisim.csv",'r')
        next(input_file)
        for line in input_file:
            p = line.strip('\n').split(', ')
            src = p[0][0].capitalize()
            tgt = p[1][0].capitalize()
            pair = "%s-%s"%(src,tgt)
            if lookfor_pair == pair:    
                pos = float(p[3])
                neg = float(p[4])
                mid = float(p[5])
                senti_percentage = senti_bearing(pos,neg,mid)
                if 'landmark' in method:
                    param = float(p[6])
                    new_list.append([pair,method,senti_percentage,param])
                else :
                    if p[2]==method:
                        new_list.append([pair,method,senti_percentage])
    # print new_list
    return new_list,methods

def collect_sentiword_params(method):
    new_list = []
    input_file = open("../work/sim/Sentiparams.%s.csv"%method,'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        src = p[0][0].capitalize()
        tgt = p[1][0].capitalize()
        pair = "%s-%s"%(src,tgt)   
        pos = float(p[3])
        neg = float(p[4])
        mid = float(p[5])
        senti_percentage = senti_bearing(pos,neg,mid)
        param = '%.1f' % float(p[6]) if (float(p[6])>0.1 or float(p[6])==0) else '%.1e'%Decimal(p[6])
        new_list.append([pair,method,senti_percentage,param])
# print new_list
    return new_list,method


def senti_bearing(pos,neg,mid):
    return float((pos+neg)/(pos+neg+mid))*100

def drawer(argmts,lookfor_pair):
    methods,ys,x = argmts
    fig, ax = plt.subplots(figsize=(12,10))
    index = np.arange(len(x))
    markers = [6,7]*(len(methods)/2)
    i = 0
    for y in ys:
        plt.errorbar(index,y,marker= markers[i],alpha=opacity, label=convert(methods[i]))
        i += 1
    plt.xticks(index,x, size = 22)

    plt.title(lookfor_pair+': SentiWordNet',size=22)
    plt.xlabel('$\\lambda$',size=22)
    plt.ylabel('Sentiment Bearing Pivots',size=22)
    # bottom box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
    # plt.show()
    plt.autoscale()
    plt.savefig(lookfor_pair+'-senti.png')
    pass

def drawer_params(argmts,pv_method):
    domain_pairs,ys,x = argmts
    fig, ax = plt.subplots(figsize=(12,6))
    index = np.arange(len(x))
    markers = ['.','x']*(len(domain_pairs)/2)
    i =0
    for y in ys:
        plt.errorbar(index,y,marker= markers[i],alpha=opacity,label=domain_pairs[i])
        i += 1
    plt.xticks(index,x)

    plt.title(convert(method),size=18)
    plt.xlabel('$\\lambda$',size=18)
    plt.ylabel('Sentiment Bearing Pivots',size=18)
    #right box
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
          fancybox=True, shadow=True, ncol=1)
    # plt.show()
    plt.autoscale()
    plt.savefig(convert(pv_method)+'-senti.png')
    pass

# convert names
def convert(method):
    if "landmark_" in method:
        if "_ppmi" in method:
            return "%s+PPMI" % method.replace("_ppmi","").replace("_pretrained","").replace("landmark_","")
        else:
            if method.replace("_pretrained","").replace("landmark_","") == "word2vec":
                return "S-CBOW"
            else:
                return "S-GloVe"
    else:
        if "un_" in method:
            return "%s$_U$" % method.replace("un_","").upper()
        else:
            return "%s$_L$" % method.upper()



def construct_senti_figure(argmts):
    param_list,methods = argmts
    ys = []
    x = list(set([p[3] for p in param_list if len(p)>3]))
    x.sort(key=float)
    # print x

    for method in methods:
        if 'landmark' in method:
            y = []
            for param in x:
                y.append([p[2] for p in param_list if (p[1]==method and len(p)>3 and p[3] == param)][0])
            ys.append(y)
        else:
            ys.append([p[2] for p in param_list if p[1]==method]*len(x))

    x = ['%.1f'%tmp if (tmp>0.1 or tmp==0) else '$10^{%d}$'%(math.log10(tmp)-1) for tmp in x]
    # print x
    # print ys
    # print methods
    return methods,ys,x

def construct_senti_params(argmts):
    param_list,method = argmts
    ys = []
    domain_pairs = list(set([p[0] for p in param_list]))
    domain_pairs.sort()

    x = list(set([p[3] for p in param_list]))
    x.sort(key=float)

    for pair in domain_pairs:  
        ys.append([tmp[2] for tmp in param_list if (p[1]==method and tmp[0] == pair)])

    # print ys
    # print x
    # print domain_pairs
    return domain_pairs,ys,x

def loop_pairs():
    domains = ['B','D','E','K']
    all_pairs = list(itertools.permutations(domains, 2))
    pairs = ['%s-%s'%(i,j) for (i,j) in all_pairs]
    return pairs

def draw(methods,lookfor_pair):
    drawer(construct_senti_figure(collect_sentiword(methods,lookfor_pair)),lookfor_pair)
    pass

def draw_params(method):
    drawer_params(construct_senti_params(collect_sentiword_params(method)),method)
    pass


if __name__ == "__main__":
    # method = "landmark_pretrained_word2vec"
    # method = "landmark_pretrained_glove"
    # draw_params(method)
    # lookfor_pair = "B-D"
    # lookfor_pair = "K-E"
    pairs = loop_pairs()
    for lookfor_pair in pairs:
        methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove"]
        methods += ["freq","un_freq","mi","un_mi","pmi","un_pmi","ppmi","un_ppmi"]
        draw(methods,lookfor_pair)