import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler
from tabulate import tabulate
from decimal import *
import itertools
import math
opacity = 0.6
matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 

def collecter(da_method,pv_method):
    new_list = []
    input_file = open("../work/sim/f-%sparams.%s.csv"%(da_method,pv_method),'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        src = p[0][0].capitalize()
        tgt = p[1][0].capitalize()
        domain_pair = "%s-%s"%(src,tgt)
        acc = float(p[3])*100
        interval = (float(p[5]) - float(p[4]))*100/2.0
        param = float(p[6])
        n_pivots = int(float(p[7])) # number of pivots
        new_list.append([domain_pair,acc,interval,param,n_pivots])

    # print new_list
    return new_list

def collect_methods(da_method,methods,lookfor_pair):
    new_list = []
    for method in methods:
        if 'landmark' in method:
            input_file = open("../work/sim/f-%sparams.%s.csv"%(da_method,method),'r')
        # else:
        #     input_file = open("../work/batch%s.%s.csv"%(da_method,method),'r')
        next(input_file)
        for line in input_file:
            p = line.strip('\n').split(', ')
            src = p[0][0].capitalize()
            tgt = p[1][0].capitalize()
            pair = "%s-%s"%(src,tgt)
            # print p
            if lookfor_pair == pair:
                acc = float(p[3])*100
                interval = (float(p[5]) - float(p[4]))*100/2.0
                if 'landmark' in method:
                    param = float(p[6])
                    n_pivots = int(float(p[7])) # number of pivots
                    new_list.append([pair,method,acc,interval,param,n_pivots])
                # else:
                #     if p[2] == method:
                #         new_list.append([pair,method,acc,interval])
        # print new_list,methods,lookfor_pair
    return new_list,methods,lookfor_pair


def draw_methods(argmts,da_method):
    methods,ys,yerrs,x,lookfor_pair = argmts
    fig, ax = plt.subplots(figsize=(12,8))
    index = np.arange(len(x))
    markers = ['.','x','o']*(len(methods)/3)
    # print methods
    i = 0
    # print index,[len(y) for y in ys]
    for y in ys: #yerr=yerrs[i]
        plt.errorbar(index,y,marker= markers[i],alpha=opacity,label=convert(methods[i]),mew=3,linewidth=3.0,markersize=10)
        i += 1
    plt.xticks(index,x)

    plt.title(lookfor_pair+': '+da_method,size=22)
    plt.xlabel('$k$(#pivots)',size=22)
    plt.ylabel('Accuracy',size=22)
    # bottom box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
    
    plt.autoscale()
    plt.ylim([57,90])
    plt.show()
    # plt.savefig('%s:%s-acc.png'%(lookfor_pair,da_method))
    pass


def construct_methods(argmts):
    param_list,methods,lookfor_pair = argmts
    ys = []
    yerrs = []
    # get number of pivots to be x
    x = list(set([p[5] for p in param_list if len(p)>5]))
    x.sort(key=float)
    # print x 

    for method in methods:
        if 'landmark' in method:
            y = []
            yerr = []
            for n_pivots in x:
                y.append([p[2] for p in param_list if (p[1]==method and len(p)>5 and p[5]==n_pivots)][0])
                yerr.append([p[3] for p in param_list if (p[1]==method and len(p)>5 and p[5]==n_pivots)][0])
            ys.append(y)
            yerrs.append(yerr)
        # else:
        #     ys.append([p[2] for p in param_list if p[1]==method]*len(x))
        #     yerrs.append([p[3] for p in param_list if p[1]==method]*len(x))

    # x = ['%.1f'%tmp if (tmp>0.1 or tmp==0) else '$10^{%d}$'%(math.log10(tmp)-1) for tmp in x]
    # print ys
    # print yerrs
    return methods,ys,yerrs,x,lookfor_pair


# convert names
def convert(method):
    if "landmark_" in method: 
        if method.replace("_pretrained","").replace("landmark_","") == "word2vec":
            return "T-CBOW"
        elif method.replace("_pretrained","").replace("landmark_","") == "glove":
            return "T-GloVe"
        else:
            return "Wiki-PPMI"
    else:
        if "un_" in method:
            return "%s$_U$" % method.replace("un_","").upper()
        else:
            return "%s$_L$" % method.upper()

def draw_methods_figure(da_method,methods,lookfor_pair):
    temp = collect_methods(da_method,methods,lookfor_pair)
    draw_methods(construct_methods(temp),da_method)
    pass

if __name__ == "__main__":
    da_method = "SFA"
    pv_method = "landmark_wiki_ppmi"
    methods = ["landmark_pretrained_word2vec"]
    methods += ["landmark_pretrained_glove"]
    methods += ['landmark_wiki_ppmi']
    lookfor_pair = "D-B"
    # collecter(da_method,pv_method)
    draw_methods_figure(da_method,methods,lookfor_pair)