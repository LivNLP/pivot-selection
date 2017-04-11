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

    print new_list
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
        print new_list,methods,lookfor_pair
    return new_list,methods,lookfor_pair

if __name__ == "__main__":
    da_method = "SFA"
    pv_method = "landmark_wiki_ppmi"
    methods = ["landmark_pretrained_word2vec"]
    methods += ["landmark_pretrained_glove"]
    methods += ['landmark_wiki_ppmi']
    lookfor_pair = "B-D"
    # collecter(da_method,pv_method)
    collect_methods(da_method,methods,lookfor_pair)