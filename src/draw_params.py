import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from tabulate import tabulate
from decimal import *
opacity = 0.6

def collecter(da_method,pv_method):
    new_list = []
    input_file = open("../work/sim/%sparams.%s.csv"%(da_method,pv_method),'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        src = p[0][0].capitalize()
        tgt = p[1][0].capitalize()
        domain_pair = "%s-%s"%(src,tgt)
        acc = float(p[3])*100
        interval = (float(p[5]) - float(p[4]))*100/2.0
        param = '%.1f' % float(p[6]) if (float(p[6])>0.1 or float(p[6])==0) else '%.1e'%Decimal(p[6])
        new_list.append([domain_pair,acc,interval,param])

    # print new_list
    return new_list

def collect_methods(da_method,methods,lookfor_pair):
    new_list = []
    for method in methods:
        if 'landmark' in method:
            input_file = open("../work/sim/%sparams.%s.csv"%(da_method,method),'r')
        else:
            input_file = open("../work/batch%s.%s.csv"%(da_method,method),'r')
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
                    param = '%.1f' % float(p[6]) if (float(p[6])>0.1 or float(p[6])==0) else '%.1e'%Decimal(p[6])
                    new_list.append([pair,method,acc,interval,param])
                else:
                    if p[2] == method:
                        new_list.append([pair,method,acc,interval])
    return new_list,methods,lookfor_pair

def drawer(argmts,pv_method,da_method):
    domain_pairs,ys,intervals,x = argmts
    fig, ax = plt.subplots(figsize=(12,8))
    index = np.arange(len(x))
    markers = ['.','x']*(len(domain_pairs)/2)
    i =0
    for y in ys: #yerr=intervals[i]
        plt.errorbar(index,y,marker= markers[i],alpha=opacity,label=domain_pairs[i])
        i += 1
    plt.xticks(index,x)

    plt.title(da_method+ ": " +convert(pv_method))
    plt.xlabel('$\\lambda$',size=18)
    plt.ylabel('Accuracy',size=18)
    #right box
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
          fancybox=True, shadow=True, ncol=1)
    # plt.show()
    plt.autoscale()
    plt.savefig('%s:%s-acc.png'%(convert(pv_method),da_method))
    pass

def draw_methods(argmts,da_method):
    methods,ys,yerrs,x,lookfor_pair = argmts
    fig, ax = plt.subplots(figsize=(12,8))
    index = np.arange(len(x))
    markers = ['.','x']*(len(methods)/2)
    i = 0
    for y in ys: #yerr=yerrs[i]
        plt.errorbar(index,y,marker= markers[i],alpha=opacity,label=convert(methods[i]))
        i += 1
    plt.xticks(index,x)

    plt.title(lookfor_pair+': '+da_method,size=18)
    plt.xlabel('$\\lambda$',size=18)
    plt.ylabel('Accuracy',size=18)
    # bottom box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
          fancybox=True, shadow=True, ncol=5)
    # plt.show()
    plt.autoscale()
    plt.ylim([57,80])
    plt.savefig('%s:%s-acc.png'%(lookfor_pair,da_method))
    pass

def constructer(param_list):
    ys = []
    yerrs = []
    domain_pairs = list(set([p[0] for p in param_list]))
    domain_pairs.sort()
    x = list(set([p[3] for p in param_list]))
    x.sort(key=float)

    for pair in domain_pairs:
        ys.append([tmp[1] for tmp in param_list if tmp[0] == pair])
        yerrs.append([tmp[2] for tmp in param_list if tmp[0] == pair])

    # print ys
    # print yerrs
    # print x
    return domain_pairs,ys,yerrs,x

def construct_methods(argmts):
    param_list,methods,lookfor_pair = argmts
    ys = []
    yerrs = []
    x = list(set([p[4] for p in param_list if len(p)>4]))
    x.sort(key=float)

    for method in methods:
        if 'landmark' in method:
            y = []
            yerr = []
            for param in x:
                y.append([p[2] for p in param_list if (p[1]==method and len(p)>4 and p[4]==param)][0])
                yerr.append([p[3] for p in param_list if (p[1]==method and len(p)>4 and p[4]==param)][0])
            ys.append(y)
            yerrs.append(yerr)
        else:
            ys.append([p[2] for p in param_list if p[1]==method]*len(x))
            yerrs.append([p[3] for p in param_list if p[1]==method]*len(x))

    # print ys
    # print yerrs
    return methods,ys,yerrs,x,lookfor_pair

def construct_accuracy_table(param_list):
    table = []
    domain_pairs = list(set([p[0] for p in param_list]))
    domain_pairs.sort()
    params = list(set([p[3] for p in param_list]))
    params.sort(key=float)

    
    for pair in domain_pairs:
        tmp = []
        tmp.append(pair)
        for param in params:
           tmp.append([x[1] for x in param_list if (x[0] == pair and x[3]==param)][0])
        
        best = max(tmp[1:])
        best_idx = [i for i, j in enumerate(tmp) if j == best]
        new_tmp = ["\\textbf{%.2f}"%x if x == best else x for x in tmp]

        table.append(new_tmp)
    headers = ['$\\lambda$']+[str(x) for x in params]
    print tabulate(table,headers,floatfmt=".2f")
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


def draw_figure(da_method,pv_method):
    temp = collecter(da_method,pv_method)
    drawer(constructer(temp),pv_method,da_method)
    pass

def draw_methods_figure(da_method,methods,lookfor_pair):
    temp = collect_methods(da_method,methods,lookfor_pair)
    draw_methods(construct_methods(temp),da_method)
    pass

def draw_table(da_method,pv_method):
    temp = collecter(da_method,pv_method)
    print 'DA method = '+da_method
    print 'PV method = '+pv_method
    construct_accuracy_table(temp)
    pass

if __name__ == "__main__":
    # pv_method = "landmark_pretrained_glove"
    pv_method = "landmark_pretrained_word2vec"
    da_method = 'SFA'
    # da_method = 'SCL'
    # draw_figure(da_method,pv_method)
    # draw_table(da_method,pv_method)
    lookfor_pair = "B-D"
    methods = ["landmark_pretrained_word2vec","landmark_pretrained_glove"]
    methods += ["freq","un_freq","mi","un_mi","pmi","un_pmi","ppmi","un_ppmi"]
    draw_methods_figure(da_method,methods,lookfor_pair)
