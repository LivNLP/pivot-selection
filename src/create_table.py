import numpy as np 
import heapq
from tabulate import tabulate
from decimal import *
import scipy.stats

# same method used for draw_barchart
def collect_accuracy(csv_file):
    new_list = []

    input_file = open (csv_file,'r')
    next(input_file)

    for line in input_file:
        p = line.strip('\n').split(',')
        src = p[0][0].capitalize()
        tgt = p[1].replace(" ", "")[0].capitalize()
        pair = "%s-%s"%(src,tgt)
        method = p[2].replace(" ", "")
        acc = float(p[3])*100
        interval = (float(p[5]) - float(p[4]))*100/2.0
        # print p
        new_list.append((pair,method,acc,interval))

    return new_list
    pass

def collect_params(da_method,pv_method):
    new_list = []
    input_file = open("../work/sim/%sparams.%s.csv"%(da_method,pv_method),'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        src = p[0][0].capitalize()
        tgt = p[1][0].capitalize()
        domain_pair = "%s-%s"%(src,tgt)
        method = pv_method
        acc = float(p[3])*100
        interval = (float(p[5]) - float(p[4]))*100/2.0
        param = '%.1f' % float(p[6]) if (float(p[6])>0.1 or float(p[6])==0) else '%.1e'%Decimal(p[6])
        new_list.append((domain_pair,method, acc,interval,param))
    return new_list
    pass

def collect_params_on_server(da_method,pv_method):
    new_list = []
    input_file = open("../work/temp/%sparams.%s.csv"%(da_method,pv_method),'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        src = p[0][0].capitalize()
        tgt = p[1][0].capitalize()
        domain_pair = "%s-%s"%(src,tgt)
        method = pv_method
        acc = float(p[3])*100
        interval = (float(p[5]) - float(p[4]))*100/2.0
        param = '%.1f' % float(p[6]) if (float(p[6])>0.1 or float(p[6])==0) else '%.1e'%Decimal(p[6])
        new_list.append((domain_pair,method, acc,interval,param))
    return new_list
    pass


# convert names
def convert(method):
    if "landmark_" in method: 
        if method.replace("_pretrained","").replace("landmark_","") == "word2vec":
            return "T-CBOW"
        elif method.replace("_pretrained","").replace("landmark_","") == "glove":
            return "T-GloVe"
        else:
            return "T-Wiki"
    else:
        if "un_" in method:
            return "%s$_U$" % method.replace("un_","").upper()
        else:
            return "%s$_L$" % method.upper()



def construct_best_landmark(param_list):
    new_list = []
    domain_pairs = list(set([p[0] for p in param_list]))
    domain_pairs.sort()

    for pair in domain_pairs:
        best =  max([p[2] for p in param_list if p[0]==pair])
        tmp = [p for p in param_list if (p[0]==pair and p[2]==best)][0]
        new_list.append(tmp)

    # print new_list
    return new_list
    pass 


def construct_accuracy_table(pv_methods,da_method):
    table = []
    table_nobest = []
    i = 0
    # collect domain pairs
    for method in pv_methods:
        if i == 0:
            m_list = collect_accuracy("../work/batch%s.%s.csv"% (da_method, method))
            pairs = [x[0] for x in m_list]
        i+=1

    # use the pairs to collect the accuracies
    for pair in pairs:
        tmp = []
        tmp.append(pair)
        for method in pv_methods:
            if "landmark" in method:
                m_list = construct_best_landmark(collect_params(da_method,method))
                tmp.append([x[2] for x in m_list if x[0]==pair][0])
            else:
                m_list = collect_accuracy("../work/batch%s.%s.csv"% (da_method, method))
                tmp.append([x[2] for x in m_list if x[0]==pair][0])
            # print tmp
        best = max(tmp[1:])
        best_idx = [i for i, j in enumerate(tmp) if j == best]
        # second_best = heapq.nlargest(2,tmp[1:])[1]
        # new_tmp = ["%.2f*"%x if (x>second_best+5 and x==best) else x for x in tmp[1:]]
        # highlight the best result
        new_tmp = ["\\textbf{%.2f}"%x if x == best else x for x in tmp]

        print pair,[convert(pv_methods[i-1]) for i in best_idx],best
        table.append(new_tmp)
        table_nobest.append(tmp)
        # print table

    avg_list = []
    a = []
    b = []
    for i in range(1,len(pv_methods)+1):
        tmp = [x[i] for x in table_nobest]
        # print tmp
        avg_list.append(np.mean(tmp))
    print avg_list
    avg_idx = heapq.nlargest(3,avg_list)
    for idx,x in enumerate(avg_idx):
        i = avg_list.index(x)
        tmp = [x[i+1] for x in table_nobest]
        if idx == 0:
            a = tmp
            print convert(pv_methods[i]),
        if idx == 2:
            b = tmp
            print convert(pv_methods[i]),
    p = scipy.stats.wilcoxon(a,b).pvalue
    print p
    if p < 0.01:
        print '*YES'
    avg_list = ['avg']+avg_list 
    table.append(avg_list)

    headers = [da_method]+[convert(x) for x in pv_methods]
    print tabulate(table,headers,floatfmt=".2f")
    # print tabulate(table,headers,tablefmt="latex")
    pass

def construct_SCL_table(pv_methods):
    table = []
    i = 0
    #collect domain pairs
    for method in pv_methods:
        if i == 0:
            m_list = collect_accuracy("../work/batchSCL.%s.csv"%method)
            pairs = [x[0] for x in m_list]
        i+=1

    # use pairs to collect accuracies
    for pair in pairs:
        tmp = []
        tmp.append(pair)
        for method in pv_methods:
            if "landmark" in method:
                m_list = construct_best_landmark(collect_params('SCL',method))
                tmp.append([x[2] for x in m_list if x[0]==pair][0])
                m_list = construct_best_landmark(collect_params_on_server('SCL',method))
                tmp.append([x[2] for x in m_list if x[0]==pair][0])
            else:
                m_list = collect_accuracy("../work/batch%s.%s.csv"% ('SCL', method))
                tmp.append([x[2] for x in m_list if x[0]==pair][0])
                m_list = collect_accuracy("../work/temp/batch%s.%s.csv"% ('SCL', method))
                tmp.append([x[2] for x in m_list if x[0]==pair][0])
        best = max(tmp[1:])
        best_idx = [i for i, j in enumerate(tmp) if j == best]
        new_tmp = ["\\textbf{%.2f}"%x if x == best else x for x in tmp]
        # print pair,[convert(pv_methods[i-1]) for i in best_idx],best
        table.append(new_tmp)

    headers = ['S-T']+sum([[convert(x)+'(S,T)',convert(x)+'(T)'] for x in pv_methods],[])
    print tabulate(table,headers,floatfmt=".2f")
    pass

def filter_num(tmp):
    return filter(lambda i: isinstance(i, float), tmp)


if __name__ == "__main__":
    methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi","ppmi","un_ppmi"]
    methods += ["landmark_pretrained_word2vec","landmark_pretrained_glove","landmark_wiki_ppmi"]
    # DAmethod = "SCL"
    DAmethod = "SFA"
    construct_accuracy_table(methods,DAmethod)
    # construct_SCL_table(methods)