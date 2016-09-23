import numpy as np 
from tabulate import tabulate

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

# convert names
def convert(method):
    if "landmark_" in method:
        if "_ppmi" in method:
            return "%s+PPMI" % method.replace("_ppmi","")
        else:
            return method
    else:
        if "un_" in method:
            return "%s$_U$" % method.replace("un_","").upper()
        else:
            return "%s$_L$" % method.upper()

def construct_accuracy_table(pv_methods,da_method):
    table = []
    i = 0
    for method in pv_methods:
        if i == 0:
            m_list = collect_accuracy("../work/batch%s.%s.csv"% (da_method, method))
            pairs = [x[0] for x in m_list]
        i+=1

    for pair in pairs:
        tmp = []
        tmp.append(pair)
        for method in pv_methods:
            m_list = collect_accuracy("../work/batch%s.%s.csv"% (da_method, method))
            tmp.append([x[2] for x in m_list if x[0]==pair][0])
            # print tmp
        table.append(tmp)

    headers = [da_method]+[convert(x) for x in pv_methods]
    print tabulate(table,headers)
    # print tabulate(table,headers,tablefmt="latex")
    pass
if __name__ == "__main__":
    methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi"]
    # DAmethod = "SCL"
    DAmethod = "SFA"
    construct_accuracy_table(methods,DAmethod)