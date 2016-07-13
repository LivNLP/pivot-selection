import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def collector(csv_file):
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

    # print new_dict.values()
    return new_list
    pass

def drawer(methods,pairs,accuracy_all,interval_all,DAmethod):
    n_pairs = len(pairs)
    colors = ['b','g','r','c','m','y','k','w']
    fig, ax = plt.subplots(figsize=(15,5))

    index = np.arange(n_pairs)
    bar_width = 0.145
    opacity = 0.4
    err_config = {'ecolor':'0.3'}

    i = 0
    for method in methods:
        acc_m = accuracy_all[i]
        interval_m = interval_all[i]
        rect = plt.bar(index + bar_width * i, acc_m, bar_width, alpha=opacity, color=colors[i],
            yerr=interval_m, error_kw=err_config, label=convert(method))
        i += 1
    
    plt.xlabel('Domain Pairs')
    plt.ylabel('Accuracy',size=18)
    plt.title('%s'% DAmethod,size=22)
    plt.xticks(index + bar_width*i/2, pairs)
    # bottom box
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=6)
    plt.show()
    pass

def convert(method):
    if "un_" in method:
        return "%s$_U$" % method.replace("un_","").upper()
    else:
        return "%s$_L$" % method.upper()

def constructer(methods,DAmethod):
    accuracy_all={}
    interval_all={}
    i = 0
    for method in methods:
        m_list = collector("../work/batch%s.%s.csv"% (DAmethod, method))
        accuracy_all[i] = [x[2] for x in m_list]
        interval_all[i] = [x[3] for x in m_list]
        if i == 0:
            pairs = [x[0] for x in m_list]
        i+=1

    return pairs,accuracy_all.values(),interval_all.values()

if __name__ == "__main__":
    DAmethod = "SCL"
    # DAmethod = "SFA"
    methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi"]
    # constructer(methods,DAmethod)
    pairs,accuracy_all,interval_all = constructer(methods,DAmethod)
    drawer(methods,pairs,accuracy_all,interval_all,DAmethod)