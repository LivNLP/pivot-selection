import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
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
        param = '%.1f' % float(p[6])
        new_list.append([domain_pair,acc,interval,param])

    print new_list
    return new_list
    pass

def drawer(argmts,pv_method,da_method):
    domain_pairs,ys,intervals,x = argmts
    fig, ax = plt.subplots(figsize=(9,6))
    index = np.arange(len(x))
    markers = ['.','x']*(len(domain_pairs)/2)
    i =0
    for y in ys:
        plt.errorbar(index,y,marker= markers[i],alpha=opacity, yerr=intervals[i],label=domain_pairs[i])
        i += 1
    plt.xticks(index,x)

    plt.title(da_method+ ": " +pv_method.capitalize())
    plt.xlabel('Lambda',size=18)
    plt.ylabel('Accuracy',size=18)
    #right box
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
          fancybox=True, shadow=True, ncol=1)
    plt.show()
    pass

def constructer(param_list):
    ys = []
    yerrs = []
    domain_pairs = list(set([p[0] for p in param_list]))
    domain_pairs.sort()
    x = list(set([p[3] for p in param_list]))
    x.sort()

    for pair in domain_pairs:
        ys.append([tmp[1] for tmp in param_list if tmp[0] == pair])
        yerrs.append([tmp[2] for tmp in param_list if tmp[0] == pair])

    print ys
    print yerrs
    print x
    return domain_pairs,ys,yerrs,x
    pass


if __name__ == "__main__":
    # pv_method = "landmark_pretrained_glove"
    pv_method = "landmark_pretrained_word2vec"
    da_method = 'SFA'
    # da_method = 'SCL'
    temp = collecter(da_method,pv_method)
    
    drawer(constructer(temp),pv_method,da_method)