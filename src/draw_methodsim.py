import numpy as np
import matplotlib.pyplot as plt
opacity = 0.6

def collector(pv_method, lookfor_pair):
    new_list = []
    input_file = open("../work/sim/MethodSim.%s.csv"%pv_method,'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        m1 = p[2].upper()
        m2 = p[3].upper()
        method_pair = (m1,m2)
        method_pair_re = (m2,m1)
        if lookfor_pair == method_pair or lookfor_pair == method_pair_re:
            src = p[0][0].capitalize()
            tgt = p[1][0].capitalize()
            domain_pair = "%s-%s"%(src,tgt)
            jaccard = float(p[4])
            n_pivots = int(float(p[6]))
            new_list.append([jaccard,n_pivots,domain_pair])

    # print new_list
    return new_list
    pass

def collector_margnal(pv_method,lookfor_pair,max_end):
    new_list = []
    input_file = open("../work/sim/MethodSim_range.%s.csv"%pv_method,'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        if float(p[6]) <= max_end:
            m1 = p[2].upper()
            m2 = p[3].upper()
            method_pair = (m1,m2)
            method_pair_re = (m2,m1)
            if lookfor_pair == method_pair or lookfor_pair == method_pair_re:
                src = p[0][0].capitalize()
                tgt = p[1][0].capitalize()
                domain_pair = "%s-%s"%(src,tgt)
                jaccard = float(p[4])
                pv_start = int(float(p[5]))
                pv_end = int(float(p[6]))
                pv_range = "%s-%s"%(pv_start,pv_end)
                new_list.append([jaccard,pv_range,domain_pair])
    return new_list
    pass

def drawer(argmts):
    x,ys,domain_pairs = argmts
    fig, ax = plt.subplots(figsize=(9,6))
    index = np.arange(len(x))
    i = 0
    for y in ys:
        plt.plot(index,y, marker="o", alpha=opacity, label=domain_pairs[i])
        i += 1

    plt.title(convert_title(lookfor_pair,pv_method),size=22)        
    plt.xlabel('$k$(#pivots)',size=18)
    plt.xticks(index,x)
    plt.ylabel('$J(M_1,M_2)$',size=18)
    #right box
    # box = ax.get_position()
    # ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    # ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
    #       fancybox=True, shadow=True, ncol=1)
    plt.show()
    pass

def drawer_margnal(argmts):
    x,ys,domain_pairs = argmts
    fig, ax = plt.subplots(figsize=(9,6))
    index = np.arange(len(x))
    i = 0
    for y in ys:
        plt.plot(index,y, marker="o", alpha=opacity, label=domain_pairs[i])
        i += 1

    plt.title(convert_title(lookfor_pair,pv_method),size=22)
    plt.xlabel('pivot range',size=18)
    plt.xticks(index,x)
    plt.ylabel('$J(M_1,M_2)$',size=18)
    #right box
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
          fancybox=True, shadow=True, ncol=1)
    plt.show()
    pass      

def constructer(sim_list):
    # sim_list.sort(lambda x,y:-1 if x[2]<y[2] else 1)
    ys = []
    domain_pairs = list(set([p[2] for p in sim_list]))
    domain_pairs.sort()
    x = list(set([p[1] for p in sim_list]))
    x.sort()

    for pair in domain_pairs:
        ys.append([tmp[0] for tmp in sim_list if tmp[2] == pair])

    # print ys
    # print sim_list
    # print domain_pairs
    # print x
    return x,ys,domain_pairs
    pass

def convert_title(lookfor_pair,pv_method):
    if pv_method == 'landmark':
        return "%s vs %s" % (lookfor_pair[0],lookfor_pair[1])
    else:
        return "%s$_%s$ vs %s$_%s$" % (lookfor_pair[0],pv_method,lookfor_pair[1],pv_method)

if __name__ == "__main__":
    # m1 = "freq"
    # m2 = "mi"
    # pv_method = "L"
    # m1 = "landmark_word2vec"
    # m2 = "landmark_word2vec_ppmi"
    m1 = "landmark_glove"
    m2 = "landmark_glove_ppmi"
    # m2 = "landmark_glove"
    pv_method = "landmark"
    lookfor_pair = (m1.upper(),m2.upper())
    # lookfor_pair = (m1,m2)
    # drawer(constructer(collector(pv_method, lookfor_pair)))
    drawer_margnal(constructer(collector_margnal(pv_method, lookfor_pair, 500)))
    
