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
            kendall = float(p[5])
            n_pivots = int(float(p[6]))
            new_list.append([jaccard,n_pivots,domain_pair,kendall])

    # print new_list
    return new_list
    pass

def collector_margnal(pv_method,lookfor_pair,max_end):
    new_list = []
    input_file = open("../work/sim/MethodSim_range.%s.csv"%pv_method,'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        if float(p[7]) <= max_end:
            m1 = p[2].upper()
            m2 = p[3].upper()
            method_pair = (m1,m2)
            method_pair_re = (m2,m1)
            if lookfor_pair == method_pair or lookfor_pair == method_pair_re:
                src = p[0][0].capitalize()
                tgt = p[1][0].capitalize()
                domain_pair = "%s-%s"%(src,tgt)
                jaccard = float(p[4])
                kendall = float(p[5])
                pv_start = int(float(p[6]))
                pv_end = int(float(p[7]))
                pv_range = "%s-%s"%(pv_start,pv_end)
                new_list.append([jaccard,pv_range,domain_pair,kendall])
    return new_list
    pass

def drawer(argmts,jaccard):
    x,ys,domain_pairs = argmts
    fig, ax = plt.subplots(figsize=(9,6))
    index = np.arange(len(x))
    markers = ['o','^']*(len(domain_pairs)/2)
    i = 0
    for y in ys:
        plt.plot(index,y, marker= markers[i], alpha=opacity, label=domain_pairs[i])
        i += 1

    plt.title(convert_title(lookfor_pair,pv_method),size=22)        
    plt.xlabel('$k$(#pivots)',size=18)
    plt.xticks(index,x)
    plt.ylabel('$J(M_1,M_2)$' if jaccard == True else '$K(M_1,M_2)$',size=18)
    #right box
    # box = ax.get_position()
    # ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    # ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
    #       fancybox=True, shadow=True, ncol=1)
    plt.show()
    pass

def drawer_margnal(argmts,jaccard):
    x,ys,domain_pairs = argmts
    fig, ax = plt.subplots(figsize=(9,6))
    index = np.arange(len(x))
    markers = ['o','^']*(len(domain_pairs)/2)
    i = 0
    for y in ys:
        plt.plot(index,y, alpha=opacity, label=domain_pairs[i],marker= markers[i])
        i += 1

    plt.title(convert_title(lookfor_pair,pv_method),size=22)
    plt.xlabel('pivot range',size=18)
    plt.xticks(index,x)
    plt.ylabel('$J(M_1,M_2)$' if jaccard == True else '$K(M_1,M_2)$',size=18)
    #right box
    box = ax.get_position()
    ax.set_position([box.x0-box.width*0.05, box.y0 , box.width*0.95, box.height])

    ax.legend(loc='upper center', bbox_to_anchor=(1.1,0.9),
          fancybox=True, shadow=True, ncol=1)
    plt.show()
    pass      

def constructer(sim_list,jaccard):
    # sim_list.sort(lambda x,y:-1 if x[2]<y[2] else 1)
    ys = []
    domain_pairs = list(set([p[2] for p in sim_list]))
    domain_pairs.sort()
    x = list(set([p[1] for p in sim_list]))
    x.sort()

    for pair in domain_pairs:
        if jaccard == True:
            ys.append([tmp[0] for tmp in sim_list if tmp[2] == pair])
        else:
            ys.append([tmp[3] for tmp in sim_list if tmp[2] == pair])

    return x,ys,domain_pairs
    pass

def convert_title(lookfor_pair,pv_method):
    if 'landmark' in pv_method or 'landmark' in lookfor_pair:
        return "%s vs %s" % (lookfor_pair[0].replace("LANDMARK_",""),lookfor_pair[1].replace("LANDMARK_",""))
    else:
        return "%s$_%s$ vs %s$_%s$" % (lookfor_pair[0],pv_method,lookfor_pair[1],pv_method)


def draw(pv_method,lookfor_pair,jaccard):
    drawer(constructer(collector(pv_method, lookfor_pair),jaccard),jaccard)
    pass

def draw_margnal(pv_method, lookfor_pair,jaccard):
    drawer_margnal(constructer(collector_margnal(pv_method, lookfor_pair, 500),jaccard),jaccard)
    pass

if __name__ == "__main__":
    m1 = "freq"
    m2 = "mi"
    pv_method = "L"
    # m1 = "landmark_word2vec"
    # m2 = "landmark_word2vec_ppmi"
    # m1 = "landmark_pretrained_glove"
    # m2 = "landmark_pretrained_glove_ppmi"
    # m1 = "freq"
    # m2 = "landmark_pretrained_word2vec"
    # m2 = "landmark_glove"
    # pv_method = "100"
    lookfor_pair = (m1.upper(),m2.upper())
    jaccard = True
    # lookfor_pair = (m1,m2)
    # draw(pv_method,lookfor_pair,jaccard)
    draw_margnal(pv_method,lookfor_pair,jaccard)
    
