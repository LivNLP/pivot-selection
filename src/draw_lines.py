import numpy as np
import matplotlib.pyplot as plt
import matplotlib
opacity = 0.6
font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 15}
matplotlib.rc('font', **font)

def collector(method,lookfor_pair):
    new_list = []
    input_file = open("../work/sim/Sim.%s.csv"%method,'r')
    next(input_file)
    for line in input_file:
        p = line.strip('\n').split(', ')
        src = p[0][0].capitalize()
        tgt = p[1][0].capitalize()
        pair = "%s-%s"%(src,tgt)
        if lookfor_pair == pair:    
            jaccard = float(p[3])
            n_pivots = int(float(p[5]))
            new_list.append((jaccard,n_pivots))   
    # print new_list
    return new_list
    pass


def drawer(draw_lists,lookfor_pair,methods):
    plt.figure(figsize=(7,5.5))
    x = [tmp[1] for tmp in draw_lists[0]]
    ys = []
    for draw_list in draw_lists:
        ys.append([tmp[0] for tmp in draw_list])
    index = np.arange(len(x))
    i = 0
    for y in ys:    
        plt.plot(index,y, marker="o", alpha=opacity, label=convert(methods[i]))
        # plt.plot(index,y, marker="o",alpha=opacity, label=convert(methods[i]),color='r')
        i += 1

    plt.title(lookfor_pair)
    plt.xlabel('#pivots')
    plt.xticks(index,x)
    plt.ylabel('Jaccard$_{L,U}$')
    plt.legend()

    plt.show()
    pass

def drawer_two_pairs(collection_1,collection_2,pair_1,pair_2,methods):
    plt.figure(figsize=(7,5.5))
    x = [tmp[1] for tmp in collection_1[0]]
    ys_1 = []
    ys_2 = []
    for draw_list in collection_1:
        ys_1.append([tmp[0] for tmp in draw_list])
    for draw_list in collection_2:
        ys_2.append([tmp[0] for tmp in draw_list])
    index = np.arange(len(x))
    i = 0
    for y_1 in ys_1:
        plt.plot(index,y_1, marker="o", alpha=opacity, label=convert_label(methods[i],pair_1))
        i+=1
    i = 0
    for y_2 in ys_2:
        plt.plot(index,y_2, marker="^", alpha=opacity, label=convert_label(methods[i],pair_2))
        i+=1

    plt.title("%s and %s"%(pair_1, pair_2))
    plt.xlabel('#pivots')
    plt.xticks(index,x)
    plt.ylabel('Jaccard$_{L,U}$')
    plt.legend()

    plt.show()
    pass

def constructer(methods,lookfor_pair):
    new_list = []
    for method in methods:
        new_list.append(collector(method,lookfor_pair))
    return new_list
    pass

def convert(method):
    return method.upper()

def convert_label(method,domain_pair):
    return "%s$_{%s}$"% (method.upper(),domain_pair)

if __name__ == "__main__":
    # methods = ["mi","pmi"]
    methods = ["freq"]
    # lookfor_pair = "E-K"
    # lookfor_pair = "K-E"
    pair_1 = "E-K"
    pair_2 = "K-E"
    drawer_two_pairs(constructer(methods,pair_1),constructer(methods,pair_2),
        pair_1, pair_2, methods)
    # drawer(constructer(methods,lookfor_pair),lookfor_pair,methods)
