import time
import math
import pickle
import numpy
import itertools
import compare_ranking as cr # submethod 
# from multiprocessing import Pool,Process,Queue

def select_pivots_freq(source, target):
    src_freq = {}
    tgt_freq = {}
    count_freq("../data/%s/train.positive" % source, src_freq)
    count_freq("../data/%s/train.negative" % source, src_freq)
    # count_freq("../data/%s/train.unlabeled" % source, src_freq)
    count_freq("../data/%s/train.positive" %  target, tgt_freq)
    count_freq("../data/%s/train.negative" %  target, tgt_freq)
    # count_freq("../data/%s/train.unlabeled" % target, tgt_freq) 
    s = {}
    features = set(src_freq.keys()).union(set(tgt_freq.keys()))
    for feat in features:
        s[feat] = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    #h = L[:k]
    return L
    # for (feat, freq) in L[:10]:
    #     print feat, src_freq.get(feat, 0), tgt_freq.get(feat, 0)    

# count frequency and return a dict h
def count_freq(fname, h):
    for line in open(fname):
        for feat in line.strip().split():
            h[feat] = h.get(feat, 0) + 1
    pass

def select_un_pivots_freq(source, target):
    # print "source =", source
    # print "target =", target
    src_freq = {}
    tgt_freq = {}
    count_freq("../data/%s/train.unlabeled" % source, src_freq)
    count_freq("../data/%s/train.unlabeled" % target, tgt_freq) 
    s = {}
    features = set(src_freq.keys()).union(set(tgt_freq.keys()))
    for feat in features:
        s[feat] = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    #h = L[:k]
    return L
    # for (feat, freq) in L[:10]:
    #     print feat, src_freq.get(feat, 0), tgt_freq.get(feat, 0)    

# recall stored objects and compute mi absolute value
def select_pivots_mi():
    features = load_obj("features")
    x_src = load_obj("x_src")
    x_tgt = load_obj("x_tgt")
    x_pos_src = load_obj("x_pos_src")
    x_neg_src = load_obj("x_neg_src")
    x_pos_tgt = load_obj("x_pos_tgt")
    x_neg_tgt = load_obj("x_neg_tgt")
    src_reviews = load_obj("src_reviews")
    tgt_reviews = load_obj("tgt_reviews")
    pos_src_reviews = load_obj("pos_src_reviews")
    neg_src_reviews = load_obj("neg_src_reviews")
    pos_tgt_reviews = load_obj("pos_tgt_reviews")
    neg_tgt_reviews = load_obj("neg_tgt_reviews")

    mi_dict = {}
    for x in features:
        if x_src.get(x,0)*x_pos_src.get(x,0)*x_neg_src.get(x,0) > 0:
            pos_mi = mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_reviews, src_reviews) 
            neg_mi = mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_reviews, src_reviews)
            mi_dict[x] = abs(pos_mi-neg_mi)
    L = mi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    #h = L[:k]
    return L
    # for (x, mi) in L[:k]:
    #     print x, mi_dict.get(x,0)
    # pass

# John Blitzer's mi method for labelled data
def select_pivots_mi_jb():
    features = load_obj("features")
    x_src = load_obj("x_src")
    x_tgt = load_obj("x_tgt")
    x_pos_src = load_obj("x_pos_src")
    x_neg_src = load_obj("x_neg_src")
    x_pos_tgt = load_obj("x_pos_tgt")
    x_neg_tgt = load_obj("x_neg_tgt")
    src_reviews = load_obj("src_reviews")
    tgt_reviews = load_obj("tgt_reviews")
    pos_src_reviews = load_obj("pos_src_reviews")
    neg_src_reviews = load_obj("neg_src_reviews")
    pos_tgt_reviews = load_obj("pos_tgt_reviews")
    neg_tgt_reviews = load_obj("neg_tgt_reviews")

    mi_dict = {}
    for x in features:
        if x_src.get(x,0)*x_pos_src.get(x,0)*x_neg_src.get(x,0) > 0:
            pos_pmi = pointwise_mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_reviews, src_reviews) 
            neg_pmi = pointwise_mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_reviews, src_reviews)
            mi_dict[x] = -(pos_pmi+neg_pmi)
    L = mi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    for (x, mi) in L:
        print x, mi_dict.get(x,0)
    #h = L[:k]
    return L

def round_5(num):
    return "%.5f" % num

# a little change to get pmi absolute value
def select_pivots_pmi(k):
    features = load_obj("features")
    x_src = load_obj("x_src")
    x_tgt = load_obj("x_tgt")
    x_pos_src = load_obj("x_pos_src")
    x_neg_src = load_obj("x_neg_src")
    x_pos_tgt = load_obj("x_pos_tgt")
    x_neg_tgt = load_obj("x_neg_tgt")
    src_reviews = load_obj("src_reviews")
    tgt_reviews = load_obj("tgt_reviews")
    pos_src_reviews = load_obj("pos_src_reviews")
    neg_src_reviews = load_obj("neg_src_reviews")
    pos_tgt_reviews = load_obj("pos_tgt_reviews")
    neg_tgt_reviews = load_obj("neg_tgt_reviews")

    pmi_dict = {}
    others = {}
    for x in features:
        if x_src.get(x,0)*x_pos_src.get(x,0)*x_neg_src.get(x,0) > 0:
            pos_pmi = pointwise_mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_reviews, src_reviews) 
            neg_pmi = pointwise_mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_reviews, src_reviews)
            pmi_dict[x] = abs(pos_pmi[0]-neg_pmi[0])
            # others[x] = [round_5(pos_pmi[1]),round_5(pos_pmi[2]),round_5(neg_pmi[2])]
            others[x] = [x_src.get(x,0),x_pos_src.get(x,0),x_neg_src.get(x,0)]
    L = pmi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    # h = L[:k]
    # return L
    for (x, pmi) in L[:k]:
        print x, pmi_dict.get(x,0),others.get(x,0)
    pass

# unlabel mi
def select_un_pivots_mi():
    un_src_reviews = load_obj("un_src_reviews")
    un_tgt_reviews = load_obj("un_tgt_reviews")
    un_reviews = load_obj("un_reviews")
    un_features = load_obj("un_features")
    x_un_src = load_obj("x_un_src")
    x_un_tgt = load_obj("x_un_tgt")
    x_un = load_obj("x_un")

    mi_dict = {}
    for x in un_features:
        if x_un.get(x,0)*x_un_src.get(x,0)*x_un_tgt.get(x,0) > 0:
            src_mi = mutual_info(x_un.get(x,0), x_un_src.get(x,0), un_src_reviews, un_reviews) 
            tgt_mi = mutual_info(x_un.get(x,0), x_un_tgt.get(x,0), un_tgt_reviews, un_reviews)
            mi_dict[x] = abs(src_mi-tgt_mi)
    L = mi_dict.items()
    L.sort(lambda x, y: -1 if x[1] < y[1] else 1)
    #h = L[:k]
    return L
    # for (x, mi) in L[:k]:
    #     print x, mi_dict.get(x,0)
    # pass

# unlabel pmi
def select_un_pivots_pmi(k):
    un_src_reviews = load_obj("un_src_reviews")
    un_tgt_reviews = load_obj("un_tgt_reviews")
    un_reviews = load_obj("un_reviews")
    un_features = load_obj("un_features")
    x_un_src = load_obj("x_un_src")
    x_un_tgt = load_obj("x_un_tgt")
    x_un = load_obj("x_un")

    pmi_dict = {}
    others = {}
    for x in un_features:
        if x_un.get(x,0)*x_un_src.get(x,0)*x_un_tgt.get(x,0) > 0:
            src_pmi = pointwise_mutual_info(x_un.get(x,0), x_un_src.get(x,0), un_src_reviews, un_reviews) 
            tgt_pmi = pointwise_mutual_info(x_un.get(x,0), x_un_tgt.get(x,0), un_tgt_reviews, un_reviews)
            pmi_dict[x] = abs(src_pmi[0]-tgt_pmi[0])
            # others[x] = [round_5(src_pmi[1]),round_5(src_pmi[2]),round_5(tgt_pmi[2])]
            others[x] = [x_un.get(x,0),x_un_src.get(x,0),x_un_tgt.get(x,0),]
    L = pmi_dict.items()
    L.sort(lambda x, y: -1 if x[1] < y[1] else 1)
    #h = L[:k]
    # return L
    for (x, pmi) in L[:k]:
        print x, pmi_dict.get(x,0),others.get(x,0)
    pass

# to construct presets of labeled data in source and target domain
def label_presets(source, target):
    # source
    pos_src_reviews = count_reviews("../data/%s/train.positive" % source)
    neg_src_reviews = count_reviews("../data/%s/train.negative" % source)
    src_reviews = pos_src_reviews + neg_src_reviews
    pos_src_features = features_list("../data/%s/train.positive" % source)
    neg_src_features = features_list("../data/%s/train.negative" % source)
    
    #target
    pos_tgt_reviews = count_reviews("../data/%s/train.positive" % target)
    neg_tgt_reviews = count_reviews("../data/%s/train.negative" % target)
    tgt_reviews = pos_tgt_reviews + neg_tgt_reviews
    pos_tgt_features = features_list("../data/%s/train.positive" % target)
    neg_tgt_features = features_list("../data/%s/train.negative" % target)

    src_features = set(pos_src_features).union(set(neg_src_features))
    tgt_features = set(pos_tgt_features).union(set(neg_tgt_features))
    # all the features in both domains, or say all the x
    features = set(src_features).union(set(tgt_features))

    # reviews contain x
    x_pos_src = reviews_contain_x(features, "../data/%s/train.positive" % source)
    x_neg_src = reviews_contain_x(features, "../data/%s/train.negative" % source)
    x_pos_tgt = reviews_contain_x(features, "../data/%s/train.positive" % target)
    x_neg_tgt = reviews_contain_x(features, "../data/%s/train.negative" % target)
    x_src = combine_dicts(x_pos_src, x_neg_src) 
    x_tgt = combine_dicts(x_pos_tgt, x_neg_tgt)

    # save to temp obj
    save_obj(features,"features")
    save_obj(x_pos_src,"x_pos_src")
    save_obj(x_neg_src,"x_neg_src")
    save_obj(x_pos_tgt,"x_pos_tgt")
    save_obj(x_neg_tgt,"x_neg_tgt")
    save_obj(x_src,"x_src")
    save_obj(x_tgt,"x_tgt")
    save_obj(src_reviews,"src_reviews")
    save_obj(tgt_reviews,"tgt_reviews")
    save_obj(pos_src_reviews,"pos_src_reviews")
    save_obj(pos_tgt_reviews,"pos_tgt_reviews")
    save_obj(neg_src_reviews,"neg_src_reviews")
    save_obj(neg_tgt_reviews,"neg_tgt_reviews")
    pass

# to construct presets for unlabeled data in source and target domain
def unlabel_presets(source, target):
    # no of reviews
    un_src_reviews = count_reviews("../data/%s/train.unlabeled" % source)
    un_tgt_reviews = count_reviews("../data/%s/train.unlabeled" % target)
    un_reviews = un_src_reviews + un_tgt_reviews

    # featrues
    un_src_features = features_list("../data/%s/train.unlabeled" % source)
    un_tgt_features = features_list("../data/%s/train.unlabeled" % target)
    un_features = set(un_src_features).union(set(un_tgt_features))

    # reviews contain x
    x_un_src = reviews_contain_x(un_features, "../data/%s/train.unlabeled" % source)
    # print x_un_src
    x_un_tgt = reviews_contain_x(un_features, "../data/%s/train.unlabeled" % target)
    x_un = combine_dicts(x_un_src, x_un_tgt)

    # save to temp obj
    save_obj(un_src_reviews,"un_src_reviews")
    save_obj(un_tgt_reviews,"un_tgt_reviews")
    save_obj(un_reviews,"un_reviews")
    save_obj(un_features,"un_features")
    save_obj(x_un_src,"x_un_src")
    save_obj(x_un_tgt,"x_un_tgt")
    save_obj(x_un,"x_un")
    pass

# count the number of reviews in specified file
def count_reviews(fname):
    return float(sum(1 for line in open(fname)))

# rather than do this for each feature one by one, use a vector for each review
def reviews_contain_x(features, fname):
    # for x in features:
    #     for line in open(fname):
    #         if x in line.strip().split():
    #             h[x] = h.get(x, 0) + 1
    features = list(features)
    feautres_vector = numpy.zeros(len(features), dtype=float)
    for line in open(fname):
        for x in set(line.strip().split()):
            i = features.index(x)
            feautres_vector[i] += 1
    return dict(zip(features,feautres_vector))

# return a list of all features in specified file
def features_list(fname):
    return list(set([word for line in open(fname) for word in line.split()]))

# method to combine dictionaries
def combine_dicts(a, b):
    return dict([(n, a.get(n, 0)+b.get(n, 0)) for n in set(a)|set(b)])

# mutual info
def mutual_info(joint_x, x_scale, y, N):
    prob_y = float(y / N)
    prob_x = float(joint_x / N)
    prob_x_scale = float(x_scale / N)
    val = float(prob_x_scale / (prob_x * prob_y))
    return prob_x_scale * math.log(val)

# only difference between mi is no addition multipler
def pointwise_mutual_info(joint_x, x_scale, y, N):
    prob_y = float(y / N)
    prob_x = float(joint_x / N)
    prob_x_scale = float(x_scale / N)
    val = float(prob_x_scale / (prob_x * prob_y))
    return math.log(val),prob_x,prob_x_scale


# to reduce duplicated computation, save object
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load object
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# load stored object
def load_stored_obj(name):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

# compare similaries between L and U
def sim_eval(method, test_k):
    resFile = open("../work/sim/Sim.%s.csv"% method, "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    resFile.write("Source, Target, Method, JC, KC, #pivots\n")
    for k in test_k:
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                label = load_stored_obj("../work/%s-%s/obj/%s"%(source,target,method))
                un_label = load_stored_obj("../work/%s-%s/obj/un_%s"%(source,target,method))
                h1 = label[:k]
                h2 = un_label[:k]
                JC = cr.jaccard_coefficient(h1,h2)
                KC = cr.kendall_rank_coefficient(h1,h2)
                resFile.write("%s, %s, %s, %f, %f, %f\n" % (source, target, method, JC, KC, k))
                resFile.flush()
    resFile.close()
    pass

# compare similaries between methods in L and U
def methods_eval(dataset,test_k):
    resFile = open("../work/sim/MethodSim.%s.csv"% dataset, "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    methods = ["freq","mi","pmi"]
    if dataset == "U":
        methods = ["un_freq","un_mi","un_pmi"]
    method_pairs = list(itertools.combinations(methods, 2))
    print "We are going to compare ", method_pairs, " in ", dataset
    resFile.write("Source, Target, Method, Method, JC, KC, #pivots\n")
    for k in test_k:
        print "#pivots = ", k
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                for (i, j) in method_pairs:
                    method_i = load_stored_obj("../work/%s-%s/obj/%s"%(source,target,i))
                    method_j = load_stored_obj("../work/%s-%s/obj/%s"%(source,target,j))
                    h1 = method_i[:k]
                    h2 = method_j[:k]
                    JC = cr.jaccard_coefficient(h1,h2)
                    KC = cr.kendall_rank_coefficient(h1,h2)
                    print "%s -> %s (%s, %s): JC = %f KC = %f" % (source, target, i, j, JC, KC)
                    resFile.write("%s, %s, %s, %s, %f, %f, %f\n" % (source, target, i, j, JC, KC, k))
                    resFile.flush()
    resFile.close()
    pass

def mi_eval(test_k):
    resFile = open("../work/sim/MISim.L.csv", "w")
    domains = ["books", "electronics", "dvd", "kitchen"]
    methods = ["mi","mi_jb"]
    method_pairs = list(itertools.combinations(methods, 2))
    print "We are going to compare ", method_pairs, " in L"
    resFile.write("Source, Target, Method, Method, JC, KC, #pivots\n")
    for k in test_k:
        print "#pivots = ", k
        for source in domains:
            for target in domains:
                if source == target:
                    continue
                for (i, j) in method_pairs:
                    method_i = load_stored_obj("../work/%s-%s/obj/%s"%(source,target,i))
                    method_j = load_stored_obj("../work/%s-%s/obj/%s"%(source,target,j))
                    h1 = method_i[:k]
                    h2 = method_j[:k]
                    JC = cr.jaccard_coefficient(h1,h2)
                    KC = 0
                    if JC != 0:
                        KC = cr.kendall_rank_coefficient(h1,h2)
                    print "%s -> %s (%s, %s): JC = %f KC = %f" % (source, target, i, j, JC, KC)
                    resFile.write("%s, %s, %s, %s, %f, %f, %f\n" % (source, target, i, j, JC, KC, k))
                    resFile.flush()
    resFile.close()

# get top-k pivots with its value and print on the screen
def top_k_pivots(source,target,method, k):
    pivotsFile = "../work/%s-%s/obj/%s" % (source, target, method)
    print "[%s -> %s]\n[method: %s]\n<top-%d>"% (source, target, method,k)
    L = load_stored_obj(pivotsFile)[:k]
    for (x, v) in L:
        print x, v
    print "###########################################\n\n"
pass

# main
if __name__ == "__main__":
    # label_presets("electronics", "books")
    # unlabel_presets("electronics", "books")
    # source = "books"
    # target = "dvd"
    select_pivots_pmi(20)
    print "###########################\n\n"
    select_un_pivots_pmi(20)
    # print "source =", source
    # print "target =", target
    # save_obj(select_pivots_freq(source,target),"freq")
    # save_obj(select_un_pivots_freq(source,target),"un_freq")
    # save_obj(select_pivots_mi(),"mi")
    # save_obj(select_pivots_mi_jb(),"mi_jb")
    # save_obj(select_un_pivots_mi(),"un_mi")
    # save_obj(select_pivots_pmi(),"pmi")
    # save_obj(select_un_pivots_pmi(),"un_pmi")
    # freq = load_obj("freq")
    # un_freq = load_obj("un_freq")
    # mi = load_obj("un_mi")
    # un_mi = load_obj("un_mi")
    # pmi = load_obj("pmi")
    # un_pmi = load_obj("un_pmi")
    # test_k = [100,500,1000,1500,2000,3000]
    # methods = ["freq","mi","pmi"]
    # for method in methods:
    #     sim_eval(method, test_k)
    # test_k = [100,200,300,400,500,1000,1500,2000]
    # datasets = ["L","U"]
    # for dataset in datasets:
    #     methods_eval(dataset, test_k)
    # methods = ["freq","un_freq","mi","un_mi","pmi","un_pmi"]
    # k = 5
    # for method in methods:
    #     top_k_pivots(source,target,method,k)
    # test_k = [100,200,300,400,500,1000,1500,2000]
    # mi_eval(test_k)
