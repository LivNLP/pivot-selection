import time
import math
import pickle
import numpy
# from multiprocessing import Pool,Process,Queue

def select_pivots_freq(source, target, k):
    print "source =", source
    print "target =", target
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
    h = L[:k]
    return h
    # for (feat, freq) in L[:10]:
    #     print feat, src_freq.get(feat, 0), tgt_freq.get(feat, 0)    

# count frequency and return a dict h
def count_freq(fname, h):
    for line in open(fname):
        for feat in line.strip().split():
            h[feat] = h.get(feat, 0) + 1
    pass

def select_un_pivots_freq(source, target, k):
    print "source =", source
    print "target =", target
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
    h = L[:k]
    return h
    # for (feat, freq) in L[:10]:
    #     print feat, src_freq.get(feat, 0), tgt_freq.get(feat, 0)    

# recall stored objects and compute mi absolute value
def select_pivots_mi(k):
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
    for (x, mi) in L[:k]:
        print x, mi_dict.get(x,0)
    pass

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
    for x in features:
        if x_src.get(x,0)*x_pos_src.get(x,0)*x_neg_src.get(x,0) > 0:
            pos_pmi = pairwise_mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_reviews, src_reviews) 
            neg_pmi = pairwise_mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_reviews, src_reviews)
            pmi_dict[x] = abs(pos_pmi-neg_pmi)
    L = pmi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    for (x, pmi) in L[:k]:
        print x, pmi_dict.get(x,0)
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

# unlabel mi
def select_un_pivots_mi(k):
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
    for (x, mi) in L[:k]:
        print x, mi_dict.get(x,0)
    pass

#unlabel pmi
def select_un_pivots_pmi(k):
    
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
def pairwise_mutual_info(joint_x, x_scale, y, N):
    prob_y = float(y / N)
    prob_x = float(joint_x / N)
    prob_x_scale = float(x_scale / N)
    val = float(prob_x_scale / (prob_x * prob_y))
    return math.log(val)

# jaccard coefficient between a and b
def jaccard_coefficient(a, b):
    A = set([i[0] for i in a])
    B = set([i[0] for i in b])
    return float(len(A & B))/float(len(A | B)) if float(len(A | B)) != 0 else 0

# to reduce duplicated computation, save object
def save_obj(obj, name):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# load object
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# main
if __name__ == "__main__":
    # start_time = time.time()
    label_presets("electronics", "kitchen")
    unlabel_presets("electronics", "kitchen")
    # select_pivots_mi(10)
    # select_pivots_pmi(10)
    # h = select_pivots_freq("books", "dvd", 100)
    # h2 = select_un_pivots_freq("books","dvd", 100)
    # print "jaccard_coefficient = ", jaccard_coefficient(h,h2)
    # select_un_pivots_mi("books","dvd")
    # features = features_list("../data/%s/train.positive" % "books")
    # b = reviews_contain_x(features, "../data/%s/train.positive" % "books")
    # print len(b)
    # print time.time() - start_time
    pass
