import math
def select_pivots_freq(source, target):
    print "source =", source
    print "target =", target
    src_freq = {}
    tgt_freq = {}
    count_freq("../data/%s/train.positive" % source, src_freq)
    count_freq("../data/%s/train.negative" % source, src_freq)
    count_freq("../data/%s/train.unlabeled" % source, src_freq)
    count_freq("../data/%s/train.positive" %  target, tgt_freq)
    count_freq("../data/%s/train.negative" %  target, tgt_freq)
    count_freq("../data/%s/train.unlabeled" % target, tgt_freq) 
    s = {}
    features = set(src_freq.keys()).union(set(tgt_freq.keys()))
    for feat in features:
        s[feat] = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    for (feat, freq) in L[:10]:
        print feat, src_freq.get(feat, 0), tgt_freq.get(feat, 0)    
    pass


def count_freq(fname, h):
    for line in open(fname):
        for feat in line.strip().split():
            h[feat] = h.get(feat, 0) + 1
    pass

# perhaps recontruct the method to here after test
def select_pivots_mi(source, target):
    print "source =", source
    print "target =", target

    pass

# to construct mutual info list 
def compute_mi(source, target):
    #initial
    x_pos_src = {}
    x_neg_src = {}
    x_pos_tgt = {}
    x_neg_tgt = {}

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
    # print len(features)

    # reviews
    reviews_contain_x(features, "../data/%s/train.positive" % source, x_pos_src)
    # print x_pos_src
    reviews_contain_x(features, "../data/%s/train.negative" % source, x_neg_src)
    reviews_contain_x(features, "../data/%s/train.positive" % target, x_pos_tgt)
    reviews_contain_x(features, "../data/%s/train.negative" % target, x_neg_tgt)
    x_src = combine_dicts(x_pos_src, x_neg_src) 
    x_tgt = combine_dicts(x_pos_tgt, x_neg_tgt)

    print x_src

    mi_dict = {}
    for x in features:
        pos_mi = mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_reviews, src_reviews) 
        neg_mi = mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_reviews, src_reviews)
        mi_dict[x] = min(pos_mi, neg_mi)
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    for (x, mi) in L[:10]:
        print x, mi_dict.get(x,0)
    pass

# to construct pairwise mutual info list
def compute_pmi():
    pass

# count the number of reviews in specified file
def count_reviews(fname):
    return float(sum(1 for line in open(fname)))

# count features appearence of reviews in specified file and assign to dict h
def reviews_contain_x(features, fname, h):
    for line in open(fname):
        for x in features:
            if x in line.strip().split():
                h[x] = h.get(x, 0) + 1
    pass

# return a list of all features in specified file
def features_list(fname):
    return list(set([word for line in open(fname) for word in line.split()]))

# method to combine dictionaries
def combine_dicts(a, b):
    return dict([(n, a.get(n, 0)+b.get(n, 0)) for n in set(a)|set(b)])

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

if __name__ == "__main__":
    # compute_mi("books", "dvd")
    # select_pivots_freq("books", "dvd")
    # source = "books"
    # print "source =", source
    # src_pos_reviews = count_reviews("../data/%s/test.positive" % source)
    # print src_pos_reviews
    # features = []
    # s1 = {}
    # s2 = {}
    # s = {}
    # features = features_list("../data/%s/test.positive" % "books")
    # print len(features)
    # reviews_contain_x(features, "../data/%s/test.positive" % "books",s1)
    # reviews_contain_x(features, "../data/%s/test.negative" % "books",s2)
    # s = combine_dicts(s1, s2)
    # for x in features[:10]:
    #     if s1[x]*s[x]>0:
    #         pos_mi = mutual_info(s1[x], s[x], 200.0, 400.0) 
    #         print x, pos_mi
    pass
