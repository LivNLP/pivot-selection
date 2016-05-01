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

def select_pivots_mi(source, target):
    print "source =", source
    print "target =", target

    pass

# to construct mutual info list 
def compute_mi():
    pass

# to construct pairwise mutual info list
def compute_pmi():
    pass

def count_reviews(fname):
    return sum(1 for line in open(fname))

def reviews_contain_x(features, fname, h):
    for line in open(fname):
        for x in features:
            if x in line.strip().split():
                h[x] = h.get(x, 0) + 1
    pass

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
    # select_pivots_freq("books", "dvd")
    # source = "books"
    # print "source =", source
    # src_pos_reviews = count_reviews("../data/%s/test.positive" % source)
    # print src_pos_reviews
    # features = []
    # s = {}
    # features = features_list("../data/%s/test.positive" % "books")
    # print len(features)
    # reviews_contain_x(features, "../data/%s/test.positive" % "books",s)
    # print len(s)
    pass
