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

def mutual_info(word, label):
	
	pass

if __name__ == "__main__":
    select_pivots_freq("books", "dvd")
    pass
