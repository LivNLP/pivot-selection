"""
pivot selection for non DA datasets

unlabelled methods
"""

def count_freq(fname, h):
    for line in open("%s-sentences" % (fname)):
        for feat in line.strip().split():
            h[feat] = h.get(feat, 0) + 1
    pass

def write_original_sentences(fname):
    lines = [line for line in open(fname)] 
    res_file = open("%s-sentences" % (fname), 'w')
    for line in open(fname):
        res_file.write("%s\n" % ' '.join([word.replace(':1','') for word in line.strip().split(' ')[1:]]))
    res_file.close()  
    print "original sentences in %s have been written to the disk"%fname 
    pass


# coreness = pivothood = freq in domains = min(h(w,S), h(w,T))
# if single domain? train = source, test = target
def select_pivots_freq(domain):
    source_fname = "../data/%s/train"%domain
    target_fname = "../data/%s/test"%domain

    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    src_freq = {}
    tgt_freq = {}
    count_freq(source_fname,src_freq)
    count_freq(target_fname,tgt_freq)

    s = {}
    features = set(src_freq.keys()).union(set(tgt_freq.keys()))
    for feat in features:
        s[feat] = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    save_obj(L,domain,'un_freq')
    pass

def select_pivots_mi(domain):
    source_fname = "../data/%s/train"%domain
    target_fname = "../data/%s/test"%domain

    src_reviews = float(count_reviews(source_fname,'all'))
    tgt_reviews = float(count_reviews(target_fname,'all'))
    total_reviews = float(src_reviews+tgt_reviews)
    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    features = set(features_list(source_fname+'-sentences')).union(set(features_list(target_fname+'-sentences')))
    # print features
    x_src = reviews_contain_x(features_list(source_fname+'-sentences'),source_fname+'-sentences')
    x_tgt = reviews_contain_x(features_list(target_fname+'-sentences'),target_fname+'-sentences')
    x_total = combine_dicts(x_src,x_tgt)
    

    mi_dict={}
    for x in features:
        if x_total.get(x,0) > 0 and x_src.get(x,0) > 0 and x_tgt.get(x,0) > 0:
            # print x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews
            src_mi = mi(x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews) 
            tgt_mi = mi(x_total.get(x,0), x_tgt.get(x,0), tgt_reviews, total_reviews)
            mi_dict[x] = (src_mi-tgt_mi)**2
    L = mi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    save_obj(L,domain,'un_mi')

def select_pivots_pmi(domain):
    source_fname = "../data/%s/train"%domain
    target_fname = "../data/%s/test"%domain

    src_reviews = float(count_reviews(source_fname,'all'))
    tgt_reviews = float(count_reviews(target_fname,'all'))
    total_reviews = float(src_reviews+tgt_reviews)
    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    features = set(features_list(source_fname+'-sentences')).union(set(features_list(target_fname+'-sentences')))
    # print features
    x_src = reviews_contain_x(features_list(source_fname+'-sentences'),source_fname+'-sentences')
    x_tgt = reviews_contain_x(features_list(target_fname+'-sentences'),target_fname+'-sentences')
    x_total = combine_dicts(x_src,x_tgt)
    

    pmi_dict={}
    for x in features:
        if x_total.get(x,0) > 0 and x_src.get(x,0) > 0 and x_tgt.get(x,0) > 0:
            # print x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews
            src_pmi = pmi(x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews) 
            tgt_pmi = pmi(x_total.get(x,0), x_tgt.get(x,0), tgt_reviews, total_reviews)
            pmi_dict[x] = (src_pmi-tgt_pmi)**2
    L = pmi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    save_obj(L,domain,'un_pmi')
    pass

def select_pivots_ppmi(domain):
    source_fname = "../data/%s/train"%domain
    target_fname = "../data/%s/test"%domain

    src_reviews = float(count_reviews(source_fname,'all'))
    tgt_reviews = float(count_reviews(target_fname,'all'))
    total_reviews = float(src_reviews+tgt_reviews)
    write_original_sentences(source_fname)
    write_original_sentences(target_fname)
    features = set(features_list(source_fname+'-sentences')).union(set(features_list(target_fname+'-sentences')))
    # print features
    x_src = reviews_contain_x(features_list(source_fname+'-sentences'),source_fname+'-sentences')
    x_tgt = reviews_contain_x(features_list(target_fname+'-sentences'),target_fname+'-sentences')
    x_total = combine_dicts(x_src,x_tgt)
    

    ppmi_dict={}
    for x in features:
        if x_total.get(x,0) > 0 and x_src.get(x,0) > 0 and x_tgt.get(x,0) > 0:
            # print x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews
            src_ppmi = ppmi(x_total.get(x,0), x_src.get(x,0), src_reviews, total_reviews) 
            tgt_ppmi = ppmi(x_total.get(x,0), x_tgt.get(x,0), tgt_reviews, total_reviews)
            ppmi_dict[x] = (src_ppmi-tgt_ppmi)**2
    L = ppmi_dict.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    save_obj(L,domain,'un_ppmi')
    pass


# write separately original sentences to positive and negative
def count_reviews(fname,opt):
    count = 0
    if opt == "pos":
        count = len([line for line in open(fname) if line.strip().split(' ')[0]=='+1'])

    elif opt == "neg":
        count = len([line for line in open(fname) if line.strip().split(' ')[0]=='-1'])
    else:
        count = len([line for line in open(fname)])
    return count
    pass

def features_list(fname):
    return list(set([word for line in open(fname) for word in line.strip().split()]))

def reviews_contain_x(features, fname):
    # for x in features:
    #     for line in open(fname):
    #         if x in line.strip().split():
    #             h[x] = h.get(x, 0) + 1
    features = list(features)
    feautres_vector = numpy.zeros(len(features), dtype=float)
    for line in open(fname):
        # print line
        for x in set(line.strip().split(',')):
            i = features.index(x)
            feautres_vector[i] += 1
    return dict(zip(features,feautres_vector))

def ppmi(joint_x, x_scale, y, N):
    prob_y = float(y) / float(N)
    prob_x = float(joint_x) / float(N)
    prob_x_scale = float(x_scale) / float(N)
    val = float(prob_x_scale) / float(prob_x * prob_y)
    return math.log(val) if math.log(val) > 0 else 0

def pmi(joint_x, x_scale, y, N):
    prob_y = float(y) / float(N)
    prob_x = float(joint_x) / float(N)
    prob_x_scale = float(x_scale) / float(N)
    val = float(prob_x_scale) / float(prob_x * prob_y)
    return prob_x_scale * math.log(val)

def mi(joint_x, x_scale, y, N):
    prob_y = float(y) / float(N)
    prob_x = float(joint_x) / float(N)
    prob_x_scale = float(x_scale) / float(N)
    val = float(prob_x_scale) / float(prob_x * prob_y)
    return math.log(val)

# method to combine dictionaries
def combine_dicts(a, b):
    return dict([(n, a.get(n, 0)+b.get(n, 0)) for n in set(a)|set(b)])

# to reduce duplicated computation, save object
def save_obj(obj, dataset,name):
    filename = '../work/%s/obj/%s.pkl'%(dataset,name)
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        print '%s saved'%filename

# load object
def load_obj(dataset,name):
    with open('../work/%s/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    datasets = ["TR", "CR", "SUBJ","MR"]
    print "starting..."
    for dataset in datasets:
        print "dataset =",dataset
        select_pivots_freq(dataset)
        select_pivots_mi(dataset)
        select_pivots_pmi(dataset)
        select_pivots_ppmi(dataset)
