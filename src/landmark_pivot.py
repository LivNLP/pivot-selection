# landmark-based pivot selection method
import select_pivots as sp 
import numpy
import pickle
import glob
import gensim,logging
from glove import Corpus
import glove

# construct training data for such domian: labeled and unlabeled
# format: [[review 1],[review 2]...]
def labeled_reviews(domain_name):
    pos_file = "../data/%s/train.positive" % domain_name
    neg_file = "../data/%s/train.negative" % domain_name
    return review_list(pos_file)+review_list(neg_file)

def unlabeled_reviews(domain_name):
    unlabeled_file = "../data/%s/train.unlabeled" % domain_name
    return review_list(unlabeled_file)

# return a list of reviews
def review_list(fname):
    return list([line.strip().split() for line in open(fname)])

# word embedding: from word to word vector
# Word2Vec
def word2vec_model(domain_name):
    model = gensim.models.Word2Vec(labeled_reviews(domain_name), min_count=1,workers=4)
    model.save('../work/%s/word2vec.model' % domain_name) 
    return model

def word2vec(feature):
    return 

# GloVe
def glo2ve(domain_name):
    corpus_model = Corpus()
    corpus_model.fit(labeled_reviews(domain_name), window=10)
    corpus_model.save('../work/%s/corpus.model'% domain_name)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')
    model = glove.Glove(no_components=100, learning_rate=0.05)
    model.fit(corpus_model.matrix, epochs=int(10),
              no_threads=6, verbose=True)
    model.add_dictionary(corpus_model.dictionary)
    model.save('../work/%s/glove.model' % domain_name) 
    return

# PPMI: replace all negative values in PMI with zero
def ppmi(pmi_score):
    return 0 if pmi_score<0 else pmi_score

# f(Wk) = document frequency of Wk in SL / # documents in SL -
# document frequency of Wk in TU / # documents in TU
def df_diff(df_source,src_reviews,df_target,tgt_reviews):
    return df_source/src_reviews - df_target/tgt_reviews

def df_function(source,target):
    print 'loading objects...'
    df_source = load_grouped_obj(source,target,'x_src')
    df_target = load_grouped_obj(source,target,'x_un_tgt')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    tgt_reviews = load_grouped_obj(source,target,'un_tgt_reviews')
    features = load_grouped_obj(source,target,'sl_tu_features')

    print 'calculating...'

    print ''
    pass

def load_grouped_obj(source,target,name):
    with open("../../group-generation/%s-%s/obj/%s.pkl" % (source,target,name), 'rb') as f:
        return pickle.load(f)

def save_loop_obj(obj,dirname,name):
    with open(dirname+"%s.pkl" % name,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def save_grouped_obj(obj,source,target,name):
    with open("../../group-generation/%s-%s/obj/%s.pkl" % (source,target,name),'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

# preset methods
def collect_source_featrues():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for domain in domains:
        pos_src_features = features_list("../data/%s/train.positive" % domain)
        neg_src_features = features_list("../data/%s/train.negative" % domain)
        src_features = set(pos_src_features).union(set(neg_src_features))
        dirnames = glob.glob('../../group-generation/%s-*'%domain)
        for dirname in dirnames:
            print 'saving src_featrues in '+ dirname + '...'
            save_loop_obj(src_features,dirname,'src_features')   
    print 'Complete!!'
    pass

def collect_features():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            pos_src_features = sp.features_list("../data/%s/train.positive" % source)
            neg_src_features = sp.features_list("../data/%s/train.negative" % source)
            src_features = set(pos_src_features).union(set(neg_src_features))
            un_tgt_features = sp.features_list("../data/%s/train.unlabeled" % target)
            sl_tu_features = src_features.union(set(un_tgt_features))

            print 'saving sl_tu_features for %s-%s ...' % (source,target)
            save_grouped_obj(sl_tu_features,source,target,'sl_tu_features')
    pass


# main
if __name__ == "__main__":
    # collect_features()
    