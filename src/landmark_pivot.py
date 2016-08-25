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
# trained by a single domain: S_L
def word2vec_single(domain_name):
    model = gensim.models.Word2Vec(labeled_reviews(domain_name), min_count=1,workers=4)
    model.save('../work/%s/word2vec.model' % domain_name) 
    return model

# trained by two domains: S_L and T_U
def word2vec(source,target):
    reviews = labeled_reviews(source) + unlabeled_reviews(target)
    model = gensim.models.Word2Vec(reviews, min_count=1,workers=4)
    model.save('../work/%s-%s/word2vec.model' % (source,target))
    return model

def word_to_vec(feature,model):
    return model[feature]

# GloVe
# trained by a single domain: S_L
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
    return 0 if pmi_score < 0 else pmi_score

# gamma function: PPMI
def gamma_function():
    
    pass

# f(Wk) = document frequency of Wk in S_L / # documents in S_L -
# document frequency of Wk in T_U / # documents in T_U
def df_diff(df_source,src_reviews,df_target,tgt_reviews):
    return df_source/src_reviews - df_target/tgt_reviews

# uk = f(Wk) * vector(Wk)
def u_function(source,target):
    print 'loading objects...'
    df_source = load_grouped_obj(source,target,'x_src')
    df_target = load_grouped_obj(source,target,'x_un_tgt')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    tgt_reviews = load_grouped_obj(source,target,'un_tgt_reviews')
    features = load_grouped_obj(source,target,'sl_tu_features')
    word2vec_model = gensim.models.Word2Vec.load('../work/%s-%s/word2vec.model' % (source,target))

    print 'calculating...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_reviews,df_target.get(x,0),tgt_reviews)
        x_vector = word_to_vec(x,word2vec_model)
        u_dict[x] = df_function * x_vector

    dirname = '../work/%s-%s/obj/'% (source,target)
    print 'saving u_dict in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict')
    print 'u_dict saved'
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

def create_word2vec_models():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'creating word2vec model for %s-%s ...' % (source,target) 
            word2vec(source,target)
    print 'Complete!!'
    pass

def calculate_all_u():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'calcualting u for %s-%s ...' % (source,target)   
            u_function(source,target) 
    print 'Complete!!'
    pass

# main
if __name__ == "__main__":
    # collect_features()
    # create_word2vec_models()
    calculate_all_u()
