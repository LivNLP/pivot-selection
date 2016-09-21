#!/usr/bin/env python
# -*- coding: utf-8 -*-

# landmark-based pivot selection method
import select_pivots as sp 
import numpy
import pickle
import glob
import gensim,logging
from glove import Corpus, Glove
import glove
from cvxopt import matrix
from cvxopt.solvers import qp

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
    model.save('../work/%s/word2vec.model' % domain_name,ignore=[]) 
    return model

# trained by two domains: S_L and T_U
def word2vec(source,target):
    reviews = labeled_reviews(source) + unlabeled_reviews(target)
    model = gensim.models.Word2Vec(reviews, min_count=1,workers=4,size=300)
    model.save('../work/%s-%s/word2vec.model' % (source,target))
    return model

# works for both pretrained models
def word_to_vec(feature,model):
    return model[feature]

# GloVe
# trained by a single domain: S_L
def glove_single(domain_name):
    corpus_model = Corpus()
    corpus_model.fit(labeled_reviews(domain_name), window=10)
    corpus_model.save('../work/%s/corpus.model'% domain_name)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')
    model = Glove(no_components=300, learning_rate=0.05)
    model.fit(corpus_model.matrix, epochs=int(10),
              no_threads=6, verbose=True)
    model.add_dictionary(corpus_model.dictionary)
    model.save('../work/%s/glove.model' % domain_name) 
    return

# trained by two domains: S_L and T_U
def glove(source,target):
    reviews = labeled_reviews(source) + unlabeled_reviews(target)
    corpus_model = Corpus()
    corpus_model.fit(reviews, window=10)
    # corpus_model.save('../work/%s-%s/corpus.model'% (source,target))
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)
    print('Training the GloVe model')
    model = Glove(no_components=300, learning_rate=0.05)
    model.fit(corpus_model.matrix, epochs=int(10),
              no_threads=6, verbose=True)
    model.add_dictionary(corpus_model.dictionary)
    output_path = '../work/%s-%s/glove.model' % (source,target)
    model.save(output_path)
    # glove_to_word2vec(output_path,output_path+'.gensim')
    return model

# read a pretrained model to get the word vector
def load_pretrained_glove(gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        model[word] = embedding
    print "Done.",len(model)," words loaded!"
    return model

# filter of pretrained glove model to decrease the memory cost
def load_filtered_glove(source,target,gloveFile):
    print "Loading Glove Model"
    f = open(gloveFile,'r')
    model = {}
    filtered_features = load_grouped_obj(source,target,'filtered_features')
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = [float(val) for val in splitLine[1:]]
        if word in filtered_features:     
            model[word] = embedding
        if word.replace('.','__') in filtered_features:
            model[word.replace('.','__')] = embedding
    print "After filtering, ",len(model)," words loaded!"
    return model


# get GloVe word vector
def glove_to_word2vec(source,target):
    path = '../work/%s-%s/glove.model' % (source,target)
    model = Glove.load(path)
    print len(model.get_word_vector('good'))
    pass

def glove_to_vec(feature,model):
    return model.get_word_vector(feature)

# PPMI: replace all negative values in PMI with zero
def ppmi(pmi_score):
    return 0 if pmi_score < 0 else pmi_score

# gamma function: PPMI, other pivot selection methods can be also used
def gamma_function(source,target):
    print 'loading objects...'
    features = load_grouped_obj(source,target,'filtered_features')
    x_src = load_grouped_obj(source,target,'x_src')
    x_pos_src = load_grouped_obj(source,target,'x_pos_src')
    x_neg_src = load_grouped_obj(source,target,'x_neg_src')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    pos_src_reviews = load_grouped_obj(source,target,'pos_src_reviews')
    neg_src_reviews = load_grouped_obj(source,target,'neg_src_reviews')

    ppmi_dict = {}
    for x in features:
        if x_src.get(x,0)*x_pos_src.get(x,0)*x_neg_src.get(x,0) > 0:
            pos_pmi = sp.pointwise_mutual_info(x_src.get(x,0), x_pos_src.get(x,0), pos_src_reviews, src_reviews) 
            neg_pmi = sp.pointwise_mutual_info(x_src.get(x,0), x_neg_src.get(x,0), neg_src_reviews, src_reviews)
            ppmi_dict[x] = (ppmi(pos_pmi)-ppmi(neg_pmi))**2
        else:
            ppmi_dict[x] = 0
    L = ppmi_dict.items()
    
    print 'sorting...'
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)

    dirname = '../work/%s-%s/obj/'% (source,target)
    print 'saving ppmi_dict in ' + dirname
    save_loop_obj(ppmi_dict,dirname,'ppmi_dict')
    return L

# f(Wk) = document frequency of Wk in S_L / # documents in S_L -
# document frequency of Wk in T_U / # documents in T_U
def df_diff(df_source,src_reviews,df_target,tgt_reviews):
    return df_source/src_reviews - df_target/tgt_reviews

# uk = f(Wk) * vector(Wk)
# word2vec model
def u_function(source,target):
    print 'loading objects...'
    df_source = load_grouped_obj(source,target,'x_src')
    df_target = load_grouped_obj(source,target,'x_un_tgt')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    tgt_reviews = load_grouped_obj(source,target,'un_tgt_reviews')
    features = load_grouped_obj(source,target,'filtered_features')
    model = gensim.models.Word2Vec.load('../work/%s-%s/word2vec.model' % (source,target))

    print 'calculating...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_reviews,df_target.get(x,0),tgt_reviews)
        x_vector = word_to_vec(x,model)
        u_dict[x] = numpy.dot(df_function,x_vector)

    dirname = '../work/%s-%s/obj/'% (source,target)
    print 'saving u_dict in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict')
    print 'u_dict saved'
    pass

# glove model
def u_function_glove(source,target,model):
    print 'loading objects...'
    df_source = load_grouped_obj(source,target,'x_src')
    df_target = load_grouped_obj(source,target,'x_un_tgt')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    tgt_reviews = load_grouped_obj(source,target,'un_tgt_reviews')
    features = load_grouped_obj(source,target,'filtered_features')

    print 'calculating with glove model...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_reviews,df_target.get(x,0),tgt_reviews)
        x_vector = glove_to_vec(x,model)
        u_dict[x] = numpy.dot(df_function,x_vector)

    dirname = '../work/%s-%s/obj/'% (source,target)
    print 'saving u_dict_glove in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict_glove')
    print 'u_dict_glove saved'
    pass

# pretrained models with domain specific model support
# Word2Vec
def u_function_pretrained(source,target,model):
    print 'loading objects...'
    df_source = load_grouped_obj(source,target,'x_src')
    df_target = load_grouped_obj(source,target,'x_un_tgt')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    tgt_reviews = load_grouped_obj(source,target,'un_tgt_reviews')
    features = load_grouped_obj(source,target,'filtered_features')
    ds_model = gensim.models.Word2Vec.load('../work/%s-%s/word2vec.model' % (source,target))

    print 'calculating with pretrained word2vec model...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_reviews,df_target.get(x,0),tgt_reviews)
        if x in model.vocab:
            x_vector = word_to_vec(x,model)
        else:
            if x.replace('__','_') in model.vocab:
                # print x.replace('__','_')
                x_vector = word_to_vec(x.replace('__','_'),model)
            else:
                x_vector = word_to_vec(x,ds_model)
        u_dict[x] = numpy.dot(df_function,x_vector)

    dirname = '../work/%s-%s/obj/'% (source,target)
    print 'saving u_dict_pretrained in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict_pretrained')
    print 'u_dict_pretrained saved'
    pass

# GloVe
def u_function_pretrained_glove(source,target,model):
    print 'loading objects...'
    df_source = load_grouped_obj(source,target,'x_src')
    df_target = load_grouped_obj(source,target,'x_un_tgt')
    src_reviews = load_grouped_obj(source,target,'src_reviews')
    tgt_reviews = load_grouped_obj(source,target,'un_tgt_reviews')
    features = load_grouped_obj(source,target,'filtered_features')
    dirname = '../work/%s-%s/'% (source,target)
    ds_model = Glove.load(dirname+'glove.model')

    print 'calculating with pretrained glove model...'
    u_dict = {}
    for x in features:
        df_function = df_diff(df_source.get(x,0),src_reviews,df_target.get(x,0),tgt_reviews)
        if model.get(x,0)==0:
            x_vector = glove_to_vec(x,ds_model)
        else:
            x_vector = word_to_vec(x,model)
        u_dict[x] = numpy.dot(df_function,x_vector)

    dirname = '../work/%s-%s/obj/'% (source,target)
    print 'saving u_dict_pretrained in ' + dirname
    save_loop_obj(u_dict,dirname,'u_dict_pretrained_glove')
    print 'u_dict_pretrained saved'
    pass


# optimization: QP
def qp_solver(Uk,Rk,param):
    print "u and gamma length: %d, %d" %(len(Uk),len(Rk))
    U = numpy.matrix(sort_by_keys(Uk).values())
    # T = numpy.transpose(U)
    R = sort_by_keys(Rk).values()

    P = numpy.dot(2,U*U.T)
    P = P.astype(float) 
    # print "%d" % len(P)
    print Rk.keys()==Uk.keys()
    q = numpy.dot(-param,R)
    n = len(q)
    G = matrix(0.0, (n,n))
    G[::n+1] = -1.0 
    # G = matrix(numpy.identity(n))
    A = matrix(1.0,(1,n))
    h = matrix(0.0,(n,1),tc='d')
    b = matrix(1.0,tc='d')

    solver = qp(matrix(P),matrix(q),G,h,A,b)
    alpha = matrix_to_array(solver['x'])
    s = two_lists_to_dictionary(Uk.keys(),alpha)
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    return L

def opt_function(dirname,param,model_name,pretrained):
    print 'loading objects...'
    ppmi_dict = load_loop_obj(dirname,'ppmi_dict')
    if pretrained == 0:
        if model_name == 'word2vec':
            u_dict = load_loop_obj(dirname,'u_dict')
        else:
            u_dict = load_loop_obj(dirname,'u_dict_glove')
    else:
        if model_name == 'word2vec':
            u_dict = load_loop_obj(dirname,'u_dict_pretrained')
        else:
            u_dict = load_loop_obj(dirname,'u_dict_pretrained_glove')

    print 'solving QP...'
    alpha_dict = qp_solver(u_dict,ppmi_dict,param)
    return alpha_dict

# Selecting pivots
# alpha in [0,1], larger is more close to be a landmark (pivot)
def select_pivots_by_alpha(source,target,param,model,pretrained,paramOn):
    temp = 'landmark' if pretrained == 0 else 'landmark_pretrained'
    method = method_name_param(temp,model,param) if paramOn==True else method_name(temp,model,param)
    dirname = '../work/%s-%s/obj/'% (source,target)
    L = opt_function(dirname,param,model,pretrained)
    save_loop_obj(L,dirname,method)
    print '%s saved' % method
    # print L[:5]# test
    return L

# helper method
def sort_by_keys(dic):
    dic.keys().sort()
    return dic

def remove_low_freq_feats(old_dict,new_keys):
    new_dict = {new_key:old_dict[new_key] for new_key in new_keys}
    return new_dict

def freq_keys(source,target,limit):
    src_freq = {}
    tgt_freq = {}
    sp.count_freq("../data/%s/train.positive" % source, src_freq)
    sp.count_freq("../data/%s/train.negative" % source, src_freq)
    sp.count_freq("../data/%s/train.unlabeled" % target, tgt_freq) 
    s = {}
    features = set(src_freq.keys()).union(set(tgt_freq.keys()))
    for feat in features:
        temp = min(src_freq.get(feat, 0), tgt_freq.get(feat, 0))
        if temp > limit:
            s[feat] = temp
    L = s.items()
    L.sort(lambda x, y: -1 if x[1] > y[1] else 1)   
    return s.keys()

def matrix_to_array(M):
    return numpy.squeeze(numpy.asarray(M))

def two_lists_to_dictionary(keys,values):
    return dict(zip(keys, values))

def method_name(method,word_model,param):
    if param > 10e-3:
        return '%s_%s_ppmi'%(method,word_model)
    else:
        return '%s_%s'%(method,word_model)
    pass

def method_name_param(method,word_model,param):
    return 'test_%s_%s_%f'%(method,word_model,param)

# save and load objects
def load_loop_obj(dirname,name):
    with open(dirname+"%s.pkl" % name,'rb') as f:
        return pickle.load(f)

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

def collect_filtered_features(limit):
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            filtered_features = freq_keys(source,target,limit)
            print 'length: %d'% len(filtered_features)
            print 'saving filtered_features for %s-%s ... ' % (source,target)
            save_grouped_obj(filtered_features,source,target,'filtered_features')
    pass

def create_word2vec_models():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'creating word2vec model for %s-%s ...' % (source,target) 
            word2vec(source,target)
    print '-----Complete!!-----'
    pass

def create_glove_models():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'creating GloVe model for %s-%s ...' % (source,target) 
            model = glove(source,target)
            print 'calculating u for %s-%s ...' % (source,target)
            u_function_glove(source,target,model)
    print '-----Complete!!-----'
    pass

def calculate_all_u_pretrained():
    # load pretrained model here
    path = '../data/GoogleNews-vectors-negative300.bin'
    model = gensim.models.Word2Vec.load_word2vec_format(path,binary=True)
    # print model.most_similar('very')
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'calcualting u_pretrained for %s-%s ...' % (source,target)
            u_function_pretrained(source,target,model) 
    print '-----Complete!!-----'
    pass

def calculate_all_u_pretrained_glove():
    # load pretrained model here
    path = '../data/glove.42B.300d.txt'
    # model = load_pretrained_glove(path)
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'calcualting u_pretrained for %s-%s ...' % (source,target)
            model = load_filtered_glove(source,target,path)
            u_function_pretrained_glove(source,target,model) 
    print '-----Complete!!-----'
    pass

def calculate_all_u():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'calcualting u for %s-%s ...' % (source,target)
            u_function(source,target) 
    print '-----Complete!!-----'
    pass

def compute_all_gamma():
    domains = ["books", "electronics", "dvd", "kitchen"]
    for source in domains:
        for target in domains:
            if source ==target:
                continue
            print 'computing gamma for %s-%s ...' % (source,target)
            gamma_function(source,target)
    print '-----Complete!!-----'
    pass

def store_all_selections(params,model,pretrained,paramOn):
    domains = ["books", "electronics", "dvd", "kitchen"]
    for param in params:
        for source in domains:
            for target in domains:
                if source ==target:
                    continue
                print 'getting alpha from %s-%s ...' % (source,target)
                select_pivots_by_alpha(source,target,param,model,pretrained,paramOn)
                print '------selection completed--------' 
    pass

# test methods
def solve_qp():
    source = 'books'
    target = 'dvd'
    model_name = 'word2vec'
    param = 1
    dirname = '../work/%s-%s/obj/'% (source,target)
    opt_function(dirname,param,model_name,1)
    pass

def construct_freq_dict():
    source = 'books'
    target = 'electronics'
    limit = 5
    print len(freq_keys(source,target,limit))
    pass

def print_alpha(param):
    source = 'books'
    target = 'dvd'
    param = param
    model = 'word2vec'
    # model = 'glove'
    pretrained = 1
    paramOn=True
    # paramOn=False
    dirname = '../work/%s-%s/obj/'% (source,target)
    temp = 'landmark' if pretrained == 0 else 'landmark_pretrained'
    method = method_name_param(temp,model,param) if paramOn==True else method_name(temp,model,param)
    alpha = load_loop_obj(dirname,method)
    for x,score in alpha[:5]:
        print x,score
    pass

def glove_model_test():
    source = 'books'
    target = 'dvd'
    # glove(source,target)
    glove_to_word2vec(source,target)
    pass

def read_glove():
    path = '../data/glove.42B.300d.txt'
    # model = load_pretrained_glove(path)
    source = 'books'
    target = 'dvd'
    model = load_filtered_glove(source,target,path)   
    # print len(model['good'])
    pass

def read_word2vec():
    path = '../data/GoogleNews-vectors-negative300.bin'
    model = gensim.models.Word2Vec.load_word2vec_format(path,binary=True)
    print len(model['good'])
    pass

def print_ppmi():
    source = 'books'
    target = 'dvd'
    dirname = '../work/%s-%s/obj/'% (source,target)
    ppmi_dict = load_loop_obj(dirname,'ppmi_dict')
    L = ppmi_dict.items()
    # L.sort(lambda x, y: -1 if x[1] > y[1] else 1)
    for x,score in L[:10]:
        print x,ppmi_dict.get(x,0)
    pass


# main
if __name__ == "__main__":
    # collect_filtered_features(5)
    # collect_features()
    # create_word2vec_models()
    # create_glove_models()
    # calculate_all_u_pretrained()
    # calculate_all_u_pretrained_glove()
    # calculate_all_u()
    # compute_all_gamma()
    params = [0]
    model_names = ['word2vec','glove']
    # ######param#########
    # params = [50,100,1000,10000]
    # params = [0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2]
    # model_names = ['word2vec']
    # model_names = ['glove']
    paramOn = True
    # paramOn = False
    for model in model_names:
        store_all_selections(params,model,1,paramOn)
    ######test##########
    # solve_qp() 
    # construct_freq_dict()
    # print_alpha(1)
    # glove_model_test()
    # read_glove()
    # read_word2vec()
    # print_ppmi()
