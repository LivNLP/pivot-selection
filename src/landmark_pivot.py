# landmark-based pivot selection method
import select_pivots as sp 
import numpy
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
def word2vec(domain_name):
    model = gensim.models.Word2Vec(labeled_reviews(domain_name), min_count=1,workers=4)
    print model
    return model

# GloVe
def glo2Ve(domain_name):
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
def df_diff():
    return

# main
if __name__ == "__main__":
    domain_name = 'books'