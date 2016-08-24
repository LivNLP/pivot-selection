# landmark-based pivot selection method
import select_pivots as sp 
import numpy
import gensim,logging
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
    return

# GloVe
def glo2Ve(domain_name):
    model = glove.Glove(labeled_reviews(domain_name), d=50, alpha=0.75, x_max=100.0)
    #glove.fit(corpus.matrix, epochs=30, no_threads=4, verbose=True)
    print model
    return

# main
if __name__ == "__main__":
    domain_name = 'books'
    glo2Ve(domain_name)