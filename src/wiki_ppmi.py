import numpy
import os
import pickle

# return a list of ids that has top 2000 frequencies
# ids from ids ( no stop words in the list )
# freq from unigrams.sorted ( useded to determine choose which 2000 )
# ppmi from wik-ppmi ( use the ids_list genereated )
def select_top_k_words(k):
    id_file = open("../data/xia_data/ids","r")
    freq_file = open("../data/xia_data/unigrams.sorted","r")
    id_list = []
    l1 = {}
    # word_list = []
    for line in id_file:
        l1[line.split()[1]]=int(line.split()[0])
    # print len(l1)
    l2 = [line.split()[0] for line in freq_file]
    # print len(l2)

    for x in l2:
        if x in frozenset(l1.keys()):
            if len(id_list) < k:
                id_list.append(l1.get(x))
            else:
                break
                pass
    # print id_list
    return id_list

# get ppmi values
def ppmi_embedding_model(id_list):
    ppmi_file = open("../data/xia_data/wiki-ppmi","r")
    model = {}
    # print len(ppmi_file.readlines())
    for line in ppmi_file:
        splitLine = line.split()
        word = splitLine[0]
        embedding = []
        id_value = {}
        for val in splitLine[1:]:
            p = val.split(':')
            # print p
            id_value[int(p[0])] =float(p[1]) 
        embedding = [id_value.get(x,0) for x in id_list]
            # print len(embedding)
        if not len(embedding)==len(id_list):
            print word,len(embedding)
        # else:
        #     print word, embedding[:10]
        model[word] = embedding
        # print word,embedding
    print len(model)," words loaded!"
    return model

# save model
def save_wiki_obj(obj, name):
    filename = '../work/wiki/'+name + '.pkl'
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_wiki_obj(name):
    with open('../work/wiki/'+name + '.pkl', 'rb') as f:
        return pickle.load(f)

# main
if __name__ == "__main__":
    k = 2000
    save_wiki_obj(ppmi_embedding_model(select_top_k_words(k)),'wiki_ppmi_%d.model'%k)