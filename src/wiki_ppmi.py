import numpy

# return a list of ids that has top 2000 frequencies
# ids from ids ( no stop words in the list )
# freq from unigrams.sorted ( useded to determine choose which 2000 )
# ppmi from wik_ppmi ( use the ids_list genereated )
def select_top_k_words(k):
    id_file = open("../data/xia_data/ids","r")
    freq_file = open("../data/xia_data/unigrams.sorted","r")
    id_list = []
    l1 = [line.strip.split()[1] for line in id_file]
    print l1
    l2 = [line.strip.split()[0] for line in freq_file]
    pirnt l2
    non_stop_words = list(set(l1) & set(l2))
    word_list = [word for word in l2 if word in non_stop_words][:k]
    for word in word_list:
        id_list.append(get_id(word))
    print id_list
    return id_list
def get_id(word):
    return [line.strip.split()[0] for line in id_file if line.strip.split()[0]==word][0]


# main
if __name__ == "__main__":
    select_top_k_words(10)