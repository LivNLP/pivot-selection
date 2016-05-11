import math

# jaccard coefficient between a and b
def jaccard_coefficient(a, b):
    A = set([i[0] for i in a])
    B = set([i[0] for i in b])
    # print A&B
    return float(len(A & B))/float(len(A | B)) if float(len(A | B)) != 0 else 0

def kandall_rank_coefficient(a, b):

    pass