import math
import itertools

# jaccard coefficient between a and b
def jaccard_coefficient(a, b):
    A = set([i[0] for i in a])
    B = set([i[0] for i in b])
    return float(len(A & B))/float(len(A | B)) if float(len(A | B)) != 0 else 0

# kendall rank correlation coefficient between a and b
def kendall_rank_coefficient(a, b):
    A = [i[0] for i in a]
    B = [i[0] for i in b]
    all_objects = list(set(A) & set(B))
    all_pairs = list(itertools.combinations(all_pairs, 2))
    # iterate all pairs to compute the kendall distance
    kendall_distance = 0.0
    for (i, j) in all_pairs:
        # 1. both in A and B
        if i in A and B and j in A and B:
            # if i and j rank is in opposite order, add 1 to the distance
            if A.index(i) > A.index(j) and B.index(i) < B.index(j) or A.index(i) < A.index(j) and B.index(i) > B.index(j):
                kendall_distance += 1
        # 2. both in A or B but only i or j in B or A
        # 3. i(not j) in A and j(not i) in B or reverse
        # 4. i and j in A or B but neither in B or A, add a netural value 0.5 to distance
    pass