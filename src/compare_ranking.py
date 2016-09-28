import math
import itertools

# jaccard coefficient between a and b
def jaccard_coefficient(a, b):
    A = set([i[0] for i in a])
    B = set([i[0] for i in b])
    return float(len(A & B))/float(len(A | B)) if float(len(A | B)) != 0 else 0

# kendall rank correlation coefficient between a and b
def kendall_rank_coefficient(a, b):
    # A = [i[0] for i in a]
    # B = [i[0] for i in b]
    # all_objects = list(set(A) | set(B))
    # all_pairs = list(itertools.combinations(all_objects, 2))
    # # iterate all pairs to compute the kendall distance
    # kendall_distance = 0.0
    # for (i, j) in all_pairs:
    #     # 1. both in A and B
    #     if i in A and i in B and j in A and j in B:
    #         # print i,j +' group 1'
    #         # if i and j rank is in opposite order, add 1 to the distance
    #         if A.index(i) > A.index(j) and B.index(i) < B.index(j) or A.index(i) < A.index(j) and B.index(i) > B.index(j):
    #             kendall_distance += 1
    #     # 2. both in A or B but only i or j in B or A
    #     if i in A and j in A and i in B and j not in B:
    #         # print i,j +' group 2.1'
    #         if not(A.index(i) < A.index(j)):
    #             kendall_distance += 1
    #     if i in A and j in A and j in B and i not in B:
    #         # print i,j +' group 2.2'
    #         if not(A.index(j) < A.index(i)):
    #             kendall_distance += 1
    #     if i in B and j in B and i in A and j not in A:
    #         # print i,j +' group 2.3'
    #         if not(B.index(i) < B.index(j)):
    #             kendall_distance += 1
    #     if i in B and j in B and j in A and i not in A:
    #         # print i,j +' group 2.4'
    #         if not(B.index(j) < B.index(i)):
    #             kendall_distance += 1
    #     # 3. i(not j) in A and j(not i) in B or reverse
    #     if i in A and j not in A and j in B and i not in B or j in A and i not in A and i in B and j not in B:
    #         # print i,j +' group 3'
    #         kendall_distance += 1
    #     # 4. i and j in A or B but neither in B or A, add a neutral value 0.5 to distance      
    #     if i in A and j in A and i not in B and j not in B or i in B and j in B and i not in A and j not in A:
    #         # print i,j +' group 4'
    #         kendall_distance += 0.5 #optimistic approach 0, neutral approach 0.5

    A = [i[0] for i in a]
    B = [i[0] for i in b]
    all_objects = set(A).intersection(set(B))
    all_pairs = list(itertools.combinations(all_objects, 2))
    kendall_distance = 0.0
    for (i,j) in all_pairs:
        if ((A.index(i) < A.index(j) and B.index(i) > B.index(j)) or (A.index(i) > A.index(j) and B.index(i) < B.index(j))):
            kendall_distance += -1.0
        if ((A.index(i) < A.index(j) and B.index(i) < B.index(j)) or (A.index(i) > A.index(j) and B.index(i) > B.index(j))):
            kendall_distance += 1.0
    # print all_pairs
    # print kendall_distance
    return kendall_distance/float(len(all_pairs)) if kendall_distance !=0 else 0.0


# test method
if __name__ == "__main__":
    # a = [('a',1),('b',2),('c',3),('d',9)]
    # b = [('a',1),('d',1),('b',2),('f',8)]
    # print kendall_rank_coefficient(a,b)
    pass