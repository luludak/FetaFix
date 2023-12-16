from math import *
from decimal import Decimal
 
# Using code from https://github.com/saimadhu-polamuri/DataAspirant_codes/tree/master/Similarity_measures
def euclidean_distance(x,y):

    """ return euclidean distance between two lists """

    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

def manhattan_distance(x,y):

    """ return manhattan distance between two lists """

    return sum(abs(a-b) for a,b in zip(x,y))

def minkowski_distance(x,y,p_value):

    """ return minkowski distance between two lists """

    return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),
        p_value)

def nth_root(value, n_root):

    """ returns the n_root of an value """

    root_value = 1/float(n_root)
    return round (Decimal(value) ** Decimal(root_value),3)

def cosine_similarity(x,y):

    """ return cosine similarity between two lists """

    numerator = sum(a*b for a,b in zip(x,y))
    denominator = square_rooted(x)*square_rooted(y)
    return round(numerator/float(denominator),3)

def square_rooted(x):

    """ return 3 rounded square rooted value """

    return round(sqrt(sum([a*a for a in x])),3)

def jaccard_similarity(x,y):
    """ returns the jaccard similarity between two lists """
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality/float(union_cardinality)


def rbo(list1, list2, p=0.9):
    # TODO: Implement
    return 0.0
