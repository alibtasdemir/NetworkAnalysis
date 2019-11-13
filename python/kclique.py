from itertools import combinations
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import argparse as arg


# Calculates the clique numbers
# with complexity of O(n^k) (Brute Force)
# Ref: https://medium.com/100-days-of-algorithms/day-64-k-clique-c03fdc565b1e
def k_cliques(graph):
    # 2-cliques
    cliques = [{i, j} for i, j in graph.edges() if i != j]
    k = 2
    
    while cliques:
        # result
        yield k, cliques
        
        # merge k-cliques into (k+1)-cliques
        cliques_1 = set()
        for u, v in combinations(cliques, 2):
            w = u ^ v
            if len(w) == 2 and graph.has_edge(*w):
                cliques_1.add(tuple(u | w))

        # remove duplicates
        cliques = list(map(set, cliques_1))
        k += 1


# Prints the output of the function k_cliques
def print_cliques(graph):
    for k, cliques in k_cliques(graph):
        print('%d-cliques: #%d, %s ...' % (k, len(cliques), cliques[:3]))


# Read Graphs as Edge List
# With format of:
# Node1 Node2
def readInput(path):
    return nx.read_edgelist(path)


# G=nx.read_edgelist("graphs/test.edges")
# G_large = nx.read_edgelist("graphs/test-large.edges")
def maximalClique(G, k=2, out=False):
    # Finding the Maximal Cliques associated with teh graph
    a = nx.find_cliques(G)
    if out:
        i = 0
        # For each clique, print the members and also print the total number of communities
        for clique in a:
            print(clique)
            i+=1
        total_comm_max_cl = i
        print('Total number of communities: ',total_comm_max_cl)
    
    cliques=[clique for clique in nx.find_cliques(G) if len(clique)>k]
    print('(Maximal Clique Number)%d-cliques: %d'%(k, len(cliques)))


if __name__ == "__main__":
    parser = arg.ArgumentParser(description="Test")
    parser.add_argument('input', help="The input file")
    parser.add_argument('-k', '--kclique', type=int, help='K-clique number')
    args = parser.parse_args()
    
    file_path = args.input
    kclique = 2
    if args.kclique:
        kclique = args.kclique

    G = readInput(file_path)
    maximalClique(G, k=kclique)