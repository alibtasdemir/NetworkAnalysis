"""
Original Code:
https://github.com/eelenberg/characterNetworks/blob/master/generateGraphsMain.py
"""

import numpy as np
import pandas as pd
from itertools import combinations
import io
from snap import GenPrefAttach, SaveEdgeList, TRnd
import subprocess
import  math
import argparse as arg


def getDegreeList(A):
    # n = np.unique(np.vstack((A[:,0],A[:,1]))).shape[0]
    n = int(np.max(np.vstack((A[:, 0], A[:, 1]))) + 1)
    degreeVec = np.zeros(n, dtype=int)
    for e in range(A.shape[0]):
        degreeVec[int(A[e, 0])] += 1
        degreeVec[int(A[e, 1])] += 1
    return degreeVec
def initializeDirectory(origGraph):

    out = subprocess.check_output(["mkdir", "graphs"])
    hdr = "#graph\tsample_prob_keep\tn3_3\tn3_2\tn3_1\tn3_0\tn4_0\tn4_1\tn4_2\tn4_3\tn4_4\tn4_5\tn4_6\tn4_7\tn4_8\tn4_9\tn4_10\truntime\n"
    with open('counts_4_profilesLocal.txt', 'w') as fpt:
        fpt.write(hdr)
    pcommand = '/Users/ethan/graphlab-master-2/release/apps/4-profiles/4profile'
    out = subprocess.check_output([pcommand, "--format", "tsv", "--graph", origGraph])
    hdr2 = "#graph\tevbin0\tevbin1\tevbin2\tevbin3\tevbin4\n"
    with open('counts_eval_bins.txt', 'w') as fpt:
        fpt.write(hdr2)
    generateEigenvalueBins(origGraph,"counts_eval_bins.txt")
    return 0

def makeWeightedEdgelist(A, outname):
    # still remove self loops, as they make no sense in this context
    Atmp = np.array([row for row in A if row[0] != row[1]])
    inds = np.lexsort((Atmp[:, 1], Atmp[:, 0]))
    Asort = Atmp[inds, :]
    # get number of unique entries by taking diff
    Adiff1 = np.vstack((np.array([1, 1]), np.diff(Asort, axis=0)))
    Adiff = np.any(Adiff1 != 0, axis=1)
    # find where the diffs are equal to 1 a and diff that to get counts of unique
    outUnique = Asort[Adiff == 1]
    outCounts = np.diff(np.hstack((np.where(Adiff == 1)[0], Adiff.shape[0])))
    out = np.column_stack((outUnique, outCounts))
    if outname:
        np.savetxt(outname, out, fmt=('%d', '%d', '%d'), delimiter='\t', comments='')
    return out


def removeDuplicateEdges(X):
    # remove duplicates and self loops (and also sort)
    # xtmp = np.vstack({tuple(row) for row in X})
    xtmp = np.vstack({tuple(row) for row in X if row[0] != row[1]})
    inds = np.lexsort((xtmp[:, 1], xtmp[:, 0]))
    out = xtmp[inds, :]
    return out


def myPA(nodes, m, seed=4639):
    np.random.seed(seed)
    edgeList = []
    degreeVec = np.zeros(nodes)
    # initialize first step
    degreeVec[0:2] = np.array([1, 1])
    edgeList.append((0, 1))
    for n in np.arange(2, nodes):
        # connect to existing vertices according to preferential attachment model
        # weighting of distribution is degreeVec[:n]
        probs = np.double(degreeVec[:n])
        neighbors = np.random.choice(np.arange(n), m, replace=True, p=probs / np.sum(probs))
        # print neighbors
        degreeVec[n] = m
        for dit in np.arange(m):
            # if edge included, increment both degrees and append edge to the list
            degreeVec[neighbors[dit]] += 1
            edgeList.append((neighbors[dit], n))
        # print degreeVec
        # print "avg degree: " + str(np.sum(degreeVec)/n)
    return np.asarray(edgeList)


def generateGraphs(params):
    graphname = params['graph']
    n = int(params['n'])
    numit = int(params['numGen'])
    graphType = params['type']

    if graphType == 'GNP':
        deg = int(params['d'])
        # every node has average degree deg, total number of edges is deg*n/2, divide by total possible edges 2/(n*(n-1))
        p = float(deg) / (n - 1)
        # print "degree is " + str(p)
        np.random.seed(4639)
        # generate all randomness at once
        pairs = np.array([t for t in combinations(np.arange(n), 2)])
        ps = np.random.rand(pairs.shape[0], numit) <= p
        for it in np.arange(numit):
            # keep the edges that are sampled
            pairsKeep = pairs[ps[:, it] == 1]
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            np.savetxt(outname, pairsKeep, fmt=('%d', '%d'), delimiter='\t', comments='')




    elif graphType == 'PA':
        deg = int(params['d'])
        for it in np.arange(numit):
            #is this degree right? or scale by 2
            #solve directly: 2/n + 2m = deg = 2|E|/n
            # x = myPA(n, int(deg-2./n), seed=it*4639+5011)
            x = myPA(n, int(deg/2.-1./n), seed=it*4639+5011)
            # x = myPA(n, int(deg/2.), seed=it*4639+5011)
            tmpname = graphname + '_' + graphType + '_' + str(it) + '_dup.txt'
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            # outname = graphname + '_' + graphType + 'mult_' + str(it) + '.txt'
            # makeWeightedEdgelist(x,tmpname)
            # np.savetxt(tmpname,x,fmt=('%d','%d'),delimiter='\t',comments='')
            xfinal = removeDuplicateEdges(x)
            np.savetxt(outname,xfinal,fmt=('%d','%d'),delimiter='\t',comments='')

    elif graphType == 'CNFG':
        w = params['dList']
        wnorm = np.sum(w)
        elist = np.zeros(wnorm)
        st = 0
        for i, wi in enumerate(w):
            elist[st:(st + wi)] = i
            st += wi
        for it in np.arange(numit):
            plist = np.random.permutation(elist)
            x = plist.reshape(-1, 2)
            # if column 1 is greater than column 0 then swap that column
            xswap = x[:, 0] > x[:, 1]
            x[xswap, 0:2] = np.column_stack((x[xswap, 1], x[xswap, 0]))
            tmpname = graphname + '_' + graphType + '_' + str(it) + '_wt.txt'
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            # sort correctly and remove self loops, duplicates
            xweighted = makeWeightedEdgelist(x, tmpname)
            np.savetxt(outname, xweighted[:, :2], fmt=('%d', '%d'), delimiter='\t', comments='')

    elif graphType == 'CL':
        # get degree sequence from input
        w = params['dList']
        wnorm = float(np.sum(w))
        nc2 = n * (n - 1) // 2
        pairs = np.zeros((nc2, 2))
        pairComp = np.zeros(nc2)
        for e, (i, j) in enumerate(combinations(np.arange(n), 2)):
            # array comparison
            pairComp[e] = w[i] * w[j] / wnorm
            pairs[e, 0] = i
            pairs[e, 1] = j
        rands = np.random.rand(nc2, numit)
        for it in np.arange(numit):
            pairsKeep = pairs[rands[:, it] < pairComp]
            outname = graphname + '_' + graphType + '_' + str(it) + '.txt'
            np.savetxt(outname, pairsKeep, fmt=('%d', '%d'), delimiter='\t', comments='')




if __name__ == '__main__':
    # Defined some variables to keep parameters.
    parser = arg.ArgumentParser(description="Ellenberg Random Graph Generation")
    parser.add_argument('input', help="The input file")
    parser.add_argument('output', help='The output file')
    parser.add_argument('algorithm', type=int, help='Generation Algorithm: \n\t0. GNP\n\t1. PA\n\t2. CL\n\t3. CNFG')
    parser.add_argument('-n', '--vertex', type=int, help='Number of Vertex')
    parser.add_argument('-e', '--edge', type=int, help='Number of Edges')
    parser.add_argument('-p', '--prob', type=float, help='Probability value.')
    parser.add_argument('-k', '--iter', type=int, help="Number of generated graphs.")
    args = parser.parse_args()
    
    # Default value of node number.
    n = 100
    if args.vertex:
        n = args.vertex
        
    # Default value of edge number.
    E = 300
    if args.edge:
        E = args.edge
    
    # Default value of probability number.
    p = float(2*1231/155)
    if args.prob:
        p = args.prob
    
    # Default value of iteration number.
    k = 100
    if args.iter:
        k = args.iter
    
    filename, outname =  args.input, args.output
    A = np.loadtxt(filename)
    degreeVec = getDegreeList(A)

    algorithm = args.algorithm
    
    print("Progress started with parameters:\n\tInput: %s\n\tOutput: %s\n\tAlgorithm: %d\n\tNode: %d\n\tEdge: %d\n\tProbability: %f\n\tIteration: %d"%(filename, outname, algorithm, n, E, p, k))
    
    # User chooses an algorithm to generate random graphs.
    # 0 -  The Binomial Random Graph
    # 1 -  Preferential Attachment
    # 2 -  The Chung-Lu model
    # 3 -  The Configuration Model
    if algorithm == 0:
        params = {'graph': 'output', 'type': 'GNP', 'n': n, 'd': p, 'numGen': k}
    elif algorithm == 1:
        params = {'graph': 'twilight','type':'PA','n': n,'d': p,'numGen': k}
    elif algorithm == 2:
        params = {'graph': 'output', 'type': 'CL', 'n': n, 'dList': degreeVec, 'numGen': k}
    elif algorithm == 3:
        params = {'graph': 'output', 'type': 'CNFG', 'n': n, 'dList': degreeVec, 'numGen': k}
    else:
        print("Algorithm not exists!")
        exit()
    
    generateGraphs(params)