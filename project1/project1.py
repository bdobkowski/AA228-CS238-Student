#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
import scipy.special as sp


class Variable:
    def __init__(self, values):
        self.values = values

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile, outfile):
    data   = pd.read_csv(infile)
    DG     = nx.DiGraph()
    idx2names = dict(zip(range(len(data.columns)), list(data.columns))) 
    variables = dict(zip(range(len(data.columns)),np.zeros(len(data.columns))))
    for i in variables:
        variables[i] = len(pd.unique(data[idx2names[i]]))
    DG.add_nodes_from(idx2names)
    DG.add_edge(0,1)
    DG.add_edge(2,1)
    # for child in DG.successors(0):
    #     print(child)
    # for parent in DG.predecessors(0):
    #     print(parent)
    # np.prod([[3,2] for parent in DG.predecessors(0)])
    # for col in data.columns:
    #     print(pd.unique(data[col]))
    bs = bayesian_score(DG, data, idx2names, variables)
    print(bs)
    plot_graph(DG)
    write_gph(DG, idx2names, outfile)
    import pdb;pdb.set_trace()
    pass

def bayesian_score(graph, data, idx2names, variables):
    names2idx = {v: k for k, v in idx2names.items()}
    alpha = prior(variables, graph)
    M = statistics(variables, graph, data, idx2names)
    n = len(variables)
    return np.sum([bayesian_score_component(M[i], alpha[i]) for i in range(n)])
    
def bayesian_score_component(M, alpha):
    p  = np.sum(sp.loggamma(alpha + M))
    p -= np.sum(sp.loggamma(alpha))
    p += np.sum(sp.loggamma(np.sum(alpha, axis=1)))
    p -= np.sum(sp.loggamma(np.sum(alpha, axis=1) + np.sum(M, axis=1)))
    return p
    
def prior(variables,G):
    n=len(variables)
    r=[variables[i] for i in range(n)]
    q=[int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
    return [np.ones((q[i],r[i])) for i in range(n)]

def statistics(variables,G,df, idx2names):
    n=len(df.columns)
    r=[variables[i] for i in range(n)]
    q=[int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
    M=[np.zeros((q[i],r[i])) for i in range(n)]
    for index, row in df.iterrows():
        for i in range(n):
            k=row[idx2names[i]] - 1
            parents=[pred for pred in G.predecessors(i)]
            j=np.array([0])
            parent_list = []
            for parent in parents: # checking if there are any parents
                parent_list.append(parent)
            parents_array = np.array([x for x in parent_list])
            dims = ()
            for p in parents_array:
                dims += (r[p],)
            for parent in parents:
                row_sub = np.array([[row[idx2names[x]] - 1] for x in parent_list])
                j=np.ravel_multi_index(row_sub, dims, order='F')
                break
            M[i][j,k]+=1.0
    return M

def sub2ind(siz, x):
    k = np.vstack((0, np.cumprod(siz)))
    return np.dot(k,x-1)+1

def plot_graph(graph):
    ax = plt.subplot()
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")
    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
