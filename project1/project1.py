#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt


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
    for child in DG.successors(0):
        print(child)
    for parent in DG.predecessors(0):
        print(parent)
    np.prod([[3,2] for parent in DG.predecessors(0)])
    bayesian_score(DG, data, idx2names, variables)
    plot_graph(DG)
    import pdb;pdb.set_trace()
    pass

def bayesian_score(graph, data, idx2names, variables):
    names2idx = {v: k for k, v in idx2names.items()}
    alpha = prior(variables, graph)
    
def prior(variables,G):
    n=len(variables)
    r=[variables[i] for i in range(n)]
    q=[int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
    return [np.ones((q[i],r[i])) for i in range(n)]

# def statistics(variables,G,df):
#     n=len(df,0)
#     r=[variables[i] for i in range(n)]
#     q=[int(np.prod([r[j] for j in G.predecessors(i)])) for i in range(n)]
#     M=[np.zeros((q[i],r[i])) for i in range(n)]
#     for o in df.itercols():
#         for i in range(n):
#             k=o[i]
#             parents=G.predecessors(i)
#             j=0
#             if not isempty(parents):
#                 j=sub2ind(r[parents],o[parents])
#         M[i][j,k]+=1.0
#     return M

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
