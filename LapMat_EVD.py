#Spectral Clustering: Laplacian Matrix of the Graph and its Eigen Values and Vectors

import numpy as np
import networkx as nx
import scipy
from scipy.sparse import csr_matrix

#Graph, Nodes and Edges
G=nx.Graph()
G.add_nodes_from(['A','B','C','D','E','F','G','H'])
G.add_edges_from([('A','B'),('A','C'),('A','D'),('A','G'),('B','C'),('B','D'),('C','D'), ('E','F'),('G','E'),('F','G'),('H','G')])
#print(G.nodes())
#print(G.edges())

#Laplacian Matrix
lap = nx.laplacian_matrix(G)
lpm = csr_matrix(lap)
lm = lpm.todense()
print(lm)

#Compute Eigen values and Eigen Vectors
eigval, eigvec = scipy.linalg.eigh(lm)
#print(eigval)
#print(eigvec)

#sorting
sortIndex = eigval.argsort()
eigval = eigval[sortIndex[1]]
eigvec = eigvec[:,sortIndex[1]]

#Display the results
print(eigval)
print(eigvec)
