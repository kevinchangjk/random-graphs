import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def block(n, k, chi, zeta, xi):
    
    edges = np.random.ranf([chi, chi])
    edges = (edges + edges.T) / 2
    targets = edges.copy()
    targets = targets < ((0.5))
    g = nx.from_numpy_matrix(targets)
    
    clusters = [[i] for i in range(chi)]
    
    for node in range(chi, n):
        cluster_p = [len(cluster) for cluster in clusters]
        p_tot = sum(cluster_p)
        p_dist = list(np.array(cluster_p) / p_tot)
        cluster = np.random.choice(range(chi), p = p_dist)
        clusters[cluster].append(node)
        clustered = clusters[cluster]
        
        def pcalc(target):
            if target in clustered:
                return (g.degree(target) / node) * zeta
            return xi
        
        p_dist = list(map(pcalc, range(0, node)))
        p_tot = sum(p_dist)
        p_dist2 = np.array(p_dist) / p_tot
        first = np.random.choice(range(node), p = p_dist2)
        g.add_edge(first, node)
        factors = np.random.ranf(node)
        
        def edge_add(i):
            if (factors[i])**(k) < (p_dist[i]):
                return 1
            return 0
        
        targets = list(map(edge_add, range(node)))
        targets = [i if targets[i] == 1 else node for i in range(node)]
        g.add_edges_from(zip(np.full(len(targets), node), targets))
    
    cluster_d = sorted([len(cluster) for cluster in clusters])

    print(f"Number of clusters: {chi}")
    print("Cluster distribution: "+str(cluster_d))
    
    return g

toc = time.time()
g = block(300, 1, 20, 1, 0.001)
tic = time.time()
print(nx.info(g))
print(f"\nGlobal clustering coefficient: {nx.average_clustering(g)}")
print(f"Transitivity: {nx.transitivity(g)}")
print(f"Time taken: {tic-toc}")
nx.draw(g)
plt.show()
plt.clf()

degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
count = Counter(degree_sequence)

plt.bar(count.keys(), count.values(), color='b')
plt.title("Degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.show()
