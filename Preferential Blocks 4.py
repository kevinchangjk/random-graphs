import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def block(n, k, chi, zeta, xi, phi):
    
    edges = np.random.ranf([chi, chi])
    edges = (edges + edges.T) / 2
    targets = edges.copy()
    targets = targets < (k / chi)
    g = nx.from_numpy_matrix(targets)
    
    clusters = [[i] for i in range(chi)]
    
    for node in range(chi, n):
        cluster_p = [len(cluster)**0.5 for cluster in clusters]
        p_tot = sum(cluster_p)
        p_dist = list(np.array(cluster_p) / p_tot)
        cluster = np.random.choice(range(chi), p = p_dist)
        clusters[cluster].append(node)
        clustered = clusters[cluster]
        
        def pcalc(target):
            if target in clustered:
                return (g.degree(target)+1)/node ** phi * zeta
            return xi
        
        def p_dis(nodes):
            p_dist = list(map(pcalc, nodes))
            p_tot = sum(p_dist)
            p_dist2 = list(np.array(p_dist) / p_tot)
            return p_dist2
        
        size = np.random.randint(1, k)
        if size > node:
            size = node
        targets = []
        available = list(range(node))
        while len(targets) < size:
            target = np.random.choice(available, p = p_dis(available))
            targets.append(target)
            ind = available.index(target)
            available.pop(ind)
        g.add_edges_from(zip(targets, np.full(len(targets), node)))
        
    cluster_d = sorted([len(cluster) for cluster in clusters])

    print(f"Number of clusters: {chi}")
    print("Cluster distribution: "+str(cluster_d))
    
    return g

toc = time.time()
g = block(300, 10, 20, 100, 1, 100)
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
