import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def tan(x):
    return np.tan(x) / np.pi *2

def block_p(n, zeta, xi, chi, phi, clustering):
    edges = np.random.ranf([chi, chi])
    edges = (edges + edges.T) / 2
    targets = edges.copy()
    targets = targets < (xi*2)
    g = nx.from_numpy_matrix(targets)
    clusters = [[i] for i in range(chi)]
    g.add_edges_from([(-1, i) for i in range(chi)])
    
    for node in range(chi, n):
        com = 1
        phit = 1
        g.add_edge(-1, node)
        
        clustering_node = np.random.ranf(1)
        if clustering_node < clustering:
            cluster_p = [1/len(cluster) for cluster in clusters]
            p_tot = sum(cluster_p)
            p_dist = list(np.array(cluster_p) / p_tot)
            cluster = np.random.choice(range(chi), p = p_dist)
            clusters[cluster].append(node)
            clustered = []
            clustered.extend(clusters[cluster])
        
        else:
            clustered = []
        
        def pcalc(target):
            maxi = max(dict(g.degree()).values())
            if target in clustered:
                return phit * zeta * (g.degree(target) / node) ** com
            return phit * xi * (g.degree(target) / node) ** com
        '''
        step = 1
        for target in range(node):
            p = pcalc(target)
            factor = np.random.ranf(1)
            if factor < (p / (step**phi))**com:
                g.add_edge(target, node)
                step+=1
        
        '''
        p_dist = list(map(pcalc, range(0, node)))
        
        factors = np.random.ranf(node)
        
        def edge_add(i):
            if factors[i] < p_dist[i]:
                return 1
            return 0
        
        targets = list(map(edge_add, range(node)))
        targets = [i if targets[i] == 1 else node for i in range(node)]
        g.add_edges_from(zip(np.full(len(targets), node), targets))
        
    g.remove_node(-1)
    print(f"Number of clusters: {chi}")
    return g

toc = time.time()
g = block_p(300, 0.8, 0.02, 10, 5, 1)
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