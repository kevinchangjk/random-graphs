import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def block(n, zeta, xi, chi = 0.5):
    groups = []

    while sum(groups) < n:
        ppln = np.random.randint(1, n * chi)
        if ppln + sum(groups) > n:
            ppln = n - sum(groups)
        groups.append(ppln)
        
    edges = np.random.ranf([n, n])
    edges = (edges + edges.T) / 2
    targets = edges.copy()
    targets = targets < xi
    group = 0
    node = 0
    while group < len(groups):
        append = edges[node:node+groups[group], node:node+groups[group]] < zeta
        targets[node:node+groups[group], node:node+groups[group]] = append
        node = node + groups[group]
        group +=1
    
    print(f"Number of clusters: {len(groups)}")
    g = nx.from_numpy_matrix(targets)
    return g




def block_p(n, zeta, xi, chi):
    clusters = [[i] for i in range(chi)]
    g = nx.empty_graph(chi)
    g.add_edges_from([(-1, i) for i in range(chi)])
    
    for node in range(chi, n):
        g.add_edge(-1, node)
        cluster_length = [len(cluster) for cluster in clusters]
        p_dist = [(cluster_length[i] / node)**1 for i in range(chi)]
        factors = np.random.ranf(chi)
        for i in range(chi):
            if factors[i] < p_dist[i]:
                clusters[i].append(node)
        
        clustered = []
        for cluster in clusters:
            if node in cluster:
                clustered.extend(cluster)
        
        def pcalc(target):
            if target in clustered:
                return (g.degree(target) / node)**(0.05/zeta)
            return (g.degree(target) / node)**(0.05/xi)
        
        p_dist = list(map(pcalc, range(node)))
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
g = block_p(300, 0.7, 0.02, 20)
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