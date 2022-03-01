import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter


def chang_chang(n, k, phi, chi, delta):
    
    g = nx.Graph()
    
    def post_gen(irrelevant):
        theta, norm = np.random.ranf() * np.pi * 2, np.random.ranf()
        x, y = np.cos(theta) * norm, np.sin(theta) * norm
        return (x, y)
    
    clusters = []
    nodes = 0
    while nodes < n:
        ppln = np.random.randint(1, n * chi)
        if ppln + nodes > n:
            ppln = n - nodes
        cluster = list(range(nodes, nodes+ppln))
        nodes+=ppln
        clusters.append(cluster)
    
    cluster_b = [post_gen(1) for i in clusters]
    
    def place(cluster):
        
        def delt_gen(node):
            delt = (np.random.ranf() / delta, np.random.ranf() / delta)
            post = (cluster_b[i][0] + delt[0], cluster_b[i][1] + delt[1])
            node_gen = (node, dict(pos = post))
            return node_gen
        
        i = clusters.index(cluster)
        nodes_gen = list(map(delt_gen, cluster))
        return nodes_gen
    
    nodes = []
    
    for cluster in clusters:
        nodes.extend(place(cluster))
        
    g.add_nodes_from(nodes)
    post = [(list(nodes[i][1].values())[0][0], list(nodes[i][1].values())[0][1]) for i in range(n)]
    
        
    for node in range(k, n):
        
        def edges(k):
            
            def distance(target):
                dist = ((post[node][0] - post[target][0]) ** 2 + (post[node][1] - post[target][1]) ** 2)**0.5
                return dist
            
            def pcalc(target):
                d = distance(target)
                return ((g.degree(target)+1)**phi) / d
            
            def p_dist(targets):
                dist = np.array(list(map(pcalc, targets)))
                dist = dist / sum(dist)
                return list(dist)
            
            size_n = np.random.randint(1, k)
                
            available = list(range(0, node))
            targets = np.random.choice(available, p = p_dist(available), size = size_n, replace = False)
            edges_add = list(zip(targets, np.full(len(targets), node)))
        
            return edges_add
    
        g.add_edges_from(edges(k))
    
    print(f"\nNo. of clusters: {len(clusters)}")
    return g


toc = time.time()
#   chang_chang(n, k, phi, chi, delta)
g = chang_chang(300, 15, 5, 0.05, 5)
tic = time.time()
print(nx.info(g))
print(f"\nGlobal clustering coefficient: {nx.average_clustering(g)}")
print(f"Transitivity: {nx.transitivity(g)}")
print(f"Time taken: {tic-toc}")
pos = nx.get_node_attributes(g, 'pos')
nx.draw(g, pos)
plt.show()
plt.clf()

degree_sequence = sorted([d for n, d in g.degree()], reverse=True)
count = Counter(degree_sequence)

plt.bar(count.keys(), count.values(), color='b')
plt.title("Degree Histogram")
plt.xlabel("Degree")
plt.ylabel("Number of nodes")
plt.show()