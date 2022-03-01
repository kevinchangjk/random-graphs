import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import time
from collections import Counter

def chang(n, zeta):
    g = nx.Graph()
    x = np.random.ranf(n)
    y = np.random.ranf(n)
    post = list(zip(x, y))
    
    def place(node):
        return (node, dict(pos = post[node]))
    
    nodes_add = list(map(place, range(n)))
    g.add_nodes_from(nodes_add)
    
    def edges(zeta):
        edges_final = []
        
        for node in range(n):
            
            def distance(target):
                dist = ((post[node][0] - post[target][0]) ** 2 + (post[node][1] - post[target][1]) ** 2)**0.5
                return dist
            
            distances = list(map(distance, range(node+1, n)))
            p_dist = list(map(lambda d: 1/ d / zeta, distances))
            factor = np.random.ranf(n-node-1)
            
            def edge_set(target):
                if factor[target] < p_dist[target]:
                    return (node, target)
                return 0
            
            edges_add = list(filter(lambda x: x !=0, list(map(edge_set, range(n-node-1)))))
            edges_final.extend(edges_add)
            
        return edges_final
    
    g.add_edges_from(edges(zeta))

    return g

def chang_2(n, k):
    
    g = nx.Graph()
    x = np.random.ranf(n)
    y = np.random.ranf(n)
    post = list(zip(x, y))
    
    def place(node):
        return (node, dict(pos = post[node]))
    
    nodes_add = list(map(place, range(n)))
    g.add_nodes_from(nodes_add)
    
    def edges(k):
        edges_final = []
        
        for node in range(n):
            if node == n-1:
                continue
            
            def distance(target):
                dist = ((post[node][0] - post[target][0]) ** 2 + (post[node][1] - post[target][1]) ** 2)**0.5
                return dist
            
            size_n = np.random.randint(1, k)
            distances = list(map(distance, range(node+1, n)))
            p_dist = np.array(list(map(lambda d: 1/d, distances)))
            p_dist = list(p_dist / sum(p_dist))
            targets = list(np.random.choice(range(node+1, n), size = size_n, p = p_dist))
            edges_add = list(zip(targets, np.full(len(targets), node)))
            edges_final.extend(edges_add)
            
        return edges_final
    
    g.add_edges_from(edges(k))

    return g


def chang_3(n, k, phi):
    
    g = nx.Graph()
    x = np.random.ranf(n)
    y = np.random.ranf(n)
    post = list(zip(x, y))
    
    def place(node):
        return (node, dict(pos = post[node]))
    
    nodes_add = list(map(place, range(n)))
    g.add_nodes_from(nodes_add)
    
    def edges(k):
        edges_final = []
        
        for node in range(n):
            if node == n-1:
                continue
            
            def distance(target):
                dist = ((post[node][0] - post[target][0]) ** 2 + (post[node][1] - post[target][1]) ** 2)**0.5
                return dist
            
            def pcalc(target):
                d = distance(target)
                return (g.degree(target)**phi)/ d
            
            def p_dist(targets):
                dist = np.array(list(map(pcalc, targets)))
                dist = dist / sum(dist)
                return list(dist)
            
            
            size_n = np.random.randint(1, k)
            if size_n > n-node-1:
                size_n = n-node-1
                
            targets = []
            available = list(range(node+1, n))
            while len(targets) < size_n:
                target = np.random.choice(available, p = p_dist(available))
                targets.append(target)
                ind = available.index(target)
                available.pop(ind)
            edges_add = list(zip(targets, np.full(len(targets), node)))
            edges_final.extend(edges_add)
        
        return edges_final
    
    g.add_edges_from(edges(k))

    return g



def chang_4(n, k, phi, chi, delta):
    
    g = nx.Graph()
    
    clusters = []
    nodes = 0
    while nodes < n:
        ppln = np.random.randint(1, n * chi)
        if ppln + nodes > n:
            ppln = n - nodes
        cluster = list(range(nodes, nodes+ppln))
        nodes+=ppln
        clusters.append(cluster)
    
    cluster_b = [(list(np.random.ranf(1))[0], list(np.random.ranf(1))[0]) for i in clusters]
    
    def place(cluster):
        
        def delt_gen(node):
            delt = (list((np.random.ranf(1)))[0] / delta, list((np.random.ranf(1)))[0] / delta)
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
#   chang_4(n, k, phi, chi, delta)
g = chang_4(300, 20, 2, 0.02, 15)
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

